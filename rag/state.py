"""Per-machine control-plane store: which corpora exist on this machine.

Kept in a small SQLite file (data/state.db) regardless of where the corpus
data itself lives, so host/join state (OTPs, memberships, join tokens added
in later phases) never mixes with chunk tables and never has to be reachable
from another machine.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rag.config import STATE_DB_PATH

CORPUS_KINDS = ("hosted", "personal")
BACKENDS = ("sqlite", "postgres")

# OTP policy. A 6-digit code is only 10^6 possibilities, so guessing is limited
# per IP, and a burst of failures anywhere revokes every outstanding code.
OTP_TTL_SECONDS = 30 * 60
OTP_MAX_USES = 10
MAX_FAILED_JOINS_PER_IP = 5
FAILED_JOIN_WINDOW_SECONDS = 10 * 60
AUTO_REVOKE_FAILURES = 50
AUTO_REVOKE_WINDOW_SECONDS = 30 * 60


class OtpError(Exception):
    """Join refused (bad / expired / used-up code)."""


class RateLimited(OtpError):
    """Too many failed join attempts from this address."""


@dataclass
class Membership:
    id: int
    corpus_id: str
    display_name: str
    ip: str
    joined_at: str
    last_seen: str


@dataclass
class JoinRecord:
    """A shared corpus this machine has joined. This is ALL that is stored for
    a host: where it is, the membership token, and a few routing keywords --
    never chunks, files or a copy of the database."""
    id: int
    host_url: str
    corpus_name: str
    display_name: str
    token: str
    partition_keywords: Dict[str, List[str]]
    joined_at: str
    status: str = "active"       # 'active' | 'revoked'
    last_error: str = ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Corpus:
    id: str
    name: str
    kind: str                      # 'hosted' (shared by this machine) | 'personal' (this machine's own overlay)
    backend: str                   # 'sqlite' | 'postgres'
    location: str                  # sqlite file path, or postgres schema name
    read_only: bool = False        # attached existing database; never written to
    retained: bool = True          # False = one-shot, removed on next start / exit
    partition_keywords: Dict[str, List[str]] = field(default_factory=dict)
    created_at: str = ""


class StateStore:
    def __init__(self, path: Path | str = STATE_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS corpora ("
                "id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
                "kind TEXT NOT NULL CHECK (kind IN ('hosted','personal')), "
                "backend TEXT NOT NULL CHECK (backend IN ('sqlite','postgres')), "
                "location TEXT NOT NULL, read_only INTEGER NOT NULL DEFAULT 0, "
                "retained INTEGER NOT NULL DEFAULT 1, "
                "partition_keywords_json TEXT NOT NULL DEFAULT '{}', "
                "created_at TEXT NOT NULL)"
            )
            # Member side: shared corpora this machine has joined (no corpus data).
            conn.execute(
                "CREATE TABLE IF NOT EXISTS joins ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, host_url TEXT NOT NULL, corpus_name TEXT NOT NULL, "
                "display_name TEXT NOT NULL, token TEXT NOT NULL, "
                "partition_keywords_json TEXT NOT NULL DEFAULT '{}', joined_at TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'active', last_error TEXT NOT NULL DEFAULT '', "
                "UNIQUE (host_url, corpus_name))"
            )
            # Host-side only: members never receive these tables' contents.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS otp_codes ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, corpus_id TEXT NOT NULL, "
                "code_hash TEXT NOT NULL, salt TEXT NOT NULL, expires_at REAL NOT NULL, "
                "max_uses INTEGER NOT NULL, uses INTEGER NOT NULL DEFAULT 0, "
                "revoked INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memberships ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, corpus_id TEXT NOT NULL, "
                "display_name TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, ip TEXT NOT NULL, "
                "joined_at REAL NOT NULL, last_seen REAL NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS join_attempts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, corpus_id TEXT NOT NULL, "
                "ip TEXT NOT NULL, ts REAL NOT NULL, success INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS security_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, corpus_id TEXT NOT NULL, "
                "ts REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL)"
            )

    # -- corpora -----------------------------------------------------------
    @staticmethod
    def _row_to_corpus(row: sqlite3.Row) -> Corpus:
        return Corpus(
            id=row["id"], name=row["name"], kind=row["kind"], backend=row["backend"],
            location=row["location"], read_only=bool(row["read_only"]),
            retained=bool(row["retained"]),
            partition_keywords=json.loads(row["partition_keywords_json"] or "{}"),
            created_at=row["created_at"],
        )

    def add_corpus(self, name: str, kind: str, backend: str, location: str,
                   read_only: bool = False, retained: bool = True,
                   corpus_id: Optional[str] = None) -> Corpus:
        name = name.strip()
        if not name:
            raise ValueError("Corpus name must not be empty.")
        if kind not in CORPUS_KINDS:
            raise ValueError(f"kind must be one of {CORPUS_KINDS}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        corpus_id = corpus_id or uuid.uuid4().hex[:12]
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO corpora (id, name, kind, backend, location, read_only, retained, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (corpus_id, name, kind, backend, location, int(read_only), int(retained),
                     datetime.now().isoformat(timespec="seconds")),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A corpus named {name!r} already exists.") from exc
        return self.get_corpus(corpus_id)

    def get_corpus(self, corpus_id: str) -> Optional[Corpus]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM corpora WHERE id = ?", (corpus_id,)).fetchone()
        return self._row_to_corpus(row) if row else None

    def find_by_name(self, name: str) -> Optional[Corpus]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM corpora WHERE name = ?", (name.strip(),)).fetchone()
        return self._row_to_corpus(row) if row else None

    def list_corpora(self, kind: Optional[str] = None) -> List[Corpus]:
        sql, params = "SELECT * FROM corpora", ()
        if kind:
            sql, params = sql + " WHERE kind = ?", (kind,)
        with self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY created_at, name", params).fetchall()
        return [self._row_to_corpus(r) for r in rows]

    def set_partition_keywords(self, corpus_id: str, keywords: Dict[str, List[str]]) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE corpora SET partition_keywords_json = ? WHERE id = ?",
                         (json.dumps(keywords), corpus_id))

    def set_retained(self, corpus_id: str, retained: bool) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE corpora SET retained = ? WHERE id = ?", (int(retained), corpus_id))

    def set_kind(self, corpus_id: str, kind: str) -> None:
        if kind not in CORPUS_KINDS:
            raise ValueError(f"kind must be one of {CORPUS_KINDS}")
        with self._conn() as conn:
            conn.execute("UPDATE corpora SET kind = ? WHERE id = ?", (kind, corpus_id))

    def delete_corpus(self, corpus_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))

    # -- OTP codes ------------------------------------------------------------
    def issue_otp(self, corpus_id: str, ttl_seconds: int = OTP_TTL_SECONDS,
                  max_uses: int = OTP_MAX_USES) -> Tuple[str, float]:
        """Create a 6-digit one-time code. Returns (plaintext code, expires_at);
        only a salted hash is stored, so the code cannot be shown again."""
        now = time.time()
        with self._conn() as conn:
            active = conn.execute(
                "SELECT salt, code_hash FROM otp_codes WHERE corpus_id = ? AND revoked = 0 "
                "AND expires_at > ? AND uses < max_uses", (corpus_id, now)).fetchall()
            while True:  # avoid two simultaneously valid identical codes
                code = f"{secrets.randbelow(10 ** 6):06d}"
                if not any(hmac.compare_digest(_sha256(r["salt"] + code), r["code_hash"]) for r in active):
                    break
            salt = secrets.token_hex(8)
            expires_at = now + ttl_seconds
            conn.execute(
                "INSERT INTO otp_codes (corpus_id, code_hash, salt, expires_at, max_uses, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)", (corpus_id, _sha256(salt + code), salt, expires_at, max_uses, now))
        return code, expires_at

    def revoke_all_otps(self, corpus_id: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("UPDATE otp_codes SET revoked = 1 WHERE corpus_id = ? AND revoked = 0", (corpus_id,))
            return cur.rowcount

    def rotate_otp(self, corpus_id: str, ttl_seconds: int = OTP_TTL_SECONDS,
                   max_uses: int = OTP_MAX_USES) -> Tuple[str, float]:
        """Invalidate every outstanding code and issue a fresh one.
        Existing members stay; only kick removes them."""
        self.revoke_all_otps(corpus_id)
        return self.issue_otp(corpus_id, ttl_seconds, max_uses)

    def active_otp_count(self, corpus_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM otp_codes WHERE corpus_id = ? AND revoked = 0 AND expires_at > ? "
                "AND uses < max_uses", (corpus_id, time.time())).fetchone()[0]

    # -- join (redeem) ---------------------------------------------------------
    def redeem_otp(self, corpus_id: str, code: str, display_name: str, ip: str) -> Tuple[str, Membership]:
        """Exchange a valid OTP for a membership token (returned once).
        Raises RateLimited / OtpError. Failures are counted per IP."""
        now = time.time()
        with self._conn() as conn:
            failures = conn.execute(
                "SELECT COUNT(*) FROM join_attempts WHERE corpus_id = ? AND ip = ? AND success = 0 AND ts > ?",
                (corpus_id, ip, now - FAILED_JOIN_WINDOW_SECONDS)).fetchone()[0]
        if failures >= MAX_FAILED_JOINS_PER_IP:
            raise RateLimited("Too many failed attempts; try again later.")

        code = (code or "").strip()
        matched_id = None
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, salt, code_hash FROM otp_codes WHERE corpus_id = ? AND revoked = 0 "
                "AND expires_at > ? AND uses < max_uses", (corpus_id, now)).fetchall()
            for r in rows:
                if hmac.compare_digest(_sha256(r["salt"] + code), r["code_hash"]):
                    # claim a use atomically (guards against concurrent redemptions)
                    cur = conn.execute("UPDATE otp_codes SET uses = uses + 1 WHERE id = ? AND uses < max_uses", (r["id"],))
                    if cur.rowcount:
                        matched_id = r["id"]
                    break
            conn.execute("INSERT INTO join_attempts (corpus_id, ip, ts, success) VALUES (?, ?, ?, ?)",
                         (corpus_id, ip, now, int(matched_id is not None)))
        if matched_id is None:
            self._maybe_auto_revoke(corpus_id, now)
            raise OtpError("Invalid or expired code.")

        name = (display_name or "").strip()[:40] or "member"
        token = secrets.token_urlsafe(32)
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO memberships (corpus_id, display_name, token_hash, ip, joined_at, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?)", (corpus_id, name, _sha256(token), ip, now, now))
            member_id = cur.lastrowid
        return token, self._membership(member_id)

    def _maybe_auto_revoke(self, corpus_id: str, now: float) -> None:
        with self._conn() as conn:
            last = conn.execute("SELECT MAX(ts) FROM security_events WHERE corpus_id = ? AND kind = 'otp_auto_revoked'",
                                (corpus_id,)).fetchone()[0] or 0.0
            since = max(now - AUTO_REVOKE_WINDOW_SECONDS, last)
            failures = conn.execute(
                "SELECT COUNT(*) FROM join_attempts WHERE corpus_id = ? AND success = 0 AND ts > ?",
                (corpus_id, since)).fetchone()[0]
        if failures >= AUTO_REVOKE_FAILURES:
            revoked = self.revoke_all_otps(corpus_id)
            with self._conn() as conn:
                conn.execute("INSERT INTO security_events (corpus_id, ts, kind, detail) VALUES (?, ?, ?, ?)",
                             (corpus_id, now, "otp_auto_revoked",
                              f"{failures} failed join attempts; revoked {revoked} outstanding code(s)."))

    def security_events(self, corpus_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ts, kind, detail FROM security_events WHERE corpus_id = ? "
                                "ORDER BY ts DESC LIMIT ?", (corpus_id, limit)).fetchall()
        return [{"ts": r["ts"], "kind": r["kind"], "detail": r["detail"]} for r in rows]

    # -- members ----------------------------------------------------------------
    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    def _membership(self, member_id: int) -> Membership:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM memberships WHERE id = ?", (member_id,)).fetchone()
        return Membership(r["id"], r["corpus_id"], r["display_name"], r["ip"],
                          self._iso(r["joined_at"]), self._iso(r["last_seen"]))

    def verify_token(self, token: str, corpus_id: Optional[str] = None, touch: bool = True) -> Optional[Membership]:
        """Resolve a bearer token to a live membership (None if unknown or kicked)."""
        if not token:
            return None
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM memberships WHERE token_hash = ?", (_sha256(token),)).fetchone()
            if r is None or (corpus_id is not None and r["corpus_id"] != corpus_id):
                return None
            if touch:
                conn.execute("UPDATE memberships SET last_seen = ? WHERE id = ?", (time.time(), r["id"]))
        return self._membership(r["id"])

    def list_members(self, corpus_id: str) -> List[Membership]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM memberships WHERE corpus_id = ? ORDER BY joined_at", (corpus_id,)).fetchall()
        return [self._membership(r["id"]) for r in rows]

    def kick(self, member_id: int) -> bool:
        """Delete a membership; its token stops working immediately."""
        with self._conn() as conn:
            return conn.execute("DELETE FROM memberships WHERE id = ?", (member_id,)).rowcount > 0

    def remove_all_members(self, corpus_id: str) -> int:
        with self._conn() as conn:
            return conn.execute("DELETE FROM memberships WHERE corpus_id = ?", (corpus_id,)).rowcount

    def leave(self, token: str) -> bool:
        with self._conn() as conn:
            return conn.execute("DELETE FROM memberships WHERE token_hash = ?", (_sha256(token),)).rowcount > 0

    # -- joins (member side) ------------------------------------------------------
    @staticmethod
    def _row_to_join(r: sqlite3.Row) -> JoinRecord:
        return JoinRecord(r["id"], r["host_url"], r["corpus_name"], r["display_name"], r["token"],
                          json.loads(r["partition_keywords_json"] or "{}"), r["joined_at"],
                          r["status"], r["last_error"])

    def add_join(self, host_url: str, corpus_name: str, display_name: str, token: str,
                 partition_keywords: Optional[Dict[str, List[str]]] = None) -> JoinRecord:
        """Record a membership. Re-joining the same host/corpus replaces the old token."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO joins (host_url, corpus_name, display_name, token, partition_keywords_json, joined_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (host_url, corpus_name) DO UPDATE SET "
                "display_name = excluded.display_name, token = excluded.token, "
                "partition_keywords_json = excluded.partition_keywords_json, status = 'active', last_error = ''",
                (host_url, corpus_name, display_name, token, json.dumps(partition_keywords or {}),
                 datetime.now().isoformat(timespec="seconds")))
            row = conn.execute("SELECT * FROM joins WHERE host_url = ? AND corpus_name = ?",
                               (host_url, corpus_name)).fetchone()
        return self._row_to_join(row)

    def get_join(self, join_id: int) -> Optional[JoinRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM joins WHERE id = ?", (join_id,)).fetchone()
        return self._row_to_join(row) if row else None

    def list_joins(self) -> List[JoinRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM joins ORDER BY joined_at, id").fetchall()
        return [self._row_to_join(r) for r in rows]

    def set_join_status(self, join_id: int, status: str, error: str = "") -> None:
        with self._conn() as conn:
            conn.execute("UPDATE joins SET status = ?, last_error = ? WHERE id = ?", (status, error, join_id))

    def remove_join(self, join_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM joins WHERE id = ?", (join_id,))
