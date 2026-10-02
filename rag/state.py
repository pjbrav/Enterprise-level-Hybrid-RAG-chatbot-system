"""Per-machine control-plane store: which corpora exist on this machine.

Kept in a small SQLite file (data/state.db) regardless of where the corpus
data itself lives, so host/join state (OTPs, memberships, join tokens added
in later phases) never mixes with chunk tables and never has to be reachable
from another machine.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from rag.config import STATE_DB_PATH

CORPUS_KINDS = ("hosted", "personal")
BACKENDS = ("sqlite", "postgres")


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
