"""CorpusManager: create / attach / ingest / delete corpora and cache their engines.

Isolation model: every corpus is its own SQLite file (data/corpora/<id>.db)
or its own Postgres schema (corpus_<id>), so existing DatabaseManager code is
reused unchanged per corpus, deleting a corpus is an unlink / DROP SCHEMA, and
no table needs a corpus_id column. Engines are built lazily and cached (LRU).
"""

from __future__ import annotations

import atexit
import os
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.config import CORPORA_DIR
from rag.db import DatabaseManager
from rag.engine import build_large_corpus_engine
from rag.ingest import DocumentProcessor, IngestionSummary
from rag.routing_autogen import generate_partition_keywords
from rag.state import Corpus, StateStore


class CorpusManager:
    def __init__(self, state: Optional[StateStore] = None, database_url: Optional[str] = None,
                 corpora_dir: Optional[Path | str] = None, max_engines: int = 4,
                 sweep_on_start: bool = True):
        self.state = state or StateStore()
        self.database_url = (database_url if database_url is not None else os.getenv("DATABASE_URL", "")).strip()
        self.corpora_dir = Path(corpora_dir or CORPORA_DIR)
        self.corpora_dir.mkdir(parents=True, exist_ok=True)
        self.max_engines = max_engines
        self._engines: "OrderedDict[str, Tuple[Any, Any, Any]]" = OrderedDict()
        self._lock = threading.RLock()
        self._build_locks: Dict[str, threading.Lock] = {}
        if sweep_on_start:
            self.sweep_one_shot()

    # -- helpers -------------------------------------------------------------
    @property
    def backend(self) -> str:
        return "postgres" if self.database_url else "sqlite"

    def _require(self, corpus_id: str) -> Corpus:
        corpus = self.state.get_corpus(corpus_id)
        if corpus is None:
            raise KeyError(f"Unknown corpus: {corpus_id}")
        return corpus

    def _db_for(self, corpus: Corpus) -> DatabaseManager:
        if corpus.backend == "postgres":
            if not self.database_url:
                raise RuntimeError("This corpus lives in Postgres but DATABASE_URL is not set.")
            return DatabaseManager(database_url=self.database_url, schema=corpus.location)
        return DatabaseManager(db_path=corpus.location, database_url="", read_only=corpus.read_only)

    # -- create / attach / list --------------------------------------------------
    def create_corpus(self, name: str, kind: str = "personal", retained: bool = True) -> Corpus:
        """Create an empty writable corpus (SQLite file or Postgres schema)."""
        import uuid
        corpus_id = uuid.uuid4().hex[:12]
        if self.backend == "postgres":
            corpus = self.state.add_corpus(name, kind, "postgres", f"corpus_{corpus_id}",
                                           retained=retained, corpus_id=corpus_id)
        else:
            corpus = self.state.add_corpus(name, kind, "sqlite",
                                           str(self.corpora_dir / f"{corpus_id}.db"),
                                           retained=retained, corpus_id=corpus_id)
        try:
            self._db_for(corpus)  # creates schema / tables
        except Exception:
            self.state.delete_corpus(corpus.id)
            raise
        return corpus

    def attach_sqlite(self, name: str, path: Path | str, kind: str = "hosted") -> Corpus:
        """Register an existing SQLite database read-only (never copied or modified)."""
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Database file not found: {path}")
        try:
            conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            try:
                has_chunks = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'").fetchone()
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Not a readable SQLite database: {path} ({exc})") from exc
        if not has_chunks:
            raise ValueError(f"{path.name} has no 'chunks' table; not a Hybrid RAG database.")
        return self.state.add_corpus(name, kind, "sqlite", str(path), read_only=True, retained=True)

    def list_corpora(self, kind: Optional[str] = None) -> List[Corpus]:
        return self.state.list_corpora(kind)

    # -- ingest ------------------------------------------------------------------
    def ingest(self, corpus_id: str, file_paths: Sequence[str], force: bool = False) -> IngestionSummary:
        corpus = self._require(corpus_id)
        if corpus.read_only:
            raise PermissionError(f"Corpus {corpus.name!r} is attached read-only; ingestion is disabled.")
        db = self._db_for(corpus)
        summary = DocumentProcessor(db, lazy=True).process_uploads(file_paths, force=force)
        if summary.processed:
            self._fill_missing_keywords(corpus, db)
        self.invalidate(corpus_id)
        return summary

    # -- routing keywords ------------------------------------------------------------
    def _fill_missing_keywords(self, corpus: Corpus, db: DatabaseManager) -> None:
        """Generate keywords for partitions that have none yet. Existing entries are
        kept, so keywords a user edited are never overwritten by a re-ingest."""
        try:
            generated = generate_partition_keywords(db)
        except Exception as exc:
            print(f"Warning: routing keyword generation failed: {exc}")
            return
        current = self.state.get_corpus(corpus.id).partition_keywords
        merged = {**generated, **current}
        if merged != current:
            self.state.set_partition_keywords(corpus.id, merged)

    def regenerate_keywords(self, corpus_id: str) -> Dict[str, List[str]]:
        """Replace all keywords with freshly generated ones (discards manual edits).
        Slow on a very large attached database (entity statistics over millions of rows)."""
        corpus = self._require(corpus_id)
        generated = generate_partition_keywords(self._db_for(corpus))
        self.state.set_partition_keywords(corpus_id, generated)
        self.invalidate(corpus_id)
        return generated

    def set_keywords(self, corpus_id: str, keywords: Dict[str, List[str]]) -> None:
        """Save user-edited routing keywords ({partition_id: [keyword, ...]})."""
        self._require(corpus_id)
        cleaned = {pid: [k.strip().lower() for k in kws if k and k.strip()] for pid, kws in keywords.items()}
        self.state.set_partition_keywords(corpus_id, {pid: kws for pid, kws in cleaned.items() if kws})
        self.invalidate(corpus_id)

    # -- engines -----------------------------------------------------------------
    def get_engine(self, corpus_id: str) -> Tuple[DatabaseManager, DocumentProcessor, Any]:
        """(db, processor, engine) for a corpus, built once and cached.
        The engine carries no API key: pass provider/api_key per call."""
        with self._lock:
            if corpus_id in self._engines:
                self._engines.move_to_end(corpus_id)
                return self._engines[corpus_id]
            build_lock = self._build_locks.setdefault(corpus_id, threading.Lock())
        with build_lock:  # one build per corpus even under concurrent requests
            with self._lock:
                if corpus_id in self._engines:
                    return self._engines[corpus_id]
            corpus = self._require(corpus_id)
            if corpus.backend == "postgres":
                built = build_large_corpus_engine(database_url=self.database_url, schema=corpus.location)
            else:
                built = build_large_corpus_engine(db_path=corpus.location, database_url="",
                                                  read_only=corpus.read_only)
            built[2].set_partition_keywords(corpus.partition_keywords)
            with self._lock:
                self._engines[corpus_id] = built
                while len(self._engines) > self.max_engines:
                    oldest = next(k for k in self._engines if k != "__scratch__")
                    self._engines.pop(oldest)
            return built

    def scratch_engine(self) -> Any:
        """An engine over an empty hidden database. A member who only joined a host
        (no local corpus) needs *an* engine for select/answer; the chunk data comes
        from the request's ChunkStore, so this database stays empty."""
        with self._lock:
            if "__scratch__" not in self._engines:
                path = self.corpora_dir / "_scratch.db"
                self._engines["__scratch__"] = build_large_corpus_engine(db_path=str(path), database_url="")
            self._engines.move_to_end("__scratch__")
            return self._engines["__scratch__"][2]

    def invalidate(self, corpus_id: str) -> None:
        with self._lock:
            self._engines.pop(corpus_id, None)

    # -- delete / one-shot sweep ---------------------------------------------------
    def delete_corpus(self, corpus_id: str) -> None:
        """Delete a corpus' data and registry row. An attached (read-only)
        database file is only unregistered -- the file itself is never touched."""
        corpus = self._require(corpus_id)
        self.invalidate(corpus_id)
        if not corpus.read_only:
            if corpus.backend == "postgres":
                self._db_for(corpus).drop_schema()
            else:
                base = Path(corpus.location)
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        Path(str(base) + suffix).unlink()
                    except FileNotFoundError:
                        pass
        self.state.delete_corpus(corpus_id)

    def sweep_one_shot(self) -> int:
        """Remove corpora that were created as one-shot (retained=False)."""
        removed = 0
        for corpus in self.state.list_corpora():
            if not corpus.retained:
                try:
                    self.delete_corpus(corpus.id)
                    removed += 1
                except Exception as exc:
                    print(f"Could not remove one-shot corpus {corpus.name!r}: {exc}")
        return removed

    def register_exit_sweep(self) -> None:
        def sweep() -> None:
            try:
                self.sweep_one_shot()
            except Exception:
                pass  # data directory already gone (e.g. a temp dir) -- nothing left to sweep
        atexit.register(sweep)
