"""SQLite / PostgreSQL persistence (DatabaseManager)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    ThreadedConnectionPool = None

from rag.config import DATABASE_PATH, PGVECTOR_DIMENSIONS, PROVIDER_PRICING_DEFAULTS, is_running_in_docker
from rag.text import clean_text, extract_entities


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
@dataclass
class ChunkMetadata:
    chunk_id: str
    file_path: str
    partition_id: str
    chunk_index: int
    token_count: int
    entity_count: int


_PG_POOLS: Dict[str, Any] = {}
_PG_POOLS_LOCK = threading.Lock()
_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _get_pg_pool(database_url: str) -> Any:
    """One shared connection pool per Postgres URL (all corpora/schemas on
    that server share it; each checkout selects its own schema)."""
    with _PG_POOLS_LOCK:
        pool = _PG_POOLS.get(database_url)
        if pool is None:
            pool = ThreadedConnectionPool(1, int(os.getenv("PG_POOL_MAX", "10")), dsn=database_url)
            _PG_POOLS[database_url] = pool
        return pool


class DatabaseManager:
    """SQLite or PostgreSQL storage for one corpus.

    schema:    Postgres only -- every corpus lives in its own schema, so
               isolation and deletion need no per-table corpus_id column.
    read_only: SQLite only -- attach an existing database (e.g. the 16 GB
               evaluation DB) without ever writing to it or creating tables.
    """

    CREDIT_PROVIDERS = ("claude", "openai", "gemini")

    def __init__(self, db_path: str = DATABASE_PATH, database_url: str = None,
                 schema: Optional[str] = None, read_only: bool = False):
        self.db_path = db_path
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
        if database_url and is_running_in_docker():
            database_url = database_url.replace("localhost", "host.docker.internal")
        self.database_url = (database_url or "").strip()
        self.use_postgres = bool(self.database_url)
        self.pgvector_enabled = False
        self.read_only = read_only
        self.schema = schema
        if schema is not None and not _SCHEMA_RE.match(schema):
            raise ValueError(f"Invalid schema name: {schema!r}")
        if self.use_postgres and psycopg2 is None:
            raise RuntimeError("DATABASE_URL is set but psycopg2-binary is not installed.")
        if self.use_postgres and read_only:
            raise ValueError("read_only attach is only supported for SQLite corpora.")
        self._init_db()

    @property
    def backend_name(self) -> str:
        return "PostgreSQL + pgvector" if self.use_postgres else "SQLite"

    def _open_sqlite(self) -> sqlite3.Connection:
        if self.read_only:
            uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
        else:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextlib.contextmanager
    def _conn(self):
        """Yield a connection. SQLite: opened and closed per call (cheap and
        thread-safe). Postgres: checked out of the shared pool with this
        corpus's schema selected, then returned."""
        if not self.use_postgres:
            conn = self._open_sqlite()
            try:
                yield conn
            finally:
                conn.close()
            return
        pool = _get_pg_pool(self.database_url)
        conn = pool.getconn()
        try:
            if self.schema:
                with conn.cursor() as cur:
                    cur.execute(f'SET search_path TO "{self.schema}", public')
            yield conn
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            pool.putconn(conn, close=bool(conn.closed))

    def _query(self, query: str, params: Sequence[Any] = (), fetch: bool = False) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor) if self.use_postgres else conn.cursor()
            cursor.execute(query.replace("?", "%s") if self.use_postgres else query, tuple(params))
            rows = [dict(row) for row in cursor.fetchall()] if fetch else []
            conn.commit()
            return rows

    def _many(self, query: str, records: Sequence[Sequence[Any]]) -> None:
        if not records:
            return
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.executemany(query.replace("?", "%s") if self.use_postgres else query, records)
            conn.commit()

    def _init_db(self) -> None:
        if self.read_only:
            return  # attached database: never create tables or change settings
        if self.use_postgres:
            if self.schema:
                self._query(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            try:
                # Always in `public`: created while a corpus schema leads the
                # search_path it would land inside that one corpus, break every
                # other corpus' VECTOR column, and vanish when it is dropped.
                self._query("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
                self.pgvector_enabled = True
            except Exception as exc:
                print(f"pgvector unavailable; storing embeddings as JSON: {exc}")
            vector_column = f"VECTOR({PGVECTOR_DIMENSIONS})" if self.pgvector_enabled else "TEXT"
            statements = [
                """CREATE TABLE IF NOT EXISTS documents (file_path TEXT PRIMARY KEY, file_name TEXT NOT NULL, partition_id TEXT NOT NULL, processed_date TEXT NOT NULL, chunk_count INTEGER NOT NULL, content_hash TEXT)""",
                f"""CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, partition_id TEXT NOT NULL, chunk_text TEXT NOT NULL, token_count INTEGER NOT NULL, entity_count INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding {vector_column})""",
                """CREATE TABLE IF NOT EXISTS comparisons (id BIGSERIAL PRIMARY KEY, query TEXT NOT NULL, timestamp TEXT NOT NULL, method TEXT NOT NULL, latency DOUBLE PRECISION NOT NULL, tokens_used INTEGER NOT NULL, chunks_retrieved INTEGER NOT NULL, relevance_score DOUBLE PRECISION NOT NULL, confidence_score DOUBLE PRECISION NOT NULL, answer TEXT NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS usage (id BIGSERIAL PRIMARY KEY, provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens BIGINT NOT NULL, output_tokens BIGINT NOT NULL, cost_usd DOUBLE PRECISION NOT NULL, created_at TEXT NOT NULL)""",
                """CREATE TABLE IF NOT EXISTS credit_accounts (provider TEXT PRIMARY KEY, initial_usd DOUBLE PRECISION NOT NULL, initial_tokens BIGINT NOT NULL, created_at TEXT NOT NULL)""",
                "CREATE INDEX IF NOT EXISTS idx_chunks_partition ON chunks(partition_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider)",
            ]
            for statement in statements:
                self._query(statement)
            self._ensure_legacy_columns(vector_column)
            if self.pgvector_enabled:
                try:
                    self._query("CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")
                except Exception as exc:
                    print(f"pgvector HNSW index not created: {exc}")
            # Disk-backed lexical search: a generated tsvector column + GIN
            # index. Ranking happens in Postgres via ts_rank_cd, so a query
            # only pulls back matching rows -- the corpus never needs to be
            # loaded into Python to be searched.
            try:
                self._query("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED")
                self._query("CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (chunk_text_tsv)")
            except Exception as exc:
                print(f"Postgres full-text index not created (falling back to LIKE-based search): {exc}")
            self._query("""CREATE TABLE IF NOT EXISTS chunk_entities (chunk_id TEXT NOT NULL, entity TEXT NOT NULL, partition_id TEXT NOT NULL, PRIMARY KEY (chunk_id, entity))""")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity)")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_partition ON chunk_entities(partition_id)")
            self._query("""CREATE TABLE IF NOT EXISTS entity_frequency (entity TEXT PRIMARY KEY, freq INTEGER NOT NULL)""")
            # Per-lexeme document frequency (the Postgres counterpart of the
            # fts5vocab view), refreshed by rebuild_term_df() after ingest.
            self._query("""CREATE TABLE IF NOT EXISTS term_df (term TEXT PRIMARY KEY, df INTEGER NOT NULL)""")
        else:
            # WAL is a persistent property of the database file, so it is set
            # once here rather than on every connection.
            with self._conn() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
            self._query("""CREATE TABLE IF NOT EXISTS documents (file_path TEXT PRIMARY KEY, file_name TEXT NOT NULL, partition_id TEXT NOT NULL, processed_date TEXT NOT NULL, chunk_count INTEGER NOT NULL, content_hash TEXT)""")
            self._query("""CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT NOT NULL, partition_id TEXT NOT NULL, chunk_text TEXT NOT NULL, token_count INTEGER NOT NULL, entity_count INTEGER NOT NULL, chunk_index INTEGER NOT NULL, embedding TEXT)""")
            self._query("""CREATE TABLE IF NOT EXISTS comparisons (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, timestamp TEXT NOT NULL, method TEXT NOT NULL, latency REAL NOT NULL, tokens_used INTEGER NOT NULL, chunks_retrieved INTEGER NOT NULL, relevance_score REAL NOT NULL, confidence_score REAL NOT NULL, answer TEXT NOT NULL)""")
            self._query("""CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY AUTOINCREMENT, provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, created_at TEXT NOT NULL)""")
            self._query("""CREATE TABLE IF NOT EXISTS credit_accounts (provider TEXT PRIMARY KEY, initial_usd REAL NOT NULL, initial_tokens INTEGER NOT NULL, created_at TEXT NOT NULL)""")
            for index in (
                "CREATE INDEX IF NOT EXISTS idx_chunks_partition ON chunks(partition_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)",
                "CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash)",
                "CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider)",
            ):
                self._query(index)
            self._ensure_legacy_columns("TEXT")
            # Disk-backed lexical search: FTS5, so a query only touches the
            # rows it matches instead of requiring every chunk to be
            # tokenized and held in RAM. 'porter' stemming keeps this close
            # in spirit to the existing stem_token()-based TF-IDF path.
            self._query(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                "chunk_id UNINDEXED, partition_id UNINDEXED, chunk_text, "
                "tokenize='porter unicode61')"
            )
            # Live per-term document-frequency view over chunks_fts's own
            # index -- no backfill needed (unlike entity_frequency), since
            # this reads directly from the FTS5 index structure. Used by
            # fts_search to cap how many high-frequency terms go into the
            # OR-joined MATCH query (see MAX_FTS_TERM_FANOUT below): a long
            # question can generate 15-20+ query terms, and BM25 already
            # discounts common terms in scoring, but FTS5 still has to
            # evaluate every chunk containing ANY of them before it can
            # rank and apply LIMIT -- confirmed via profiling to dominate
            # retrieval time on long, cross-domain questions.
            try:
                self._query("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_vocab USING fts5vocab(chunks_fts, 'row')")
            except Exception as exc:
                print(f"fts5vocab not available ({exc}); fts_search will run without term-frequency capping.")
            # Disk-backed entity graph: one row per (chunk, entity), instead
            # of the in-RAM entity_chunks/neighbours sets. partition_id is
            # denormalised onto each row so cross-domain partition detection
            # never needs a join against a full chunk-metadata table.
            self._query(
                "CREATE TABLE IF NOT EXISTS chunk_entities (chunk_id TEXT NOT NULL, "
                "entity TEXT NOT NULL, partition_id TEXT NOT NULL, "
                "PRIMARY KEY (chunk_id, entity))"
            )
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity)")
            self._query("CREATE INDEX IF NOT EXISTS idx_chunk_entities_partition ON chunk_entities(partition_id)")
            # Precomputed per-entity corpus-wide frequency, used by
            # entity_search/get_partition_ids_for_entities to cheaply filter
            # out high-fanout entities before the expensive aggregation --
            # see _filter_high_fanout_entities. Small table (one row per
            # unique entity, not per chunk_entities row), refreshed by
            # rebuild_entity_frequency() after every ingest.
            self._query("""CREATE TABLE IF NOT EXISTS entity_frequency (entity TEXT PRIMARY KEY, freq INTEGER NOT NULL)""")

    def _ensure_legacy_columns(self, vector_type: str) -> None:
        if self.use_postgres:
            rows = self._query("SELECT column_name FROM information_schema.columns WHERE table_name = ? AND table_schema = current_schema()", ("chunks",), fetch=True)
            columns = {row["column_name"] for row in rows}
            if "embedding" not in columns:
                self._query(f"ALTER TABLE chunks ADD COLUMN embedding {vector_type}")
        else:
            rows = self._query("PRAGMA table_info(chunks)", fetch=True)
            columns = {row["name"] for row in rows}
            if "embedding" not in columns:
                self._query("ALTER TABLE chunks ADD COLUMN embedding TEXT")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM documents ORDER BY processed_date DESC", fetch=True)

    def get_all_chunks(self) -> Tuple[Dict[str, str], Dict[str, ChunkMetadata]]:
        rows = self._query("SELECT * FROM chunks ORDER BY file_path, chunk_index", fetch=True)
        texts: Dict[str, str] = {}
        metadata: Dict[str, ChunkMetadata] = {}
        for row in rows:
            texts[row["chunk_id"]] = clean_text(row["chunk_text"])
            metadata[row["chunk_id"]] = ChunkMetadata(row["chunk_id"], row["file_path"], row["partition_id"], int(row["chunk_index"]), int(row["token_count"]), int(row["entity_count"]))
        return texts, metadata

    def get_content_hashes(self) -> Dict[str, str]:
        rows = self._query("SELECT content_hash, file_path FROM documents WHERE content_hash IS NOT NULL", fetch=True)
        return {row["content_hash"]: row["file_path"] for row in rows}

    def save_document(self, file_path: str, partition_id: str, chunks: Dict[str, str], metadata: Dict[str, ChunkMetadata], content_hash: str = "") -> None:
        if self.read_only:
            raise PermissionError("This corpus is attached read-only; ingestion is disabled.")
        canonical_path = str(Path(file_path).resolve())
        with self._conn() as conn:
            cursor = conn.cursor()
            marker = "%s" if self.use_postgres else "?"
            # Re-ingesting a file: also drop the old chunks' FTS / entity rows,
            # otherwise a file that now yields fewer chunks leaves stale hits.
            cursor.execute(f"SELECT chunk_id FROM chunks WHERE file_path = {marker}", (canonical_path,))
            old_ids = [row[0] for row in cursor.fetchall()]
            if old_ids:
                if not self.use_postgres:
                    cursor.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(cid,) for cid in old_ids])
                cursor.executemany(f"DELETE FROM chunk_entities WHERE chunk_id = {marker}", [(cid,) for cid in old_ids])
            cursor.execute(f"DELETE FROM chunks WHERE file_path = {marker}", (canonical_path,))
            if self.use_postgres:
                cursor.execute("""INSERT INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET file_name=EXCLUDED.file_name, partition_id=EXCLUDED.partition_id, processed_date=EXCLUDED.processed_date, chunk_count=EXCLUDED.chunk_count, content_hash=EXCLUDED.content_hash""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            else:
                cursor.execute("""INSERT OR REPLACE INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (?, ?, ?, ?, ?, ?)""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            statement = "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (%s, %s, %s, %s, %s, %s, %s)" if self.use_postgres else "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)"
            cursor.executemany(statement, [(cid, canonical_path, partition_id, text, metadata[cid].token_count, metadata[cid].entity_count, metadata[cid].chunk_index) for cid, text in chunks.items()])

            if not self.use_postgres:
                # SQLite only: mirror into the FTS5 table. (Postgres gets its
                # full-text index for free via the generated tsvector column
                # on `chunks` itself, so nothing extra to insert there.)
                cursor.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(cid,) for cid in chunks.keys()],
                )
                cursor.executemany(
                    "INSERT INTO chunks_fts (chunk_id, partition_id, chunk_text) VALUES (?, ?, ?)",
                    [(cid, partition_id, text) for cid, text in chunks.items()],
                )

            # Entity index (both backends): extract once, at ingest, and
            # store as plain rows -- this is what lets cross-domain routing
            # and graph-style ranking work without ever holding an
            # entity_chunks/neighbours structure in RAM.
            entity_rows = []
            for cid, text in chunks.items():
                for entity in extract_entities(text):
                    entity_rows.append((cid, entity, partition_id))
            del_stmt = "DELETE FROM chunk_entities WHERE chunk_id = %s" if self.use_postgres else "DELETE FROM chunk_entities WHERE chunk_id = ?"
            cursor.executemany(del_stmt, [(cid,) for cid in chunks.keys()])
            if entity_rows:
                ins_stmt = (
                    "INSERT INTO chunk_entities (chunk_id, entity, partition_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING"
                    if self.use_postgres else
                    "INSERT OR IGNORE INTO chunk_entities (chunk_id, entity, partition_id) VALUES (?, ?, ?)"
                )
                cursor.executemany(ins_stmt, entity_rows)

            conn.commit()

    # -------------------------------------------------------------------------
    # Disk-backed search (used by ComparisonEngine in "lazy" / large-corpus
    # mode instead of holding document_tokens/entity_graph in RAM)
    # -------------------------------------------------------------------------
    MAX_FTS_TERM_FANOUT = int(os.getenv("MAX_FTS_TERM_FANOUT", "20000"))
    MAX_FTS_QUERY_TERMS = int(os.getenv("MAX_FTS_QUERY_TERMS", "6"))

    def _cap_fts_terms(self, clean_terms: List[str]) -> List[str]:
        """Keep only the rarest MAX_FTS_QUERY_TERMS terms (by corpus-wide
        document frequency, via the live fts5vocab view), dropping any
        term matching more than MAX_FTS_TERM_FANOUT chunks outright. A
        long question can generate 15-20+ terms after stopword filtering;
        BM25 already discounts common terms in the final ranking, so
        dropping the highest-frequency ones costs little accuracy while
        removing most of the evaluation cost. Falls through to the
        original (uncapped) term list if fts5vocab isn't available or the
        lookup fails, rather than breaking search.
        """
        if len(clean_terms) <= self.MAX_FTS_QUERY_TERMS:
            return clean_terms
        try:
            if self.use_postgres:
                # term_df is keyed by stemmed lexeme; map each raw term to its
                # lexeme with the same english_stem dictionary the index used.
                rows = self._query(
                    "SELECT x.t AS term, d.df AS doc FROM unnest(%s::text[]) AS x(t) "
                    "LEFT JOIN term_df d ON d.term = (ts_lexize('english_stem', x.t))[1]",
                    (list(clean_terms),), fetch=True,
                )
                freq_map = {row["term"]: (row["doc"] or 0) for row in rows}
            else:
                placeholders = ",".join("?" for _ in clean_terms)
                rows = self._query(
                    f"SELECT term, doc FROM chunks_fts_vocab WHERE term IN ({placeholders})",
                    clean_terms, fetch=True,
                )
                freq_map = {row["term"]: row["doc"] for row in rows}
            # Terms not found in the vocab (rare/misspelled/not yet indexed)
            # are treated as maximally rare -- freq 0 -- so they're kept.
            ranked_terms = sorted(clean_terms, key=lambda t: freq_map.get(t, 0))
            kept = [t for t in ranked_terms if freq_map.get(t, 0) <= self.MAX_FTS_TERM_FANOUT]
            return (kept or ranked_terms)[: self.MAX_FTS_QUERY_TERMS]
        except Exception:
            return clean_terms  # chunks_fts_vocab unavailable -- unfiltered behavior

    @staticmethod
    def _pg_tsquery(terms: Sequence[str]) -> str:
        """OR-joined to_tsquery string. plainto_tsquery would AND the terms
        (it ignores the pipe character), which is not the 'any query term'
        behaviour of the SQLite FTS5 path. Terms are reduced to [a-z0-9]
        pieces so no tsquery syntax can be injected; pieces of 4+ chars get a
        prefix match because query_terms() applies a crude suffix-stripper
        that can leave non-words ('statu', 'frag') the english stemmer would
        not match."""
        parts: List[str] = []
        for term in terms:
            for piece in re.findall(r"[A-Za-z0-9]+", term):
                parts.append(piece.lower() + (":*" if len(piece) >= 4 else ""))
        return " | ".join(dict.fromkeys(parts))

    def fts_search(self, terms: Sequence[str], partition_id: Optional[str] = None, limit: int = 50) -> List[Tuple[str, float]]:
        """Lexical search that never loads the corpus into Python -- the
        database does the matching and ranking, and only `limit` results
        (chunk_id, relevance) come back. Higher score = more relevant,
        matching the convention every other ranking method in this file
        uses (unlike FTS5's raw bm25(), which is lower-is-better).
        """
        clean_terms = [t.replace('"', ' ').strip() for t in terms if t and t.strip()]
        if not clean_terms:
            return []
        clean_terms = self._cap_fts_terms(clean_terms)
        if self.use_postgres:
            tsquery = self._pg_tsquery(clean_terms)
            if not tsquery:
                return []
            sql = "SELECT chunk_id, ts_rank_cd(chunk_text_tsv, query) AS score FROM chunks, to_tsquery('english', %s) query WHERE chunk_text_tsv @@ query"
            params: List[Any] = [tsquery]
            if partition_id:
                sql += " AND partition_id = %s"
                params.append(partition_id)
            sql += " ORDER BY score DESC LIMIT %s"
            params.append(limit)
            rows = self._query(sql, params, fetch=True)
            return [(row["chunk_id"], float(row["score"])) for row in rows]
        else:
            match_query = " OR ".join(f'"{t}"' for t in clean_terms)
            sql = "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ?"
            params = [match_query]
            if partition_id:
                sql += " AND partition_id = ?"
                params.append(partition_id)
            sql += " ORDER BY score ASC LIMIT ?"  # FTS5 bm25() is a cost: lower = better match
            params.append(limit)
            rows = self._query(sql, params, fetch=True)
            # Invert sign so "higher = more relevant", consistent with the
            # rest of the ranking pipeline (RRF fusion, calibration, etc).
            return [(row["chunk_id"], -float(row["score"])) for row in rows]

    # -------------------------------------------------------------------------
    # Post-ingest statistics (used to cap high-fanout terms/entities at query time)
    # -------------------------------------------------------------------------
    def rebuild_entity_frequency(self) -> int:
        """Recompute entity_frequency from chunk_entities (one GROUP BY).
        Replaces the old standalone backfill script; run after every ingest."""
        if self.read_only:
            return 0
        self._query("DELETE FROM entity_frequency")
        self._query("INSERT INTO entity_frequency (entity, freq) SELECT entity, COUNT(*) FROM chunk_entities GROUP BY entity")
        return int(self._query("SELECT COUNT(*) AS n FROM entity_frequency", fetch=True)[0]["n"])

    def rebuild_term_df(self) -> int:
        """Postgres only: refresh term_df from the tsvector index (ts_stat).
        SQLite reads the live fts5vocab view, so there is nothing to rebuild."""
        if not self.use_postgres or self.read_only:
            return 0
        self._query("DELETE FROM term_df")
        self._query("INSERT INTO term_df (term, df) SELECT word, ndoc FROM ts_stat('SELECT chunk_text_tsv FROM chunks')")
        return int(self._query("SELECT COUNT(*) AS n FROM term_df", fetch=True)[0]["n"])

    def finalize_ingest(self) -> None:
        """Refresh every derived statistic after documents were added."""
        self.rebuild_entity_frequency()
        self.rebuild_term_df()

    def drop_schema(self) -> None:
        """Postgres only: delete this corpus entirely."""
        if self.use_postgres and self.schema:
            self._query(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')

    MAX_ENTITY_FANOUT = int(os.getenv("MAX_ENTITY_FANOUT", "5000"))

    def _filter_high_fanout_entities(self, entities_lower: List[str]) -> List[str]:
        """Drop entities matching more than MAX_ENTITY_FANOUT chunks
        corpus-wide, via a cheap lookup against the small precomputed
        entity_frequency table (see rebuild_entity_frequency()). Falls
        through to the original list, unfiltered, if that table doesn't
        exist yet or if every given entity happens to be high-frequency
        (searching something over searching nothing).
        """
        try:
            placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities_lower)
            freq_rows = self._query(
                f"SELECT entity, freq FROM entity_frequency WHERE entity IN ({placeholders})",
                entities_lower, fetch=True,
            )
            freq_map = {row["entity"]: row["freq"] for row in freq_rows}
            filtered = [e for e in entities_lower if freq_map.get(e, 0) <= self.MAX_ENTITY_FANOUT]
            return filtered if filtered else entities_lower
        except Exception:
            return entities_lower  # entity_frequency table doesn't exist yet

    def entity_search(self, entities: Iterable[str], partition_id: Optional[str] = None, limit: int = 100) -> List[Tuple[str, float]]:
        """Graph-style ranking without an in-RAM entity_chunks/neighbours
        structure: score a chunk by how many of the query's entities it
        also contains, via a single indexed SQL query.

        Entities are first filtered against entity_frequency (a small,
        precomputed table -- see rebuild_entity_frequency()) to drop
        anything matching more than MAX_ENTITY_FANOUT chunks corpus-wide.
        This is the general form of the "api"/"what"/"2026" problem found
        via profiling: a handful of specific words are now excluded at
        the stopword level, but any OTHER unexpectedly common term would
        cause the same expensive GROUP BY over hundreds of thousands of
        rows for near-zero discriminative benefit. Checking frequency via
        a primary-key lookup against a small table is cheap regardless of
        how large chunk_entities itself is; if entity_frequency hasn't
        been backfilled yet, this filter silently no-ops (falls through
        to the original, unfiltered behavior) rather than breaking search.
        """
        entities = [e for e in entities if e]
        if not entities:
            return []
        entities_lower = self._filter_high_fanout_entities([e.lower() for e in entities])

        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities_lower)
        sql = f"SELECT chunk_id, COUNT(DISTINCT entity) AS matches FROM chunk_entities WHERE entity IN ({placeholders})"
        params: List[Any] = list(entities_lower)
        if partition_id:
            sql += (" AND partition_id = %s" if self.use_postgres else " AND partition_id = ?")
            params.append(partition_id)
        sql += " GROUP BY chunk_id ORDER BY matches DESC"
        sql += (" LIMIT %s" if self.use_postgres else " LIMIT ?")
        params.append(limit)
        rows = self._query(sql, params, fetch=True)
        max_matches = max((row["matches"] for row in rows), default=1) or 1
        return [(row["chunk_id"], row["matches"] / max_matches) for row in rows]

    def get_partition_ids_for_entities(self, entities: Iterable[str]) -> List[str]:
        """Rank partitions by entity DENSITY, not raw hit count.

        Density = hits / partition_size. This stops big partitions (Slack
        800K chunks) from drowning out small relevant ones (Jira 25K)
        just because they have more total mentions of a common term.
        """
        entities = [e.lower() for e in entities if e]
        if not entities:
            return []
        entities = self._filter_high_fanout_entities(entities)
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities)
        sql = (
            f"SELECT ce.partition_id, "
            f"COUNT(*) AS hits, "
            f"pc.total_chunks, "
            f"1.0 * COUNT(*) / pc.total_chunks AS density "
            f"FROM chunk_entities ce "
            f"JOIN (SELECT partition_id, COUNT(*) AS total_chunks FROM chunks GROUP BY partition_id) pc "
            f"ON ce.partition_id = pc.partition_id "
            f"WHERE ce.entity IN ({placeholders}) "
            f"GROUP BY ce.partition_id, pc.total_chunks "
            f"ORDER BY density DESC LIMIT 50"
        )
        rows = self._query(sql, entities, fetch=True)
        return [row["partition_id"] for row in rows]


    def get_chunk_texts(self, chunk_ids: Sequence[str]) -> Dict[str, str]:
        """Fetch text for a small, specific set of chunks (a router's
        candidate list, never the whole corpus)."""
        chunk_ids = list({cid for cid in chunk_ids if cid})
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in chunk_ids)
        rows = self._query(f"SELECT chunk_id, chunk_text FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids, fetch=True)
        return {row["chunk_id"]: clean_text(row["chunk_text"]) for row in rows}

    def get_chunk_metadata_batch(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkMetadata]:
        """Fetch metadata for a small, specific set of chunks."""
        chunk_ids = list({cid for cid in chunk_ids if cid})
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in chunk_ids)
        rows = self._query(
            f"SELECT chunk_id, file_path, partition_id, chunk_index, token_count, entity_count "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids, fetch=True,
        )
        return {
            row["chunk_id"]: ChunkMetadata(
                row["chunk_id"], row["file_path"], row["partition_id"],
                int(row["chunk_index"]), int(row["token_count"]), int(row["entity_count"]),
            )
            for row in rows
        }

    def get_partition_ids(self) -> List[str]:
        """Distinct partition ids (index-only scan; ~2.5 s cold on the 1.9M-chunk DB)."""
        return [row["partition_id"] for row in self._query(
            "SELECT DISTINCT partition_id FROM chunks ORDER BY partition_id", fetch=True)]

    def get_partition_counts(self) -> Dict[str, int]:
        """Cheap partition sizes (a GROUP BY COUNT, not a full row fetch) --
        used for router/context stats without materialising chunk-id lists."""
        rows = self._query("SELECT partition_id, COUNT(*) AS n FROM chunks GROUP BY partition_id", fetch=True)
        return {row["partition_id"]: int(row["n"]) for row in rows}

    def save_embeddings(self, embeddings: Dict[str, Any]) -> None:
        if not embeddings or self.read_only:
            return
        records = []
        for chunk_id, vector in embeddings.items():
            values = vector.detach().cpu().tolist() if hasattr(vector, "detach") else list(vector)
            records.append((json.dumps(values), chunk_id))
        if self.use_postgres and self.pgvector_enabled:
            self._many("UPDATE chunks SET embedding = ?::vector WHERE chunk_id = ?", records)
        else:
            self._many("UPDATE chunks SET embedding = ? WHERE chunk_id = ?", records)

    def save_comparison(self, query: str, results: Dict[str, Dict[str, Any]]) -> None:
        if self.read_only:
            return
        records = [(query, datetime.now().isoformat(timespec="seconds"), method, float(data.get("latency", 0)), int(data.get("tokens_used", 0)), int(data.get("chunks_retrieved", 0)), float(data.get("relevance_score", 0)), float(data.get("confidence_score", 0)), str(data.get("answer", ""))) for method, data in results.items()]
        self._many("INSERT INTO comparisons (query, timestamp, method, latency, tokens_used, chunks_retrieved, relevance_score, confidence_score, answer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)

    @staticmethod
    def _credit_prefix(provider: str) -> str:
        return {"claude": "CLAUDE", "openai": "OPENAI", "gemini": "GEMINI"}[provider]

    def get_credit_status(self, provider: str) -> Dict[str, Any]:
        prefix = self._credit_prefix(provider)
        usd_raw, token_raw = os.getenv(f"{prefix}_CREDITS_USD"), os.getenv(f"{prefix}_CREDITS_TOKENS")
        configured_in_env = usd_raw not in (None, "") or token_raw not in (None, "")
        account_rows = self._query("SELECT initial_usd, initial_tokens FROM credit_accounts WHERE provider = ?", (provider,), fetch=True)
        if not account_rows and configured_in_env and not self.read_only:
            self._query("INSERT INTO credit_accounts (provider, initial_usd, initial_tokens, created_at) VALUES (?, ?, ?, ?)", (provider, float(usd_raw or 0.0), int(token_raw or 0), datetime.now().isoformat(timespec="seconds")))
            account_rows = self._query("SELECT initial_usd, initial_tokens FROM credit_accounts WHERE provider = ?", (provider,), fetch=True)
        configured = bool(account_rows)
        initial_usd = float(account_rows[0]["initial_usd"]) if account_rows else 0.0
        initial_tokens = int(account_rows[0]["initial_tokens"]) if account_rows else 0
        rows = self._query("SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens, COALESCE(SUM(output_tokens), 0) AS output_tokens, COALESCE(SUM(cost_usd), 0) AS used_usd FROM usage WHERE provider = ?", (provider,), fetch=True)
        used = rows[0]
        used_tokens = int(used["input_tokens"] or 0) + int(used["output_tokens"] or 0)
        return {"provider": provider, "configured": configured, "initial_usd": initial_usd, "initial_tokens": initial_tokens, "used_usd": float(used["used_usd"] or 0), "used_tokens": used_tokens, "remaining_usd": max(0.0, initial_usd - float(used["used_usd"] or 0)), "remaining_tokens": max(0, initial_tokens - used_tokens)}

    def get_all_credit_statuses(self) -> List[Dict[str, Any]]:
        return [self.get_credit_status(provider) for provider in self.CREDIT_PROVIDERS]

    def can_generate(self, provider: str, input_tokens: int, max_output_tokens: int) -> Tuple[bool, str]:
        status = self.get_credit_status(provider)
        if not status["configured"]:
            return True, "Credit limit is not configured."
        estimated_cost = estimate_provider_cost(provider, input_tokens, max_output_tokens)
        if status["remaining_usd"] < estimated_cost:
            return False, f"{provider.title()} credits are exhausted or insufficient (${status['remaining_usd']:.4f} remaining)."
        if status["remaining_tokens"] < input_tokens + max_output_tokens:
            return False, f"{provider.title()} token credits are exhausted or insufficient ({status['remaining_tokens']:,} remaining)."
        return True, ""

    def record_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = estimate_provider_cost(provider, input_tokens, output_tokens)
        if self.read_only:
            return cost
        self._query("INSERT INTO usage (provider, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?)", (provider, model, input_tokens, output_tokens, cost, datetime.now().isoformat(timespec="seconds")))
        return cost


def estimate_provider_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    prefix = {"claude": "CLAUDE", "openai": "OPENAI", "gemini": "GEMINI"}.get(provider, "OPENAI")
    default_input, default_output = PROVIDER_PRICING_DEFAULTS.get(provider, PROVIDER_PRICING_DEFAULTS["openai"])
    input_price = float(os.getenv(f"{prefix}_INPUT_PRICE", str(default_input)))
    output_price = float(os.getenv(f"{prefix}_OUTPUT_PRICE", str(default_output)))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
