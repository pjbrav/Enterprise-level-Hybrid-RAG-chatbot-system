"""
Hybrid RAG v22 – Universal Enterprise Edition
==============================================

- No hardcoded domains: partitions are derived from the folder structure of uploaded files.
- Generic entity extraction (capitals, acronyms, numbers) and co‑occurrence graph.
- Same core retrieval: TF‑IDF, Graph, RRF, optional embeddings/cross‑encoder.
- Fully backwards‑compatible with existing SQLite/PostgreSQL schemas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Environment & optional imports
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

_EMBEDDING_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.getenv("SQLITE_DATABASE_PATH", "hybrid_rag_v22.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".js", ".ipynb"}
MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "250"))
MAX_ZIP_MEMBER_BYTES = int(os.getenv("MAX_ZIP_MEMBER_BYTES", str(25 * 1024 * 1024)))
PGVECTOR_DIMENSIONS = int(os.getenv("PGVECTOR_DIMENSIONS", "384"))
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
CHUNK_WORDS = 220
CHUNK_OVERLAP = 25
MAX_CHUNKS = 5
MAX_CROSS_DOMAIN_CHUNKS = 3
MIN_CHUNKS_PER_ACTIVE_PARTITION = 1
MAX_CROSS_DOMAIN_CHUNKS_CEILING = 6
MAX_ANSWER_TOKENS = 3000
CONTEXT_CHARS_PER_CHUNK = 1100

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CROSS_ENCODER_RERANK_TOP_K = 10

PROVIDER_PRICING_DEFAULTS = {
    "claude": (0.80, 2.40),
    "openai": (1.00, 6.00),
    "gemini": (0.15, 0.60),
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for",
    "with", "on", "at", "from", "by", "in", "and", "or", "but", "as",
    "what", "why", "how", "when", "where", "who", "which", "does", "do",
    "compare", "comparison", "contrast", "explain", "describe", "analyze",
    "analyse", "evaluate", "between", "both", "versus", "than", "into",
}

# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------
def is_running_in_docker() -> bool:
    return os.path.exists('/.dockerenv')

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def estimate_tokens(text: str) -> int:
    if text and tiktoken is not None:
        try:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            pass
    return math.ceil(len(text) / 4) if text else 0

def stem_token(token: str) -> str:
    token = token.lower().strip("-_")
    for suffix in ("izations", "ization", "ations", "ation", "ments", "ment", "ingly", "ing", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return token

def query_terms(query: str) -> List[str]:
    return [
        stem_token(term) for term in re.findall(r"\b[\w-]+\b", query.lower())
        if len(term) > 2 and term not in STOPWORDS
    ]

def extract_entities(text: str) -> Set[str]:
    """Generic entity extraction: capitalized phrases, acronyms, and numbers."""
    # Capitalized multi-word sequences and acronyms
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*)*\b", text)
    entities = set()
    for c in candidates:
        c = c.strip()
        if len(c) >= 3:
            entities.add(c)
    # Also add numbers with units (e.g., "$10,000", "2,800 psi")
    number_pattern = r"\b[\$€]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:psi|USD|EUR|bar|%|k|M)?\b"
    numbers = re.findall(number_pattern, text)
    for num in numbers:
        if num.strip():
            entities.add(num.strip())
    return {e for e in entities if len(e) > 2}

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


class DatabaseManager:
    CREDIT_PROVIDERS = ("claude", "openai", "gemini")

    def __init__(self, db_path: str = DATABASE_PATH, database_url: str = None):
        self.db_path = db_path
        if database_url is None:
            database_url = os.getenv("DATABASE_URL")
        if database_url and is_running_in_docker():
            database_url = database_url.replace("localhost", "host.docker.internal")
        self.database_url = (database_url or "").strip()
        self.use_postgres = bool(self.database_url)
        self.pgvector_enabled = False
        if self.use_postgres and psycopg2 is None:
            raise RuntimeError("DATABASE_URL is set but psycopg2-binary is not installed.")
        self._init_db()

    @property
    def backend_name(self) -> str:
        return "PostgreSQL + pgvector" if self.use_postgres else "SQLite"

    def _connect(self) -> Any:
        if self.use_postgres:
            return psycopg2.connect(self.database_url)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, query: str, params: Sequence[Any] = (), fetch: bool = False) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor) if self.use_postgres else conn.cursor()
            cursor.execute(query.replace("?", "%s") if self.use_postgres else query, tuple(params))
            rows = [dict(row) for row in cursor.fetchall()] if fetch else []
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _many(self, query: str, records: Sequence[Sequence[Any]]) -> None:
        if not records:
            return
        conn = self._connect()
        try:
            cursor = conn.cursor()
            cursor.executemany(query.replace("?", "%s") if self.use_postgres else query, records)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        if self.use_postgres:
            try:
                self._query("CREATE EXTENSION IF NOT EXISTS vector")
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
        else:
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

    def _ensure_legacy_columns(self, vector_type: str) -> None:
        if self.use_postgres:
            rows = self._query("SELECT column_name FROM information_schema.columns WHERE table_name = ?", ("chunks",), fetch=True)
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
        canonical_path = str(Path(file_path).resolve())
        conn = self._connect()
        try:
            cursor = conn.cursor()
            marker = "%s" if self.use_postgres else "?"
            cursor.execute(f"DELETE FROM chunks WHERE file_path = {marker}", (canonical_path,))
            if self.use_postgres:
                cursor.execute("""INSERT INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET file_name=EXCLUDED.file_name, partition_id=EXCLUDED.partition_id, processed_date=EXCLUDED.processed_date, chunk_count=EXCLUDED.chunk_count, content_hash=EXCLUDED.content_hash""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            else:
                cursor.execute("""INSERT OR REPLACE INTO documents (file_path, file_name, partition_id, processed_date, chunk_count, content_hash) VALUES (?, ?, ?, ?, ?, ?)""", (canonical_path, Path(canonical_path).name, partition_id, datetime.now().isoformat(timespec="seconds"), len(chunks), content_hash))
            statement = "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (%s, %s, %s, %s, %s, %s, %s)" if self.use_postgres else "INSERT INTO chunks (chunk_id, file_path, partition_id, chunk_text, token_count, entity_count, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)"
            cursor.executemany(statement, [(cid, canonical_path, partition_id, text, metadata[cid].token_count, metadata[cid].entity_count, metadata[cid].chunk_index) for cid, text in chunks.items()])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_embeddings(self, embeddings: Dict[str, Any]) -> None:
        if not embeddings:
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
        if not account_rows and configured_in_env:
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
        self._query("INSERT INTO usage (provider, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?, ?, ?, ?, ?, ?)", (provider, model, input_tokens, output_tokens, cost, datetime.now().isoformat(timespec="seconds")))
        return cost


def estimate_provider_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    prefix = {"claude": "CLAUDE", "openai": "OPENAI", "gemini": "GEMINI"}.get(provider, "OPENAI")
    default_input, default_output = PROVIDER_PRICING_DEFAULTS.get(provider, PROVIDER_PRICING_DEFAULTS["openai"])
    input_price = float(os.getenv(f"{prefix}_INPUT_PRICE", str(default_input)))
    output_price = float(os.getenv(f"{prefix}_OUTPUT_PRICE", str(default_output)))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestionSummary:
    processed: int = 0
    duplicates: List[str] = None
    skipped: List[str] = None
    failures: List[str] = None
    partitions: Dict[str, int] = None

    def __post_init__(self) -> None:
        self.duplicates = self.duplicates or []
        self.skipped = self.skipped or []
        self.failures = self.failures or []
        self.partitions = self.partitions or {}

    def as_markdown(self) -> str:
        lines = [f"**Ingestion complete:** {self.processed} file(s) indexed."]
        if self.partitions:
            lines.append("Partitions: " + ", ".join(f"`{name.replace('partition_', '')}`: {count}" for name, count in sorted(self.partitions.items())))
        if self.duplicates:
            lines.append("**Duplicates ignored:** " + ", ".join(self.duplicates))
        if self.skipped:
            lines.append("**Unsupported/skipped:** " + ", ".join(self.skipped))
        if self.failures:
            lines.append("**Could not ingest:** " + "; ".join(self.failures))
        return "\n\n".join(lines)


class DocumentProcessor:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.chunk_texts, self.chunks_metadata = self.db.get_all_chunks()

    @staticmethod
    def _load_document(file_path: str) -> str:
        path = Path(file_path.strip().strip('"\''))
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
            except ImportError as exc:
                raise ImportError("PDF support requires: pip install pymupdf") from exc
            with fitz.open(path) as pdf:
                return clean_text("\n".join(page.get_text() for page in pdf))
        if suffix in {".txt", ".md", ".py", ".js"}:
            return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                cells = notebook.get("cells", [])
                return clean_text("\n\n".join("".join(cell.get("source", [])) if isinstance(cell.get("source", []), list) else str(cell.get("source", "")) for cell in cells if cell.get("cell_type") in {"markdown", "code"}))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid notebook JSON: {path.name}") from exc
        raise ValueError(f"Unsupported file type: {suffix}")

    _CITATION_MARKER_RE = re.compile(r"\[\d{1,3}(?:,\s*\d{1,3})*\]|\(\d{4}[a-z]?\)|et al\.")

    @classmethod
    def _is_non_content_chunk(cls, text: str) -> bool:
        opening = text[:180].lower().strip()
        if (
            len(text.split()) < 25
            or opening.startswith("references")
            or opening.startswith("bibliography")
            or opening.startswith("acknowledg")
        ):
            return True
        words = text.split()
        if not words:
            return True
        markers = cls._CITATION_MARKER_RE.findall(text)
        citation_density_per_100_words = (len(markers) / max(1, len(words))) * 100
        if citation_density_per_100_words >= 12.0:
            return True
        return False

    def _classify_document(self, text: str, file_path: str) -> str:
        """
        Assigns a partition ID based on the immediate parent folder of the file.
        If the file is in the root, partition = 'general'.
        """
        path = Path(file_path)
        parent = path.parent.name
        if parent and parent != ".":
            return f"partition_{parent}"
        return "partition_general"

    def _chunk_document(
        self, text: str, source_path: str, partition_id: str
    ) -> Tuple[Dict[str, str], Dict[str, ChunkMetadata]]:
        words = clean_text(text).split()
        chunks: Dict[str, str] = {}
        metadata: Dict[str, ChunkMetadata] = {}
        source_hash = hashlib.sha1(str(Path(source_path).resolve()).encode("utf-8")).hexdigest()[:12]
        step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
        chunk_number = 0
        for start in range(0, len(words), step):
            chunk_words = words[start:start + CHUNK_WORDS]
            if len(chunk_words) < 25:
                continue
            chunk_text = " ".join(chunk_words)
            if self._is_non_content_chunk(chunk_text):
                continue
            chunk_number += 1
            chunk_id = f"{source_hash}_{chunk_number:04d}"
            entities = extract_entities(chunk_text)
            chunks[chunk_id] = chunk_text
            metadata[chunk_id] = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=str(Path(source_path).resolve()),
                partition_id=partition_id,
                chunk_index=chunk_number,
                token_count=estimate_tokens(chunk_text),
                entity_count=len(entities),
            )
            if start + CHUNK_WORDS >= len(words):
                break
        return chunks, metadata

    def _extract_zip(self, archive_path: Path, summary: IngestionSummary) -> List[Path]:
        try:
            if not zipfile.is_zipfile(archive_path):
                summary.failures.append(f"{archive_path.name} (invalid ZIP archive)")
                return []
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()[:16]
            destination = (UPLOAD_DIR / f"{archive_path.stem}_{archive_hash}").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            extracted: List[Path] = []
            with zipfile.ZipFile(archive_path) as archive:
                members = [member for member in archive.infolist() if not member.is_dir()]
                if len(members) > MAX_ZIP_FILES:
                    summary.failures.append(f"{archive_path.name} (contains {len(members)} files; limit is {MAX_ZIP_FILES})")
                    return []
                for member in members:
                    member_path = Path(member.filename)
                    suffix = member_path.suffix.lower()
                    if suffix not in SUPPORTED_EXTENSIONS:
                        summary.skipped.append(f"{member.filename} (unsupported {suffix or 'file type'})")
                        continue
                    if member.file_size > MAX_ZIP_MEMBER_BYTES:
                        summary.skipped.append(f"{member.filename} (larger than {MAX_ZIP_MEMBER_BYTES // (1024 * 1024)} MB limit)")
                        continue
                    target = (destination / member_path).resolve()
                    if destination not in target.parents:
                        summary.skipped.append(f"{member.filename} (unsafe archive path)")
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("wb") as output:
                        output.write(source.read())
                    extracted.append(target)
            return extracted
        except (OSError, zipfile.BadZipFile) as exc:
            summary.failures.append(f"{archive_path.name} ({exc})")
            return []

    def _process_document(self, path: Path, existing_hashes: Dict[str, str], force: bool, summary: IngestionSummary) -> None:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            summary.skipped.append(f"{path.name} (unsupported {path.suffix or 'file type'})")
            return
        if not path.exists():
            summary.failures.append(f"{path.name} (file not found)")
            return
        canonical_path = str(path.resolve())
        try:
            file_bytes = path.read_bytes()
            content_hash = hashlib.sha256(file_bytes).hexdigest()
            if not force and content_hash in existing_hashes:
                summary.duplicates.append(f"{path.name} (same as {Path(existing_hashes[content_hash]).name})")
                return
            text = self._load_document(canonical_path)
            if not text:
                summary.skipped.append(f"{path.name} (empty)")
                return
            partition = self._classify_document(text, canonical_path)
            chunks, metadata = self._chunk_document(text, canonical_path, partition)
            if not chunks:
                summary.skipped.append(f"{path.name} (no usable content chunks)")
                return
            old_chunk_ids = [chunk_id for chunk_id, old_metadata in self.chunks_metadata.items() if old_metadata.file_path == canonical_path]
            for chunk_id in old_chunk_ids:
                self.chunk_texts.pop(chunk_id, None)
                self.chunks_metadata.pop(chunk_id, None)
            self.db.save_document(canonical_path, partition, chunks, metadata, content_hash=content_hash)
            self.chunk_texts.update(chunks)
            self.chunks_metadata.update(metadata)
            existing_hashes[content_hash] = canonical_path
            summary.processed += 1
            summary.partitions[partition] = summary.partitions.get(partition, 0) + 1
        except Exception as exc:
            summary.failures.append(f"{path.name} ({exc})")

    def process_uploads(self, file_paths: Sequence[str], force: bool = False) -> IngestionSummary:
        existing_hashes = self.db.get_content_hashes()
        summary = IngestionSummary()
        for raw_path in file_paths:
            path = Path(raw_path.strip().strip('"\''))
            if path.suffix.lower() == ".zip":
                for extracted_path in self._extract_zip(path, summary):
                    self._process_document(extracted_path, existing_hashes, force, summary)
                continue
            self._process_document(path, existing_hashes, force, summary)
        return summary

    def process_documents(self, file_paths: Sequence[str], force: bool = False) -> int:
        summary = self.process_uploads(file_paths, force=force)
        print(summary.as_markdown())
        return summary.processed


def select_files_dialog() -> List[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        files = filedialog.askopenfilenames(
            title="Select research papers",
            filetypes=[("Supported research files", "*.pdf *.txt *.md *.py *.js *.ipynb *.zip"), ("All files", "*.*")],
        )
        root.destroy()
        return list(files)
    except Exception as exc:
        print(f"File picker unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# Retrieval primitives
# ---------------------------------------------------------------------------
def build_tfidf_index(chunks: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    document_tokens = {
        chunk_id: [stem_token(token) for token in re.findall(r"\b[\w-]+\b", text.lower())]
        for chunk_id, text in chunks.items() if text
    }
    frequencies: Dict[str, int] = defaultdict(int)
    for tokens in document_tokens.values():
        for term in set(tokens):
            frequencies[term] += 1
    document_count = max(1, len(document_tokens))
    idf = {
        term: math.log((document_count + 1) / (frequency + 1)) + 1
        for term, frequency in frequencies.items()
    }
    return document_tokens, idf


def vector_rank(
    documents: Dict[str, List[str]], idf: Dict[str, float], terms: Sequence[str]
) -> List[Tuple[str, float]]:
    if not documents or not terms:
        return []
    query_tf = Counter(terms)
    query_vector = {
        term: count * idf.get(term, 1.0)
        for term, count in query_tf.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    scores = []
    for chunk_id, tokens in documents.items():
        term_frequency = Counter(tokens)
        dot = sum(
            term_frequency.get(term, 0) * idf.get(term, 1.0) * query_weight
            for term, query_weight in query_vector.items()
        )
        document_norm = math.sqrt(sum(
            (count * idf.get(term, 1.0)) ** 2
            for term, count in term_frequency.items()
        )) or 1.0
        scores.append((chunk_id, dot / (query_norm * document_norm)))
    return sorted(scores, key=lambda item: item[1], reverse=True)


class EntityKnowledgeGraph:
    def __init__(self, chunks: Dict[str, str]):
        self.chunk_entities: Dict[str, Set[str]] = {}
        self.entity_chunks: Dict[str, Set[str]] = defaultdict(set)
        self.neighbours: Dict[str, Set[str]] = defaultdict(set)
        for chunk_id, text in chunks.items():
            entities = extract_entities(text)
            self.chunk_entities[chunk_id] = entities
            for entity in entities:
                self.entity_chunks[entity].add(chunk_id)
                self.neighbours[entity].update(entities - {entity})

    def rank(self, query: str, candidate_ids: Iterable[str]) -> List[Tuple[str, float]]:
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        ranked = []
        for chunk_id in candidate_ids:
            entities = self.chunk_entities.get(chunk_id, set())
            direct = (len(entities & query_entities) / len(query_entities)) if query_entities else 0.0
            connected = (
                sum(
                    any(
                        entity in self.neighbours.get(query_entity, set())
                        for query_entity in query_entities
                    )
                    for entity in entities
                ) / max(1, len(entities))
                if query_entities else 0.0
            )
            score = (0.75 * direct) + (0.25 * connected)
            if score > 0:
                ranked.append((chunk_id, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)


# ---------------------------------------------------------------------------
# Embedding-based semantic ranker (optional)
# ---------------------------------------------------------------------------
class EmbeddingRanker:
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.model = None
        self.db = db
        self.chunk_embeddings: Dict[str, Any] = {}
        if _EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
            except Exception as e:
                print(f"⚠️ Failed to load embedding model: {e}")
                self.model = None

    def build_index(self, chunk_texts: Dict[str, str]) -> None:
        if self.model is None:
            return
        if not chunk_texts:
            return
        texts = list(chunk_texts.values())
        chunk_ids = list(chunk_texts.keys())
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            self.chunk_embeddings = {cid: emb for cid, emb in zip(chunk_ids, embeddings)}
            if self.db is not None:
                self.db.save_embeddings(self.chunk_embeddings)
        except Exception as e:
            print(f"⚠️ Embedding indexing failed: {e}")
            self.chunk_embeddings = {}

    def rank(self, query: str, candidate_ids: Iterable[str]) -> List[Tuple[str, float]]:
        if self.model is None or not self.chunk_embeddings:
            return []
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = []
        for cid in candidate_ids:
            if cid not in self.chunk_embeddings:
                continue
            sim = (query_emb @ self.chunk_embeddings[cid]).item()
            scores.append((cid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional)
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import CrossEncoder
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        if _CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
                print(f"✅ Loaded cross-encoder reranker: {CROSS_ENCODER_MODEL_NAME}")
            except Exception as e:
                print(f"⚠️ Failed to load cross-encoder reranker: {e}")
                self.model = None

    def rerank(self, query: str, chunk_texts: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if self.model is None or not chunk_texts:
            return []
        try:
            pairs = [(query, text) for _, text in chunk_texts]
            scores = self.model.predict(pairs)
            return [(cid, float(score)) for (cid, _), score in zip(chunk_texts, scores)]
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            return []


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------
@dataclass
class ProviderConfig:
    name: str
    model: str
    api_key: str
    base_url: Optional[str] = None


def available_provider_configs(preferred: str = "auto", key_override: Optional[str] = None) -> List[ProviderConfig]:
    def _resolve_key(name: str, env_var: str) -> str:
        if key_override and key_override.strip() and preferred == name:
            return key_override.strip()
        return os.getenv(env_var, "")

    configured = {
        "claude": ProviderConfig("claude", os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL), _resolve_key("claude", "ANTHROPIC_API_KEY")),
        "gemini": ProviderConfig("gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), _resolve_key("gemini", "GEMINI_API_KEY")),
        "openai": ProviderConfig("openai", os.getenv("OPENAI_MODEL", "gpt-5.6-luna"), _resolve_key("openai", "OPENAI_API_KEY")),
        "compatible": ProviderConfig("compatible", os.getenv("COMPATIBLE_MODEL", ""), _resolve_key("compatible", "COMPATIBLE_API_KEY"), os.getenv("COMPATIBLE_BASE_URL", "")),
    }
    priority = ["claude", "gemini", "openai", "compatible"]
    if preferred != "auto":
        priority = [preferred] + [name for name in priority if name != preferred]
    return [configured[name] for name in priority if name in configured and configured[name].api_key and configured[name].model]


def choose_provider() -> str:
    choices = {"1": "auto", "2": "claude", "3": "gemini", "4": "openai", "5": "compatible"}
    print("\nGeneration provider: 1) Auto  2) Claude  3) Gemini  4) OpenAI  5) Compatible")
    return choices.get(input("Select provider [1]: ").strip() or "1", "auto")


class ComparisonEngine:
    """Holds the (expensive, read-only-after-build) retrieval index: TF-IDF,
    entity graph, embedding index, and cross-encoder. These are safe to
    share across many concurrent requests/threads since queries only read
    them.
    """

    def __init__(self, processor: DocumentProcessor, db: DatabaseManager, provider: str = "auto", api_key: Optional[str] = None):
        self.processor = processor
        self.db = db
        self.document_tokens, self.idf = build_tfidf_index(processor.chunk_texts)
        self.entity_graph = EntityKnowledgeGraph(processor.chunk_texts)
        self.partition_chunks: Dict[str, List[str]] = defaultdict(list)
        for chunk_id, metadata in processor.chunks_metadata.items():
            self.partition_chunks[metadata.partition_id].append(chunk_id)
        self.provider = provider
        self.provider_configs = available_provider_configs(provider, key_override=api_key)
        self.embedding_ranker = EmbeddingRanker(db)
        if self.embedding_ranker.model is not None:
            self.embedding_ranker.build_index(processor.chunk_texts)
        self.cross_encoder = CrossEncoderReranker()
        self.CROSS_DOMAIN_THRESHOLD_RELIEF = 0.12
        self.MIN_THRESHOLD = 0.20

    # -------------------------------------------------------------------------
    # Domain detection using entity graph
    # -------------------------------------------------------------------------
    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs that contain entities matching the query."""
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        partition_scores = defaultdict(int)
        for entity in query_entities:
            for chunk_id in self.entity_graph.entity_chunks.get(entity, []):
                metadata = self.processor.chunks_metadata.get(chunk_id)
                if metadata:
                    partition_scores[metadata.partition_id] += 1
        # Sort by score descending
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions]

    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        lower = query.lower()
        comparison_words = {"compare", "contrast", "difference", "versus", "vs", "between", "similar", "differ", "vs."}
        return any(word in lower for word in comparison_words)

    def _is_simple_single_domain(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) != 1:
            return False
        terms = query_terms(query)
        return len(terms) <= 6

    def _get_adaptive_threshold(self, query: str, is_cross_domain: Optional[bool] = None) -> float:
        lower = query.lower()
        complexity_markers = (
            "compare", "contrast", "difference", "versus", " vs ",
            "relationship", "how does", "why does", "analyze", "analyse",
            "evaluate", "synthesize", "critique", "assess",
        )
        complexity = sum(marker in lower for marker in complexity_markers) + len(self._get_active_partitions(query))
        if complexity <= 2:
            base = 0.35
        elif complexity <= 4:
            base = 0.45
        elif complexity <= 6:
            base = 0.55
        else:
            base = 0.60
        if is_cross_domain is None:
            is_cross_domain = self._is_cross_domain_question(query)
        if is_cross_domain:
            base = max(self.MIN_THRESHOLD, base - self.CROSS_DOMAIN_THRESHOLD_RELIEF)
        return base

    def _max_chunks_for_query(self, query: str, is_cross_domain: bool) -> int:
        if self._is_simple_single_domain(query):
            return 2
        if is_cross_domain:
            partitions = self._get_active_partitions(query)
            num_domains = max(1, len(partitions))
            return min(
                MAX_CROSS_DOMAIN_CHUNKS_CEILING,
                max(MAX_CROSS_DOMAIN_CHUNKS, num_domains * MIN_CHUNKS_PER_ACTIVE_PARTITION)
            )
        complexity = len(query_terms(query))
        if complexity <= 6:
            return 2
        elif complexity <= 10:
            return 3
        elif complexity <= 15:
            return 4
        else:
            return MAX_CHUNKS

    # -------------------------------------------------------------------------
    # Ranking and fusion (unchanged from original)
    # -------------------------------------------------------------------------
    @staticmethod
    def _calibrate_lexical_scores(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        pivot = 0.10
        return [(chunk_id, score / (score + pivot) if score > 0 else 0.0) for chunk_id, score in ranked]

    def _fuse_rankings(self, *rankings: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not rankings:
            return []
        non_empty = [r for r in rankings if r]
        if not non_empty:
            return []
        if len(non_empty) == 1:
            return non_empty[0]
        rrf_scores: Dict[str, float] = defaultdict(float)
        rrf_k = 60
        for ranking in non_empty:
            for position, (chunk_id, _) in enumerate(ranking, start=1):
                rrf_scores[chunk_id] += 1.0 / (rrf_k + position)
        return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    def _rank_candidates(
        self, query: str, candidate_ids: Sequence[str], use_hybrid: bool
    ) -> List[Tuple[str, float]]:
        terms = query_terms(query)
        documents = {
            chunk_id: self.document_tokens[chunk_id]
            for chunk_id in candidate_ids
            if chunk_id in self.document_tokens
        }
        lexical = self._calibrate_lexical_scores(vector_rank(documents, self.idf, terms))
        if not use_hybrid:
            return lexical
        rankings = [lexical]
        graph = self.entity_graph.rank(query, documents.keys())
        if graph:
            rankings.append(graph)
        if self.embedding_ranker.model is not None and len(candidate_ids) > 0:
            emb_rank = self.embedding_ranker.rank(query, candidate_ids)
            if emb_rank:
                max_score = max(s for _, s in emb_rank) or 1.0
                emb_rank = [(cid, s / max_score) for cid, s in emb_rank]
                rankings.append(emb_rank)
        fused = self._fuse_rankings(*rankings)
        if self.cross_encoder.model is not None and len(fused) > 3:
            top_k = fused[:CROSS_ENCODER_RERANK_TOP_K]
            top_k_texts = [(cid, self.processor.chunk_texts.get(cid, "")) for cid, _ in top_k]
            reranked_raw = self.cross_encoder.rerank(query, top_k_texts)
            if reranked_raw:
                reranked = [(cid, 1.0 / (1.0 + math.exp(-score))) for cid, score in reranked_raw]
                reranked_ids = {cid for cid, _ in reranked}
                remainder = [item for item in fused if item[0] not in reranked_ids]
                fused = sorted(reranked + remainder, key=lambda item: item[1], reverse=True)
        return fused

    WEAK_TOP_SCORE = 0.05

    def _get_cross_domain_rankings(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        active_partitions = self._get_active_partitions(query)
        if not active_partitions:
            return self._rank_candidates(query, list(self.processor.chunk_texts), use_hybrid)
        by_partition: Dict[str, List[Tuple[str, float]]] = {}
        for partition_id in active_partitions:
            own_pool = self.partition_chunks.get(partition_id, [])
            ranking: List[Tuple[str, float]] = (
                self._rank_candidates(query, own_pool, use_hybrid) if own_pool else []
            )
            # Fallback: if partition has weak scores, pull from global pool
            if not ranking or ranking[0][1] <= self.WEAK_TOP_SCORE:
                global_pool = [cid for cid in self.processor.chunk_texts if cid not in own_pool]
                fallback = self._rank_candidates(query, global_pool, use_hybrid)
                ranking = self._merge_rankings_unique(ranking, fallback)
            if ranking:
                by_partition[partition_id] = ranking
        if not by_partition:
            return self._rank_candidates(query, list(self.processor.chunk_texts), use_hybrid)

        seeds: List[Tuple[str, float]] = []
        selected: Set[str] = set()
        for partition_id, ranking in by_partition.items():
            for chunk_id, score in ranking[:MIN_CHUNKS_PER_ACTIVE_PARTITION]:
                if chunk_id not in selected:
                    seeds.append((chunk_id, score))
                    selected.add(chunk_id)
        remaining = [
            item
            for ranking in by_partition.values()
            for item in ranking
            if item[0] not in selected
        ]
        remaining.sort(key=lambda item: item[1], reverse=True)
        return seeds + remaining

    @staticmethod
    def _merge_rankings_unique(
        primary: List[Tuple[str, float]], fallback: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        best: Dict[str, float] = {}
        for chunk_id, score in primary + fallback:
            if chunk_id not in best or score > best[chunk_id]:
                best[chunk_id] = score
        return sorted(best.items(), key=lambda item: item[1], reverse=True)

    # -------------------------------------------------------------------------
    # Confidence, context, validation (unchanged)
    # -------------------------------------------------------------------------
    def _calculate_confidence(
        self,
        chunk_ids: Sequence[str],
        normalised_scores: Sequence[float],
        require_cross_domain: bool,
    ) -> float:
        if not chunk_ids or not normalised_scores:
            return 0.0
        strengths = [max(0.0, min(1.0, score)) for score in normalised_scores]
        average_strength = sum(strengths) / len(strengths)
        weakest_evidence = min(strengths)
        partitions = {
            self.processor.chunks_metadata[chunk_id].partition_id
            for chunk_id in chunk_ids
            if chunk_id in self.processor.chunks_metadata
        }
        if require_cross_domain and len(partitions) < 2:
            return 0.0
        cross_domain_bonus = 0.10 if require_cross_domain and len(partitions) >= 2 else 0.0
        return max(
            0.0,
            min(1.0, (0.65 * average_strength) + (0.25 * weakest_evidence) + cross_domain_bonus),
        )

    @staticmethod
    def _truncate_on_sentence_boundary(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        window = text[:limit]
        best_end = -1
        for match in re.finditer(r"[.!?](?:\s|$)", window):
            best_end = match.end()
        if best_end >= int(limit * 0.4):
            return window[:best_end].strip()
        return window.rsplit(" ", 1)[0]

    def _build_context(self, chunk_ids: Sequence[str]) -> str:
        sections = []
        for chunk_id in chunk_ids:
            text = self.processor.chunk_texts.get(chunk_id, "")
            metadata = self.processor.chunks_metadata.get(chunk_id)
            if not text or not metadata:
                continue
            partition = metadata.partition_id.replace("partition_", "")
            excerpt = self._truncate_on_sentence_boundary(text, CONTEXT_CHARS_PER_CHUNK)
            sections.append(f"[{chunk_id}|{partition}]\n{excerpt}")
        return "\n\n".join(sections)

    def _source_summary(self, chunk_ids: Sequence[str]) -> List[str]:
        sources = []
        for chunk_id in chunk_ids:
            metadata = self.processor.chunks_metadata.get(chunk_id)
            if not metadata:
                continue
            sources.append(
                f"{chunk_id} | {Path(metadata.file_path).name} | "
                f"{metadata.partition_id.replace('partition_', '')}"
            )
        return sources

    def _validate_answer_quality(
        self,
        query: str,
        context: str,
        chunk_ids: Sequence[str],
        require_cross_domain: bool,
    ) -> bool:
        if not context.strip() or not chunk_ids:
            return False
        if require_cross_domain:
            partitions = {
                self.processor.chunks_metadata[chunk_id].partition_id
                for chunk_id in chunk_ids
                if chunk_id in self.processor.chunks_metadata
            }
            if len(partitions) < 2:
                return False
            # Ensure at least one entity from the query appears in each partition's text
            query_entities = extract_entities(query)
            if query_entities:
                for partition_id in partitions:
                    partition_text = " ".join(
                        self.processor.chunk_texts.get(chunk_id, "")
                        for chunk_id in chunk_ids
                        if self.processor.chunks_metadata.get(chunk_id)
                        and self.processor.chunks_metadata[chunk_id].partition_id == partition_id
                    ).lower()
                    if not any(entity.lower() in partition_text for entity in query_entities):
                        return False
        terms = set(query_terms(query))
        if not terms:
            return True
        context_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", context)}
        coverage = sum(term in context_tokens for term in terms) / len(terms)
        coverage_threshold = 0.45 if require_cross_domain else 0.55
        if coverage < coverage_threshold:
            return False
        return True

    # -------------------------------------------------------------------------
    # Intelligent retrieval and answer generation (unchanged)
    # -------------------------------------------------------------------------
    def _intelligent_retrieve(
        self,
        ranked_chunks: Sequence[Tuple[str, float]],
        query: str,
        confidence_threshold: float,
        require_cross_domain: bool,
        max_chunks: int = MAX_CHUNKS,
    ) -> Tuple[List[str], float]:
        ranked_chunks = list(ranked_chunks[:max_chunks])
        if not ranked_chunks:
            return [], 0.0
        selected_ids: List[str] = []
        selected_scores: List[float] = []
        for chunk_id, score in ranked_chunks:
            if chunk_id in selected_ids:
                continue
            selected_ids.append(chunk_id)
            selected_scores.append(max(0.0, min(1.0, score)))
            if len(selected_ids) < min(2, len(ranked_chunks)):
                continue
            confidence = self._calculate_confidence(
                selected_ids, selected_scores, require_cross_domain
            )
            context = self._build_context(selected_ids)
            quality_ok = self._validate_answer_quality(
                query, context, selected_ids, require_cross_domain
            )
            if confidence >= confidence_threshold and quality_ok:
                return selected_ids, confidence
        return selected_ids, self._calculate_confidence(
            selected_ids, selected_scores, require_cross_domain
        )

    def _calculate_chunk_relevance(self, query: str, chunk_ids: Sequence[str]) -> float:
        terms = query_terms(query)
        if not terms or not chunk_ids:
            return 0.0
        per_chunk = []
        for chunk_id in chunk_ids:
            text = self.processor.chunk_texts.get(chunk_id, "").lower()
            if not text:
                continue
            text_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", text)}
            covered = sum(term in text_tokens for term in set(terms)) / len(set(terms))
            entity_bonus = 0.1 * len(extract_entities(query) & extract_entities(text))
            per_chunk.append(min(1.0, covered + entity_bonus))
        return 100.0 * (sum(per_chunk) / len(per_chunk)) if per_chunk else 0.0

    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        terms = set(query_terms(query))
        if not terms or not answer:
            return 0.0
        answer_tokens = {stem_token(token) for token in re.findall(r"\b[\w-]+\b", answer.lower())}
        coverage = sum(term in answer_tokens for term in terms) / len(terms)
        return 100.0 * min(1.0, coverage)

    def _local_answer(self, query: str, context: str, method: str) -> str:
        if not context:
            return "No retrieved context is available."
        terms = query_terms(query)
        sentences = re.split(r"(?<=[.!?])\s+", context)
        relevant = [
            sentence.strip() for sentence in sentences
            if len(sentence.strip()) > 40 and any(term in sentence.lower() for term in terms)
        ]
        if not relevant:
            return "Relevant evidence was retrieved, but no concise local extract was found."
        return f"{method} local evidence summary:\n" + "\n".join(f"- {sentence}" for sentence in relevant[:4])

    def _generate_answer(self, query: str, context: str, method: str, provider_configs: List[ProviderConfig]) -> Tuple[str, Dict[str, Any]]:
        if not provider_configs:
            return self._local_answer(query, context, method), {"provider": None, "notice": ""}
        if context:
            prompt = (
                "You are answering a technical question using retrieved evidence "
                "from research papers.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Write a clear, complete answer. Ground it in the evidence above "
                "and cite the relevant [chunk_id] next to any claim it supports. "
                "If the evidence only partially covers the question, you may use "
                "your own general knowledge to fill gaps or add context -- but "
                "make it clear when you're doing so (e.g. 'more generally, ...') "
                "rather than presenting it as coming from the cited papers."
            )
        else:
            prompt = f"Q: {query}\nGive a concise answer and state uncertainty where relevant."
        failures = []
        for config in provider_configs:
            try:
                input_tokens = estimate_tokens(prompt)
                if config.name in self.db.CREDIT_PROVIDERS:
                    allowed, reason = self.db.can_generate(
                        config.name, input_tokens, MAX_ANSWER_TOKENS
                    )
                    if not allowed:
                        failures.append(reason)
                        continue
                if config.name == "claude":
                    if anthropic is None:
                        raise RuntimeError("install anthropic")
                    response = anthropic.Anthropic(api_key=config.api_key).messages.create(
                        model=config.model, max_tokens=MAX_ANSWER_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = "".join(block.text for block in response.content if block.type == "text")
                    actual_input = int(getattr(response.usage, "input_tokens", input_tokens) or input_tokens)
                    actual_output = int(getattr(response.usage, "output_tokens", estimate_tokens(answer)) or estimate_tokens(answer))
                elif config.name == "gemini":
                    if genai is None:
                        raise RuntimeError("install google-genai")
                    response = genai.Client(api_key=config.api_key).models.generate_content(
                        model=config.model, contents=prompt,
                    )
                    answer = response.text or ""
                    usage = getattr(response, "usage_metadata", None)
                    actual_input = int(getattr(usage, "prompt_token_count", input_tokens) or input_tokens)
                    actual_output = int(getattr(usage, "candidates_token_count", estimate_tokens(answer)) or estimate_tokens(answer))
                else:
                    if OpenAI is None:
                        raise RuntimeError("install openai")
                    kwargs: Dict[str, Any] = {"api_key": config.api_key}
                    if config.base_url:
                        kwargs["base_url"] = config.base_url
                    client = OpenAI(**kwargs)
                    response = client.chat.completions.create(
                        model=config.model, max_completion_tokens=MAX_ANSWER_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = response.choices[0].message.content or ""
                    usage = response.usage
                    actual_input = int(getattr(usage, "prompt_tokens", input_tokens) or input_tokens)
                    actual_output = int(getattr(usage, "completion_tokens", estimate_tokens(answer)) or estimate_tokens(answer))
                if clean_text(answer):
                    cost = 0.0
                    if config.name in self.db.CREDIT_PROVIDERS:
                        cost = self.db.record_usage(config.name, config.model, actual_input, actual_output)
                    generation_info = {
                        "provider": config.name,
                        "input_tokens": actual_input,
                        "output_tokens": actual_output,
                        "cost_usd": cost,
                        "notice": "",
                    }
                    return clean_text(answer), generation_info
                raise RuntimeError("empty response")
            except Exception as exc:
                failures.append(f"{config.name}: {exc}")
        notice = "No provider call was made; local evidence extraction was used. " + "; ".join(failures)
        generation_info = {"provider": None, "notice": notice}
        return self._local_answer(query, context, method) + "\n[" + notice + "]", generation_info

    # -------------------------------------------------------------------------
    # Main RAG execution
    # -------------------------------------------------------------------------
    def _run_rag(self, query: str, method: str, use_hybrid: bool, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        start = time.perf_counter()
        is_cross_domain = self._is_cross_domain_question(query) if use_hybrid else False
        threshold = self._get_adaptive_threshold(query, is_cross_domain)
        max_chunks = self._max_chunks_for_query(query, is_cross_domain)

        if use_hybrid and not is_cross_domain:
            # Cheap-first router for single-domain
            cheap_ranked = self._rank_candidates(query, list(self.processor.chunk_texts), use_hybrid=False)
            cheap_ids, cheap_confidence = self._intelligent_retrieve(
                cheap_ranked, query, threshold, False, max_chunks
            )
            cheap_context = self._build_context(cheap_ids)
            cheap_quality = self._validate_answer_quality(query, cheap_context, cheap_ids, False)

            if cheap_quality and cheap_confidence >= threshold:
                chunk_ids, confidence = cheap_ids, cheap_confidence
                retrieval_path = "tfidf_only"
            else:
                # Escalate to full stack
                full_ranked = self._get_cross_domain_rankings(query, use_hybrid=True)
                full_ids, full_confidence = self._intelligent_retrieve(
                    full_ranked, query, threshold, False, max_chunks
                )
                if full_confidence >= cheap_confidence:
                    chunk_ids, confidence = full_ids, full_confidence
                    retrieval_path = "hybrid_escalated"
                else:
                    chunk_ids, confidence = cheap_ids, cheap_confidence
                    retrieval_path = "tfidf_only_fallback"
        elif use_hybrid:
            ranked = self._get_cross_domain_rankings(query, use_hybrid=True)
            chunk_ids, confidence = self._intelligent_retrieve(
                ranked, query, threshold, is_cross_domain, max_chunks
            )
            retrieval_path = "hybrid_cross_domain"
        else:
            ranked = self._rank_candidates(query, list(self.processor.chunk_texts), False)
            chunk_ids, confidence = self._intelligent_retrieve(
                ranked, query, threshold, is_cross_domain, max_chunks
            )
            retrieval_path = "standard_rag"

        context = self._build_context(chunk_ids)
        answer, generation_info = self._generate_answer(query, context, method, provider_configs)
        latency = time.perf_counter() - start
        quality_validated = self._validate_answer_quality(
            query, context, chunk_ids, is_cross_domain
        )

        return {
            "answer": answer,
            "chunks_retrieved": len(chunk_ids),
            "tokens_used": estimate_tokens(context),
            "latency": latency,
            "chunk_ids": chunk_ids,
            "relevance_score": self._calculate_chunk_relevance(query, chunk_ids),
            "answer_relevance": self._calculate_answer_relevance(query, answer),
            "confidence_score": confidence,
            "threshold_used": threshold,
            "quality_validated": quality_validated,
            "cross_domain": is_cross_domain,
            "sources": self._source_summary(chunk_ids),
            "generation": generation_info,
        }

    def _hybrid_rag(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        return self._run_rag(query, "Hybrid RAG", use_hybrid=True, provider_configs=provider_configs)

    def _standard_rag(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        return self._run_rag(query, "Standard RAG", use_hybrid=False, provider_configs=provider_configs)

    def _direct_ai(self, query: str, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        start = time.perf_counter()
        answer, generation_info = self._generate_answer(query, "", "Direct AI", provider_configs)
        return {
            "answer": answer,
            "chunks_retrieved": 0,
            "tokens_used": 0,
            "latency": time.perf_counter() - start,
            "chunk_ids": [],
            "relevance_score": 0.0,
            "answer_relevance": self._calculate_answer_relevance(query, answer),
            "confidence_score": 0.0,
            "threshold_used": 0.0,
            "quality_validated": False,
            "cross_domain": False,
            "sources": [],
            "generation": generation_info,
        }

    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        provider_configs = (
            available_provider_configs(provider, key_override=api_key)
            if provider is not None
            else self.provider_configs
        )
        hybrid = self._hybrid_rag(query, provider_configs)
        standard = self._standard_rag(query, provider_configs)
        direct = self._direct_ai(query, provider_configs)
        saved = max(0, standard["tokens_used"] - hybrid["tokens_used"])
        hybrid["tokens_saved_vs_standard"] = saved
        hybrid["token_savings_percent_vs_standard"] = (
            100.0 * saved / standard["tokens_used"]
            if standard["tokens_used"] else 0.0
        )
        results = {
            "hybrid_rag": hybrid,
            "standard_rag": standard,
            "direct_ai": direct,
        }
        self.db.save_comparison(query, results)
        return results

    def compare_single(self, query: str, method: str = "hybrid_rag",
                       provider: str = "auto", api_key: Optional[str] = None) -> Dict[str, Any]:
        provider_configs = available_provider_configs(provider, key_override=api_key)
        if method == "hybrid_rag":
            return self._hybrid_rag(query, provider_configs)
        elif method == "standard_rag":
            return self._standard_rag(query, provider_configs)
        elif method == "direct_ai":
            return self._direct_ai(query, provider_configs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def test_provider_connection(self, provider: str, api_key: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        provider_configs = available_provider_configs(provider, key_override=api_key)
        if not provider_configs:
            return "", {"provider": None, "notice": f"No usable configuration for '{provider}' (no API key found, typed or in env)."}
        return self._generate_answer("Reply with exactly: OK", "", "connection test", provider_configs)

    def format_comparison(self, results: Dict[str, Dict[str, Any]]) -> str:
        lines = [
            "\n" + "=" * 88,
            "COMPARISON RESULTS (UNIVERSAL RAG v22)",
            "=" * 88,
            "NOTE: 'Chunk relevance', 'confidence', and 'quality' are grounding",
            "metrics -- they measure whether retrieved evidence overlaps the query and",
            "spans the right domains. They are legitimately 0/no for Direct AI, since it",
            "retrieves nothing. Use 'Answer relevance' (query-term coverage of the",
            "actual answer text) for a metric that's computed the same way across all",
            "pipelines.",
            "-" * 88,
            "Method        Latency    Ctx tokens  Chunks  Chunk rel  Answer rel  Confidence  Quality",
            "-" * 88,
        ]
        for method, result in results.items():
            lines.append(
                f"{method:<13} {result['latency']:>7.2f}s"
                f" {result['tokens_used']:>11}"
                f" {result['chunks_retrieved']:>7}"
                f" {result['relevance_score']:>10.1f}"
                f" {result.get('answer_relevance', 0.0):>11.1f}"
                f" {result['confidence_score']:>11.2f}"
                f"  {'yes' if result['quality_validated'] else 'no'}"
            )
        hybrid = results["hybrid_rag"]
        lines.append("-" * 88)
        lines.append(
            "Hybrid context-token savings vs Standard: "
            f"{hybrid.get('tokens_saved_vs_standard', 0)} "
            f"({hybrid.get('token_savings_percent_vs_standard', 0.0):.1f}%)"
        )
        for method, result in results.items():
            lines.extend([
                "\n" + method.upper(),
                f"Threshold: {result['threshold_used']:.2f} | "
                f"Chunk IDs: {', '.join(result['chunk_ids']) or 'none'}",
                "Sources: " + ("; ".join(result.get("sources", [])) or "none"),
                result["answer"],
            ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export and CLI (unchanged)
# ---------------------------------------------------------------------------
def export_markdown(query: str, results: Dict[str, Dict[str, Any]]) -> Path:
    output_dir = Path("rag_exports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# Universal RAG comparison (v22)\n\nQuestion: {query}\n",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
    ]
    for method, result in results.items():
        lines.extend([
            f"## {method}\n",
            f"- Latency: {result['latency']:.2f}s",
            f"- Estimated context tokens: {result['tokens_used']}",
            f"- Chunks: {result['chunks_retrieved']}",
            f"- Chunk relevance: {result['relevance_score']:.1f}",
            f"- Answer relevance: {result.get('answer_relevance', 0.0):.1f}",
            f"- Confidence: {result['confidence_score']:.2f}",
            f"- Threshold: {result['threshold_used']:.2f}",
            f"- Quality validated: {result['quality_validated']}",
            f"- Chunk IDs: {', '.join(result['chunk_ids']) or 'none'}\n",
            f"### Answer\n\n{result['answer']}\n",
        ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def print_status(db: DatabaseManager, processor: DocumentProcessor) -> None:
    documents = db.get_all_documents()
    print(f"\nLoaded documents: {len(documents)} | total chunks: {len(processor.chunk_texts)}")
    for document in documents:
        print(
            f"  - {document['file_name']} | {document['partition_id']} | "
            f"{document['chunk_count']} chunks"
        )


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    print("=" * 72)
    print("UNIVERSAL HYBRID RAG v22 (ENTERPRISE EDITION)")
    print("TF-IDF + Entity Graph/RRF + Embeddings + Cross-Encoder")
    print("Cheap-first routing, credit-aware provider fallback, and ZIP ingestion")
    print("=" * 72)
    db = DatabaseManager()
    processor = DocumentProcessor(db)
    print_status(db, processor)
    provider = choose_provider()
    while True:
        print("\n1) Add papers with file picker")
        print("2) Add papers by path")
        print("3) Ask and compare")
        print("4) Show status")
        print("5) Quit")
        choice = input("Select (1-5): ").strip()
        if choice == "1":
            files = select_files_dialog()
            if files:
                processor.process_documents(files)
        elif choice == "2":
            raw_paths = input("Comma-separated PDF/TXT/MD/PY/JS/IPYNB/ZIP paths: ").strip()
            if raw_paths:
                processor.process_documents([path.strip() for path in raw_paths.split(",")])
        elif choice == "3":
            if not processor.chunk_texts:
                print("Load at least one document first.")
                continue
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            engine = ComparisonEngine(processor, db, provider)
            results = engine.compare(query)
            print(engine.format_comparison(results))
            if input("Export Markdown report? (y/n): ").strip().lower() == "y":
                path = export_markdown(query, results)
                print(f"Exported: {path.resolve()}")
        elif choice == "4":
            print_status(db, processor)
        elif choice in {"5", "q", "quit", "exit"}:
            print("Goodbye.")
            return
        else:
            print("Please enter a number from 1 to 5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")