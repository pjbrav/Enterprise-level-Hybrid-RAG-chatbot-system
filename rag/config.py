"""Environment loading, constants and small environment helpers."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment & optional imports
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATABASE_PATH = os.getenv("SQLITE_DATABASE_PATH", "data/hybrid_rag_v22.db")


DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".js", ".ipynb"}


MAX_ZIP_FILES = int(os.getenv("MAX_ZIP_FILES", "250"))


MAX_ZIP_MEMBER_BYTES = int(os.getenv("MAX_ZIP_MEMBER_BYTES", str(25 * 1024 * 1024)))


# Folder names that carry no partitioning signal (e.g. a zip whose members
# all sit under "documents/"). Seeing only these names is treated the same
# as a flat/un-foldered corpus -- auto content-partitioning kicks in instead
# of literally creating a single partition called "documents".
GENERIC_FOLDER_NAMES = {
    "documents", "docs", "uploads", "papers", "files", "data", "corpus",
    "input", "inputs", "source", "sources", "pdfs", "pdf", "",
}


# Above this much extracted text, skip TF-IDF+KMeans clustering (it needs
# the whole corpus in memory at once) and fall back to a cheap per-document
# keyword label instead. Keeps a ~1.5GB raw upload (much smaller once PDFs
# are reduced to text) from stalling ingestion or blowing up memory.
AUTO_PARTITION_MAX_CORPUS_CHARS = int(os.getenv("AUTO_PARTITION_MAX_CORPUS_CHARS", str(80_000_000)))


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


CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-TinyBERT-L-2-v2")


CROSS_ENCODER_RERANK_TOP_K = 3


PROVIDER_PRICING_DEFAULTS = {
    "claude": (0.80, 2.40),
    "openai": (1.00, 6.00),
    "gemini": (0.15, 0.60),
}


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------
def is_running_in_docker() -> bool:
    return os.path.exists('/.dockerenv')
