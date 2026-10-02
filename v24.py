"""Compatibility shim: the engine now lives in the rag/ package.

`import v24` / `from v24 import ...` and `python v24.py` keep working.
"""

from __future__ import annotations

from rag.config import (  # noqa: F401
    AUTO_PARTITION_MAX_CORPUS_CHARS,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    CONTEXT_CHARS_PER_CHUNK,
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODER_RERANK_TOP_K,
    DATABASE_PATH,
    DATABASE_URL,
    DEFAULT_MODEL,
    EMBEDDING_MODEL_NAME,
    GENERIC_FOLDER_NAMES,
    MAX_ANSWER_TOKENS,
    MAX_CHUNKS,
    MAX_CROSS_DOMAIN_CHUNKS,
    MAX_CROSS_DOMAIN_CHUNKS_CEILING,
    MAX_ZIP_FILES,
    MAX_ZIP_MEMBER_BYTES,
    MIN_CHUNKS_PER_ACTIVE_PARTITION,
    PGVECTOR_DIMENSIONS,
    PROVIDER_PRICING_DEFAULTS,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    is_running_in_docker,
    load_dotenv,
)
from rag.text import (  # noqa: F401
    STOPWORDS,
    _ENTITY_STOPWORDS,
    _SPACY_AVAILABLE,
    _SPACY_ENTITY_TYPES,
    _SPACY_NLP,
    _add_regex_ids,
    _get_spacy_nlp,
    clean_text,
    estimate_tokens,
    extract_entities,
    query_terms,
    stem_token,
    tiktoken,
)
from rag.db import (  # noqa: F401
    ChunkMetadata,
    DatabaseManager,
    RealDictCursor,
    estimate_provider_cost,
    psycopg2,
)
from rag.ingest import (  # noqa: F401
    DocumentProcessor,
    IngestionSummary,
    _hash_file_streaming,
    select_files_dialog,
)
from rag.rankers import (  # noqa: F401
    CrossEncoderReranker,
    EmbeddingRanker,
    EntityKnowledgeGraph,
    SentenceTransformer,
    _EMBEDDING_AVAILABLE,
    build_tfidf_index,
    vector_rank,
)
from rag.providers import (  # noqa: F401
    ProviderConfig,
    available_provider_configs,
    choose_provider,
)
from rag.engine import (  # noqa: F401
    ComparisonEngine,
    OpenAI,
    anthropic,
    build_large_corpus_engine,
    export_markdown,
    genai,
)
from rag.retrieval import (  # noqa: F401
    ChunkStore,
    RetrievalResult,
    result_to_wire,
    wire_to_result,
)
from rag.cli import (  # noqa: F401
    main,
    print_status,
)


if __name__ == "__main__":
    main()
