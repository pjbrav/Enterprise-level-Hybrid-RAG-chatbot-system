"""Retrieval data types shared by the engine, the host API and the member client.

The engine's pipeline is split into retrieve -> select -> answer so that the
ranked candidates can come from somewhere other than the engine's own database
(a remote host, or a fusion of host + local). Anything not stored locally
travels in a ChunkStore (texts + metadata held in RAM for one request).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag.db import ChunkMetadata
from rag.params import HybridParams

MAX_WIRE_TEXT_CHARS = 8000   # hard cap per chunk on the wire (chunks are ~220 words)


@dataclass
class ChunkStore:
    """Texts and metadata for chunks that are not (only) in the engine's own
    database. Keys are chunk ids exactly as they appear in a ranking."""
    texts: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, ChunkMetadata] = field(default_factory=dict)

    def update(self, other: "ChunkStore") -> None:
        self.texts.update(other.texts)
        self.meta.update(other.meta)


@dataclass
class RetrievalResult:
    ranked: List[Tuple[str, float]]          # (chunk_id, score 0-1), best first
    is_cross_domain: bool
    threshold: float
    max_chunks: int
    use_hybrid: bool
    start: float = 0.0                       # perf_counter() when retrieval began (end-to-end latency)
    store: Optional[ChunkStore] = None       # set when chunks live outside this engine's db
    params: Optional[HybridParams] = None    # Hybrid tuning this result was produced with


# ---------------------------------------------------------------------------
# Wire format (host API <-> member client)
# ---------------------------------------------------------------------------
def result_to_wire(result: RetrievalResult, texts: Dict[str, str], meta: Dict[str, ChunkMetadata],
                   pool: int, latency: float) -> Dict[str, Any]:
    """Top `pool` candidates with the texts the member needs. File paths are
    reduced to the file name so the host's directory layout is not disclosed."""
    chunks = []
    for chunk_id, score in result.ranked[:pool]:
        text, m = texts.get(chunk_id), meta.get(chunk_id)
        if not text or m is None:
            continue
        chunks.append({
            "chunk_id": chunk_id,
            "score": float(score),
            "text": text[:MAX_WIRE_TEXT_CHARS],
            "partition_id": m.partition_id,
            "file_name": Path(m.file_path).name,
            "chunk_index": int(m.chunk_index),
            "token_count": int(m.token_count),
            "entity_count": int(m.entity_count),
        })
    return {
        "chunks": chunks,
        "is_cross_domain": result.is_cross_domain,
        "threshold": result.threshold,
        "max_chunks": result.max_chunks,
        "use_hybrid": result.use_hybrid,
        "latency": latency,
    }


def wire_to_result(payload: Dict[str, Any], prefix: str = "") -> RetrievalResult:
    """Rebuild a RetrievalResult (with a ChunkStore holding the texts) from a
    host response. `prefix` namespaces chunk ids (e.g. 'host:') so they cannot
    collide with a local corpus' ids when the two are fused."""
    store = ChunkStore()
    ranked: List[Tuple[str, float]] = []
    for c in payload.get("chunks", []):
        cid = f"{prefix}{c['chunk_id']}"
        ranked.append((cid, float(c["score"])))
        store.texts[cid] = c["text"]
        store.meta[cid] = ChunkMetadata(cid, c.get("file_name", ""), c["partition_id"],
                                        int(c.get("chunk_index", 0)), int(c.get("token_count", 0)),
                                        int(c.get("entity_count", 0)))
    return RetrievalResult(
        ranked=ranked,
        is_cross_domain=bool(payload.get("is_cross_domain", False)),
        threshold=float(payload.get("threshold", 0.0)),
        max_chunks=int(payload.get("max_chunks", 5)),
        use_hybrid=bool(payload.get("use_hybrid", True)),
        store=store,
    )
