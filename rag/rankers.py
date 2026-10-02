"""TF-IDF, entity graph, embedding and cross-encoder rankers."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_EMBEDDING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDING_AVAILABLE = True
except ImportError:
    pass

from rag.config import CROSS_ENCODER_MODEL_NAME, EMBEDDING_MODEL_NAME
from rag.text import extract_entities, stem_token
from rag.db import DatabaseManager


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

    def rank_on_demand(self, query: str, chunk_texts: Dict[str, str]) -> List[Tuple[str, float]]:
        """Embed and rank just the given (small) set of texts, fresh, with
        no persistent index. Used by the tier-4 last-resort escalation in
        large-corpus mode, where building/holding a full-corpus embedding
        index isn't an option under the memory budget -- this way
        embeddings only ever exist transiently, for a handful of already
        pre-filtered candidates, and are garbage-collected right after.
        """
        if self.model is None or not chunk_texts:
            return []
        try:
            chunk_ids = list(chunk_texts.keys())
            texts = list(chunk_texts.values())
            query_emb = self.model.encode(query, convert_to_tensor=True)
            chunk_embs = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            scores = [(cid, (query_emb @ emb).item()) for cid, emb in zip(chunk_ids, chunk_embs)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
        except Exception as e:
            print(f"⚠️ On-demand embedding ranking failed: {e}")
            return []


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
