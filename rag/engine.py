"""ComparisonEngine: retrieval pipelines, routing, generation."""

from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

from rag.config import CONTEXT_CHARS_PER_CHUNK, CROSS_ENCODER_RERANK_TOP_K, DATABASE_PATH, MAX_ANSWER_TOKENS, MAX_CHUNKS, MAX_CROSS_DOMAIN_CHUNKS, MAX_CROSS_DOMAIN_CHUNKS_CEILING, MIN_CHUNKS_PER_ACTIVE_PARTITION
from rag.text import clean_text, estimate_tokens, extract_entities, query_terms, stem_token
from rag.db import ChunkMetadata, DatabaseManager
from rag.ingest import DocumentProcessor
from rag.rankers import CrossEncoderReranker, EmbeddingRanker, EntityKnowledgeGraph, vector_rank
from rag.providers import ProviderConfig, available_provider_configs


class ComparisonEngine:
    """Holds the (expensive, read-only-after-build) retrieval index: TF-IDF,
    entity graph, embedding index, and cross-encoder. These are safe to
    share across many concurrent requests/threads since queries only read
    them.
    """

    def __init__(self, processor: DocumentProcessor, db: DatabaseManager, provider: str = "auto", api_key: Optional[str] = None):
        self.processor = processor
        self.db = db
        self.lazy = processor.lazy
        if self.lazy:
            # Large-corpus mode: no document_tokens, no entity_graph, no
            # full-corpus embedding index -- retrieval reads chunks_fts /
            # chunk_entities directly (see _rank_candidates,
            # _get_active_partitions) so memory stays bounded by query
            # candidate-set size, not corpus size.
            self.document_tokens, self.idf, self.entity_graph = {}, {}, None
            self.partition_chunks = self.db.get_partition_counts()  # {partition_id: count}, not chunk-id lists
            self.embedding_ranker = EmbeddingRanker(db)
            # Deliberately do NOT call build_index() here: that would
            # embed the whole corpus into RAM, exactly what lazy mode
            # exists to avoid. Embeddings are computed on demand, only for
            # a query's small candidate set, by the tier-4 escalation path.
        else:
            self.document_tokens, self.idf, self.entity_graph = self._build_index_streaming(processor.chunk_texts)
            self.partition_chunks: Dict[str, List[str]] = defaultdict(list)
            for chunk_id, metadata in processor.chunks_metadata.items():
                self.partition_chunks[metadata.partition_id].append(chunk_id)
            self.embedding_ranker = EmbeddingRanker(db)
            if self.embedding_ranker.model is not None:
                self.embedding_ranker.build_index(processor.chunk_texts)
        self.provider = provider
        self.provider_configs = available_provider_configs(provider, key_override=api_key)
        self.cross_encoder = CrossEncoderReranker()
        self.CROSS_DOMAIN_THRESHOLD_RELIEF = float(os.getenv("CROSS_DOMAIN_THRESHOLD_RELIEF", "0.12"))
        # Load per-customer routing config if available
        self._load_routing_config()
        self.MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.20"))
        # Smart router thresholds (see _should_use_hybrid).
        # Default 2.0 (unreachable) forces Hybrid on every query — maximises quality
        # and token savings (+0.9pp relevance, 7.7% tokens) at ~10s latency cost.
        # Set ROUTER_STANDARD_CONFIDENCE_HIGH=0.55 to enable the smart router.
        self.ROUTER_STANDARD_CONFIDENCE_HIGH = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_HIGH", "2.0"))
        self.ROUTER_STANDARD_CONFIDENCE_LOW = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_LOW", "0.42"))
        self.ROUTER_HYBRID_SCORE_THRESHOLD = float(os.getenv("ROUTER_HYBRID_SCORE_THRESHOLD", "0.50"))
        # Tier-4 last-resort threshold: if the disk-backed FTS5+graph
        # Hybrid tier is *still* not confident after running, escalate to
        # cross-encoder/embedding reranking of just its own top candidates
        # (never the full corpus). Only meaningful in lazy mode -- in
        # non-lazy mode cross-encoder/embeddings already run as part of
        # every Hybrid call, unchanged from before.
        self.ROUTER_LAST_RESORT_THRESHOLD = float(os.getenv("ROUTER_LAST_RESORT_THRESHOLD", "0.40"))
        self.LAST_RESORT_CANDIDATE_POOL = int(os.getenv("LAST_RESORT_CANDIDATE_POOL", "20"))

    def _load_routing_config(self, config_path: str = "routing_config.yaml") -> None:
        """Load per-customer partition routing keywords from a YAML config file.

        If the file doesn't exist, the hardcoded defaults in
        _DEFAULT_PARTITION_KEYWORDS are used. This lets each deployment
        customize routing without editing code.
        """
        try:
            import yaml
        except ImportError:
            return  # PyYAML not installed — use defaults
        if not os.path.exists(config_path):
            return  # no config file — use defaults
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if config and "partitions" in config:
                loaded = {}
                for partition_id, settings in config["partitions"].items():
                    keywords = settings.get("keywords", [])
                    if keywords:
                        loaded[partition_id] = [kw.lower() for kw in keywords]
                if loaded:
                    self.PARTITION_KEYWORDS = loaded
                    print(f"  [routing] Loaded {len(loaded)} partition keyword mappings from {config_path}")
        except Exception as e:
            print(f"  [routing] Could not load {config_path}: {e}. Using defaults.")

    def _build_index_streaming(
        self, chunk_texts: Dict[str, str], batch_size: int = 50000
    ) -> Tuple[Dict[str, List[str]], Dict[str, float], "EntityKnowledgeGraph"]:
        """Build the TF-IDF token index and the entity knowledge graph in
        batches instead of constructing every intermediate structure over
        the whole corpus in one shot.

        build_tfidf_index() and EntityKnowledgeGraph() each build a full
        tokenized/entity-extracted copy of every chunk plus several large
        supporting dicts (document_tokens, per-term frequency counts,
        entity_chunks, neighbours) all at once. On a ~1.9M-chunk corpus
        those coexist in memory simultaneously during construction, which
        is what blows up. Processing chunk_texts in batches keeps peak
        memory bounded by batch_size: each batch's transient locals go
        out of scope (and get gc'd) before the next batch starts.

        Every token and entity string is also passed through sys.intern()
        before being stored. A word like "compliance" appearing 50,000
        times across the corpus would otherwise allocate 50,000 separate
        (equal but distinct) string objects; interning makes all of them
        point at the same one object, so memory scales with vocabulary
        size (tens of thousands of unique terms/entities) instead of
        total token count (hundreds of millions). This changes nothing
        about tokenization, TF-IDF math, or entity matching -- Python
        strings compare and hash by value either way, so
        document_tokens, idf, and every ranking/retrieval method that
        reads them behave identically to before.

        The final document_tokens/idf/entity_graph end up identical in
        *content* to what building everything at once would have
        produced -- this only changes how (and how cheaply) they're
        built, not what they contain.
        """
        import gc

        document_tokens: Dict[str, List[str]] = {}
        self._doc_freq: Dict[str, int] = defaultdict(int)

        entity_graph = EntityKnowledgeGraph.__new__(EntityKnowledgeGraph)
        entity_graph.chunk_entities = {}
        entity_graph.entity_chunks = defaultdict(set)
        entity_graph.neighbours = defaultdict(set)

        chunk_ids = list(chunk_texts.keys())
        total = len(chunk_ids)
        for start in range(0, total, batch_size):
            batch_ids = chunk_ids[start:start + batch_size]

            for chunk_id in batch_ids:
                text = chunk_texts.get(chunk_id)
                if not text:
                    continue

                # --- TF-IDF: tokenize this chunk, intern each token so
                # repeated words share one string object, fold presence
                # into the running doc-freq counter ---
                tokens = [
                    sys.intern(stem_token(token))
                    for token in re.findall(r"\b[\w-]+\b", text.lower())
                ]
                document_tokens[chunk_id] = tokens
                for term in set(tokens):
                    self._doc_freq[term] += 1

                # --- Entity graph: same per-chunk extraction/merge the
                # original EntityKnowledgeGraph.__init__ did, with
                # entities interned for the same reason as tokens ---
                entities = {sys.intern(entity) for entity in extract_entities(text)}
                entity_graph.chunk_entities[chunk_id] = entities
                for entity in entities:
                    entity_graph.entity_chunks[entity].add(chunk_id)
                    entity_graph.neighbours[entity].update(entities - {entity})

            print(f"  Indexed {min(start + batch_size, total):,}/{total:,} chunks...")
            gc.collect()

        document_count = max(1, len(document_tokens))
        idf = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1
            for term, frequency in self._doc_freq.items()
        }
        return document_tokens, idf, entity_graph

    # -------------------------------------------------------------------------
    # Domain detection using entity graph
    # -------------------------------------------------------------------------
    MAX_ACTIVE_PARTITIONS = 3  # cap: don't trigger cross-domain for all 9

    # Default keyword -> partition mapping. Loaded from routing_config.yaml
    # if it exists (per-customer config), otherwise uses these defaults.
    # To customize for a new deployment: edit routing_config.yaml, not this code.
    _DEFAULT_PARTITION_KEYWORDS = {
        "partition_jira": ["jira", "ticket", "epic", "sprint", "backlog", "kanban"],
        "partition_github": ["github", "pr ", "pull request", "pull-request", "repo", "repository", "commit", "branch", "merge request", "merge-request"],
        "partition_slack": ["slack", "channel", "dm ", "direct message", "slack message"],
        "partition_gmail": ["gmail", "email", "inbox", "thread", "mail "],
        "partition_confluence": ["confluence", "wiki", "page", "documentation page"],
        "partition_hubspot": ["hubspot", "deal", "contact", "company", "pipeline", "crm"],
        "partition_linear": ["linear", "linear issue", "project "],
        "partition_fireflies": ["fireflies", "meeting", "transcript", "call recording", "firefly"],
        "partition_google_drive": ["google drive", "drive", "doc ", "document", "sheet", "spreadsheet"],
    }
    PARTITION_KEYWORDS = _DEFAULT_PARTITION_KEYWORDS  # may be overridden in __init__

    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs relevant to this query.

        Two-tier routing:
        1. KEYWORD MATCH: scan the query for partition-specific keywords.
           If "jira" is mentioned, partition_jira is active. This is fast,
           deterministic, and correct for explicit-source questions.
        2. ENTITY DENSITY FALLBACK: if no keywords matched, fall back to
           entity-density routing for questions that don't name a source.
        """
        query_lower = query.lower()

        # --- Tier 1: keyword matching ---
        matched = []
        for partition_id, keywords in self.PARTITION_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    if partition_id not in matched:
                        matched.append(partition_id)
                    break  # one keyword match is enough for this partition

        if matched:
            return matched[:3]

        # --- Tier 2: entity density fallback ---
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        if self.lazy:
            return self.db.get_partition_ids_for_entities(query_entities)[:3]
        partition_scores = defaultdict(int)
        for entity in query_entities:
            for chunk_id in self.entity_graph.entity_chunks.get(entity, []):
                metadata = self.processor.chunks_metadata.get(chunk_id)
                if metadata:
                    partition_scores[metadata.partition_id] += 1
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions][:3]


    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        # 2+ active partitions = cross-domain, no comparison keywords needed
        return True


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
    # Smart router: Standard vs Hybrid
    # -------------------------------------------------------------------------
    def _cheap_standard_confidence(self, query: str) -> float:
        """Fast, retrieval-only probe used by the router: ranks with
        TF-IDF/FTS5 alone, keeps at most 2 chunks, and returns the
        resulting confidence. No generation call, no graph/embeddings/
        cross-encoder -- this is meant to be cheap enough to run before
        deciding whether Hybrid's extra latency and cost are worth it.
        """
        candidate_ids = None if self.lazy else list(self.processor.chunk_texts)
        ranked = self._rank_candidates(query, candidate_ids, use_hybrid=False)
        # confidence_threshold=1.1 is unreachable, so _intelligent_retrieve
        # never "succeeds" early -- it just walks up to 2 chunks and hands
        # back whatever confidence that yields, which is all we want here.
        _, confidence = self._intelligent_retrieve(
            ranked, query, confidence_threshold=1.1, require_cross_domain=False, max_chunks=2,
        )
        return confidence

    @staticmethod
    def _calibrate_hybrid_confidence(scores: Sequence[float]) -> List[float]:
        """Rescale Hybrid's fused ranking scores onto roughly the same
        footing as Standard's calibrated TF-IDF scores, so confidence
        numbers (and the router) aren't comparing apples to oranges.

        Hybrid's ranking is a mix of RRF fusion scores (~0.01-0.05 for a
        couple of fused rankings at rrf_k=60) and, for the top
        CROSS_ENCODER_RERANK_TOP_K items, an already-0-1 cross-encoder
        sigmoid score. A single fixed multiplier tuned for RRF's scale
        (e.g. "x20, then clip to 1.0") pushes the already-well-scaled
        cross-encoder scores straight into the clip, which hides the
        mismatch rather than fixing it, and it silently breaks if rrf_k
        or the number of fused rankings ever changes. Reusing the same
        score/(score+pivot) squashing _calibrate_lexical_scores already
        applies to Standard -- just with a pivot sized for RRF's range --
        keeps ordering, self-normalises regardless of how many rankings
        were fused, and never over/undershoots 0-1. It's a heuristic, not
        a rigorous calibration; treat ROUTER_* thresholds as tunable.
        """
        if not scores:
            return []
        pivot = 0.03  # ballpark for a single fused RRF list at rrf_k=60
        return [score / (score + pivot) if score > 0 else 0.0 for score in scores]

    def _should_use_hybrid(self, query: str) -> Dict[str, Any]:
        """Decide whether Hybrid is likely to outperform Standard for
        `query`, returning the full decision trail (not just a bool) so
        compare() can show its work via router_decision.

        Two hard early-exits cover the unambiguous cases cheaply:
          - Standard confidence already high (> 0.75)  -> Standard.
          - Standard confidence clearly weak  (< 0.40) -> Hybrid.
        Only the ambiguous 0.40-0.75 band falls through to a weighted
        blend of cross-domain + query-complexity signals.

        These early-exits are checked BEFORE the weighted formula rather
        than folded into it, because folding them in creates a real
        contradiction: a confidence of 0.30 should trigger Hybrid per the
        "< 0.40" rule, but the weighted formula alone
        (0.40*(1-0.30) = 0.28, plus 0 for the other two terms) evaluates
        to 0.28 < 0.50 and would recommend Standard -- the opposite of
        the intended rule. Checking the hard bounds first removes that
        contradiction, and also skips the extra work (partition lookup,
        entity extraction) in the clear-cut cases.
        """
        standard_confidence = self._cheap_standard_confidence(query)

        if standard_confidence > self.ROUTER_STANDARD_CONFIDENCE_HIGH:
            return {
                "chosen": "standard",
                "standard_confidence": standard_confidence,
                "cross_domain": False,
                "complexity_score": 0.0,
                "hybrid_score": 0.0,
                "reason": (
                    f"Standard confidence was high ({standard_confidence:.2f} > "
                    f"{self.ROUTER_STANDARD_CONFIDENCE_HIGH}); Hybrid not needed."
                ),
            }
        if standard_confidence < self.ROUTER_STANDARD_CONFIDENCE_LOW:
            return {
                "chosen": "hybrid",
                "standard_confidence": standard_confidence,
                "cross_domain": None,
                "complexity_score": None,
                "hybrid_score": 1.0,
                "reason": (
                    f"Standard confidence was weak ({standard_confidence:.2f} < "
                    f"{self.ROUTER_STANDARD_CONFIDENCE_LOW}); escalating to Hybrid."
                ),
            }

        active_partitions = self._get_active_partitions(query)
        cross_domain = len(active_partitions) >= 2
        term_count = len(query_terms(query))
        entity_count = len(extract_entities(query))
        lower = query.lower()
        comparison_words = sum(
            word in lower for word in
            ("compare", "contrast", "versus", "vs", "between", "difference", "similar")
        )
        complexity_score = min(1.0, (term_count + entity_count + comparison_words) / 15)

        hybrid_score = (
            0.40 * (1 - standard_confidence)
            + 0.30 * (1.0 if cross_domain else 0.0)
            + 0.30 * complexity_score
        )
        chosen = "hybrid" if hybrid_score > self.ROUTER_HYBRID_SCORE_THRESHOLD else "standard"
        return {
            "chosen": chosen,
            "standard_confidence": standard_confidence,
            "cross_domain": cross_domain,
            "complexity_score": complexity_score,
            "hybrid_score": hybrid_score,
            "reason": (
                f"Mixed signal (confidence {standard_confidence:.2f}, cross-domain "
                f"{cross_domain}, complexity {complexity_score:.2f}) -> hybrid_score "
                f"{hybrid_score:.2f} {'>' if chosen == 'hybrid' else '<='} "
                f"{self.ROUTER_HYBRID_SCORE_THRESHOLD}"
            ),
        }

    # -------------------------------------------------------------------------
    # Ranking and fusion
    # -------------------------------------------------------------------------
    @staticmethod
    def _calibrate_lexical_scores(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        # Pivot sized for BM25's typical 5-30 score range. A pivot of 0.10
        # saturated every score to ~0.99 (std dev 0.001 across queries), making
        # the router's confidence signal degenerate. Pivot 10.0 spreads scores
        # across ~0.33-0.75 so the router can discriminate easy vs hard queries.
        pivot = float(os.getenv("LEXICAL_CALIBRATION_PIVOT", "10.0"))
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
        self, query: str, candidate_ids: Optional[Sequence[str]], use_hybrid: bool,
        partition_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        if self.lazy:
            return self._rank_candidates_disk(query, use_hybrid, partition_id)
        return self._rank_candidates_in_memory(query, candidate_ids, use_hybrid)

    def _rank_candidates_in_memory(
        self, query: str, candidate_ids: Sequence[str], use_hybrid: bool
    ) -> List[Tuple[str, float]]:
        """Original small-corpus ranking path: everything already resident
        in self.document_tokens/self.entity_graph/embedding index. Used
        whenever processor.lazy is False -- completely unchanged behavior.
        """
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

    LEXICAL_CANDIDATE_LIMIT = 30  # rows pulled from FTS5/entity_search per query, not the whole corpus

    def _rank_candidates_disk(
        self, query: str, use_hybrid: bool, partition_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Large-corpus ranking path: lexical via FTS5, graph via the
        chunk_entities SQL table -- both disk-backed, so this touches at
        most LEXICAL_CANDIDATE_LIMIT rows regardless of corpus size.
        Cross-encoder/embedding reranking are deliberately NOT run here;
        they're the tier-4 last resort (see _last_resort_rerank), applied
        only to this method's own output when its confidence is low,
        never to the full corpus.
        """
        terms = query_terms(query)
        lexical_raw = self.db.fts_search(terms, partition_id=partition_id, limit=self.LEXICAL_CANDIDATE_LIMIT)
        lexical = self._calibrate_lexical_scores(lexical_raw)
        if not use_hybrid:
            return lexical
        rankings = [lexical]
        entities = extract_entities(query)
        if entities:
            graph = self.db.entity_search(entities, partition_id=partition_id, limit=self.LEXICAL_CANDIDATE_LIMIT)
            if graph:
                rankings.append(graph)
        return self._fuse_rankings(*rankings)

    def _last_resort_rerank(self, query: str, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Tier 4: cross-encoder + on-demand embedding reranking, applied
        only to Hybrid's own top candidates (LAST_RESORT_CANDIDATE_POOL of
        them, never the full corpus) and only when Hybrid's own
        FTS5+graph confidence wasn't good enough on its own. Embeddings
        are computed fresh for just this small set -- no persistent
        full-corpus embedding index is built or held in RAM.
        """
        if not ranked:
            return ranked
        top = ranked[: self.LAST_RESORT_CANDIDATE_POOL]
        chunk_ids = [cid for cid, _ in top]
        texts = self.db.get_chunk_texts(chunk_ids)
        if not texts:
            return ranked
        rankings = [top]
        if self.embedding_ranker.model is not None:
            emb_rank = self.embedding_ranker.rank_on_demand(query, texts)
            if emb_rank:
                max_score = max(s for _, s in emb_rank) or 1.0
                rankings.append([(cid, s / max_score) for cid, s in emb_rank])
        fused = self._fuse_rankings(*rankings)
        if self.cross_encoder.model is not None and len(fused) > 3:
            top_k = fused[:CROSS_ENCODER_RERANK_TOP_K]
            top_k_texts = [(cid, texts.get(cid, "")) for cid, _ in top_k]
            reranked_raw = self.cross_encoder.rerank(query, top_k_texts)
            if reranked_raw:
                reranked = [(cid, 1.0 / (1.0 + math.exp(-score))) for cid, score in reranked_raw]
                reranked_ids = {cid for cid, _ in reranked}
                remainder = [item for item in fused if item[0] not in reranked_ids]
                fused = sorted(reranked + remainder, key=lambda item: item[1], reverse=True)
        # Anything outside the reranked top pool keeps its original
        # (lower-confidence) position, appended after the reranked items.
        reranked_ids = {cid for cid, _ in fused}
        remainder = [item for item in ranked if item[0] not in reranked_ids]
        return fused + remainder

    WEAK_TOP_SCORE = 0.001

    def _get_cross_domain_rankings(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        if self.lazy:
            return self._get_cross_domain_rankings_disk(query, use_hybrid)
        return self._get_cross_domain_rankings_in_memory(query, use_hybrid)

    def _get_cross_domain_rankings_disk(
        self, query: str, use_hybrid: bool = True
    ) -> List[Tuple[str, float]]:
        """Disk-backed equivalent of the in-memory cross-domain blend:
        rank within each active partition via a partition-filtered SQL
        query (own_pool), fall back to an unfiltered (global) query only
        if a partition's own results are weak -- same logic as the
        in-memory version, just backed by fts_search/entity_search
        instead of materialised chunk-id pools.
        """
        active_partitions = self._get_active_partitions(query)
        if not active_partitions:
            return self._rank_candidates(query, None, use_hybrid)
        by_partition: Dict[str, List[Tuple[str, float]]] = {}
        for partition_id in active_partitions:
            ranking = self._rank_candidates(query, None, use_hybrid, partition_id=partition_id)
            if not ranking or ranking[0][1] <= self.WEAK_TOP_SCORE:
                fallback = self._rank_candidates(query, None, use_hybrid)  # unfiltered = global
                ranking = self._merge_rankings_unique(ranking, fallback)
            if ranking:
                by_partition[partition_id] = ranking
        if not by_partition:
            return self._rank_candidates(query, None, use_hybrid)

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

    def _get_cross_domain_rankings_in_memory(
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
    # Confidence, context, validation
    # -------------------------------------------------------------------------
    def _get_texts_batch(self, chunk_ids: Sequence[str]) -> Dict[str, str]:
        """Fetch text for a small, specific set of chunk_ids (never the
        whole corpus) -- from RAM in the small-corpus case, from the
        database in lazy/large-corpus mode."""
        if self.lazy:
            return self.db.get_chunk_texts(chunk_ids)
        return {cid: self.processor.chunk_texts[cid] for cid in chunk_ids if cid in self.processor.chunk_texts}

    def _get_metadata_batch(self, chunk_ids: Sequence[str]) -> Dict[str, ChunkMetadata]:
        """Same idea as _get_texts_batch, for ChunkMetadata."""
        if self.lazy:
            return self.db.get_chunk_metadata_batch(chunk_ids)
        return {cid: self.processor.chunks_metadata[cid] for cid in chunk_ids if cid in self.processor.chunks_metadata}

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
        metadata_batch = self._get_metadata_batch(chunk_ids)
        partitions = {meta.partition_id for meta in metadata_batch.values()}
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
        texts = self._get_texts_batch(chunk_ids)
        metadata_batch = self._get_metadata_batch(chunk_ids)
        sections = []
        for chunk_id in chunk_ids:
            text = texts.get(chunk_id, "")
            metadata = metadata_batch.get(chunk_id)
            if not text or not metadata:
                continue
            partition = metadata.partition_id.replace("partition_", "")
            excerpt = self._truncate_on_sentence_boundary(text, CONTEXT_CHARS_PER_CHUNK)
            sections.append(f"[{chunk_id}|{partition}]\n{excerpt}")
        return "\n\n".join(sections)

    def _source_summary(self, chunk_ids: Sequence[str]) -> List[str]:
        metadata_batch = self._get_metadata_batch(chunk_ids)
        sources = []
        for chunk_id in chunk_ids:
            metadata = metadata_batch.get(chunk_id)
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
            metadata_batch = self._get_metadata_batch(chunk_ids)
            partitions = {meta.partition_id for meta in metadata_batch.values()}
            if len(partitions) < 2:
                return False
            query_entities = extract_entities(query)
            if query_entities:
                texts_batch = self._get_texts_batch(chunk_ids)
                for partition_id in partitions:
                    partition_text = " ".join(
                        texts_batch.get(chunk_id, "")
                        for chunk_id in chunk_ids
                        if metadata_batch.get(chunk_id)
                        and metadata_batch[chunk_id].partition_id == partition_id
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
    # Intelligent retrieval and answer generation
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
        texts_batch = self._get_texts_batch(chunk_ids)
        per_chunk = []
        for chunk_id in chunk_ids:
            text = texts_batch.get(chunk_id, "").lower()
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
                    response = anthropic.Anthropic(api_key=config.api_key, timeout=30.0, max_retries=1).messages.create(
                        model=config.model, max_tokens=MAX_ANSWER_TOKENS,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = "".join(block.text for block in response.content if block.type == "text")
                    actual_input = int(getattr(response.usage, "input_tokens", input_tokens) or input_tokens)
                    actual_output = int(getattr(response.usage, "output_tokens", estimate_tokens(answer)) or estimate_tokens(answer))
                elif config.name == "gemini":
                    if genai is None:
                        raise RuntimeError("install google-genai")
                    response = genai.Client(api_key=config.api_key, http_options={"timeout": 30_000}).models.generate_content(
                        model=config.model, contents=prompt,
                    )
                    answer = response.text or ""
                    usage = getattr(response, "usage_metadata", None)
                    actual_input = int(getattr(usage, "prompt_token_count", input_tokens) or input_tokens)
                    actual_output = int(getattr(usage, "candidates_token_count", estimate_tokens(answer)) or estimate_tokens(answer))
                else:
                    if OpenAI is None:
                        raise RuntimeError("install openai")
                    kwargs: Dict[str, Any] = {"api_key": config.api_key, "timeout": 30.0, "max_retries": 1}
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
    # Main RAG execution (FORCED HYBRID – no cheap router)
    # -------------------------------------------------------------------------
    def _run_rag(self, query: str, method: str, use_hybrid: bool, provider_configs: List[ProviderConfig]) -> Dict[str, Any]:
        start = time.perf_counter()
        is_cross_domain = self._is_cross_domain_question(query) if use_hybrid else False
        threshold = self._get_adaptive_threshold(query, is_cross_domain)
        max_chunks = self._max_chunks_for_query(query, is_cross_domain)

        if use_hybrid:
            ranked = self._get_cross_domain_rankings(query, use_hybrid=True)
            # Hybrid's fused (RRF + cross-encoder) scores live on a different
            # scale than Standard's calibrated TF-IDF scores -- calibrate
            # before confidence/threshold comparisons so the two pipelines
            # are judged on comparable footing (see _calibrate_hybrid_confidence).
            calibrated_scores = self._calibrate_hybrid_confidence([score for _, score in ranked])
            ranked = list(zip((chunk_id for chunk_id, _ in ranked), calibrated_scores))
            chunk_ids, confidence = self._intelligent_retrieve(
                ranked, query, threshold, is_cross_domain, max_chunks
            )
            retrieval_path = "hybrid"
            # Tier 4 (large-corpus mode only): the disk-backed FTS5+graph
            # Hybrid tier above never touches cross-encoder/embeddings --
            # that's what keeps its latency low across the whole query set.
            # If ITS OWN confidence still isn't good enough, escalate just
            # this query's own top candidates (never the full corpus) to
            # cross-encoder + on-demand embedding reranking, as a rare
            # last resort rather than a per-query default.
            if self.lazy and confidence < self.ROUTER_LAST_RESORT_THRESHOLD and (
                self.cross_encoder.model is not None or self.embedding_ranker.model is not None
            ):
                reranked = self._last_resort_rerank(query, ranked)
                escalated_ids, escalated_confidence = self._intelligent_retrieve(
                    reranked, query, threshold, is_cross_domain, max_chunks
                )
                if escalated_confidence > confidence:
                    chunk_ids, confidence = escalated_ids, escalated_confidence
                    retrieval_path = "hybrid_last_resort"
        else:
            # Standard RAG (lexical only: TF-IDF in-memory, or FTS5 disk-backed)
            candidate_ids = None if self.lazy else list(self.processor.chunk_texts)
            ranked = self._rank_candidates(query, candidate_ids, False)
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

    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None,
                include_direct_ai: bool = True, force_hybrid: bool = False) -> Dict[str, Dict[str, Any]]:
        """Runs Standard and Hybrid RAG (both, unconditionally -- Hybrid is
        NOT routed/skipped here; see the hardcoded 'Forced Hybrid' decision
        below). This is deliberate for evaluation: every question gets a
        real answer from both pipelines so they can be compared fairly,
        rather than the smart-router behavior (used elsewhere) where
        Hybrid gets skipped whenever Standard looks confident enough.

        include_direct_ai=False skips the third, no-retrieval "ask the
        model directly" generation call -- a real, separate API call on
        every question, wasted if the caller (e.g. a batch eval script)
        never reads results["direct_ai"].
        """
        provider_configs = (
            available_provider_configs(provider, key_override=api_key)
            if provider is not None
            else self.provider_configs
        )
        if force_hybrid:
            decision = {"chosen": "hybrid", "standard_confidence": 0.0, "reason": "Forced Hybrid (eval mode)"}
        else:
            decision = self._should_use_hybrid(query)
        standard = self._standard_rag(query, provider_configs)

        if decision["chosen"] == "hybrid":
            hybrid = self._hybrid_rag(query, provider_configs)
            decision["hybrid_confidence_calibrated"] = hybrid["confidence_score"]
            decision["standard_was_used"] = False
            decision["hybrid_was_skipped"] = False
        else:
            hybrid = dict(standard)
            hybrid["generation"] = dict(standard.get("generation", {}))
            decision["hybrid_confidence_calibrated"] = None
            decision["standard_was_used"] = True
            decision["hybrid_was_skipped"] = True

        direct = self._direct_ai(query, provider_configs) if include_direct_ai else None
        saved = max(0, standard["tokens_used"] - hybrid["tokens_used"])
        hybrid["tokens_saved_vs_standard"] = saved
        hybrid["token_savings_percent_vs_standard"] = (
            100.0 * saved / standard["tokens_used"]
            if standard["tokens_used"] else 0.0
        )
        hybrid["router_decision"] = decision
        results = {
            "hybrid_rag": hybrid,
            "standard_rag": standard,
        }
        if direct is not None:
            # Omitted entirely (not set to None) when skipped: both
            # save_comparison and format_comparison iterate results.items()
            # assuming every value is a populated result dict -- a None
            # value would crash both on the first .get()/[...] access.
            results["direct_ai"] = direct
        self.db.save_comparison(query, results)
        print(f"[router] {decision['reason']}")
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
            "COMPARISON RESULTS (SMART ROUTER v23)",
            "=" * 88,
        ]
        router = results.get("hybrid_rag", {}).get("router_decision")
        if router:
            lines.append(
                f"Router: {router['chosen'].title()} "
                f"(confidence: {router['standard_confidence']:.2f}, "
                f"cross-domain: {'Yes' if router.get('cross_domain') else 'No'}, "
                f"complexity: {router.get('complexity_score') or 0:.2f})"
            )
            lines.append(router["reason"])
            if router.get("hybrid_was_skipped"):
                lines.append("Hybrid: NOT RUN (Standard was sufficient)")
            lines.append("-" * 88)
        lines.extend([
            "NOTE: 'Chunk relevance', 'confidence', and 'quality' are grounding",
            "metrics -- they measure whether retrieved evidence overlaps the query and",
            "spans the right domains. They are legitimately 0/no for Direct AI, since it",
            "retrieves nothing. Use 'Answer relevance' (query-term coverage of the",
            "actual answer text) for a metric that's computed the same way across all",
            "pipelines.",
            "-" * 88,
            "Method        Latency    Ctx tokens  Chunks  Chunk rel  Answer rel  Confidence  Quality",
            "-" * 88,
        ])
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
# Export and CLI
# ---------------------------------------------------------------------------
def export_markdown(query: str, results: Dict[str, Dict[str, Any]]) -> Path:
    output_dir = Path("rag_exports")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = [
        f"# Forced Hybrid RAG comparison (v23)\n\nQuestion: {query}\n",
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


def build_large_corpus_engine(db_path: str = DATABASE_PATH, database_url: str = None, provider: str = "auto", api_key: Optional[str] = None) -> Tuple[DatabaseManager, DocumentProcessor, "ComparisonEngine"]:
    """Construct db/processor/engine in large-corpus (memory-bounded) mode:
    DocumentProcessor never loads the full corpus into RAM, and
    ComparisonEngine retrieves via disk-backed FTS5/entity-table queries
    instead of in-memory TF-IDF/entity-graph structures. Use this for
    corpora too large to hold as Python objects (e.g. ~1.9M chunks);
    use plain DocumentProcessor(db) + ComparisonEngine(...) (main()'s
    default) for small corpora where the original in-RAM path is fine
    and slightly faster.
    """
    db = DatabaseManager(db_path=db_path, database_url=database_url)
    processor = DocumentProcessor(db, lazy=True)
    engine = ComparisonEngine(processor, db, provider=provider, api_key=api_key)
    return db, processor, engine
