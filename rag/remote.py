"""Member side of host/join: join a host, query it remotely, fuse with local corpora.

Nothing from a host is ever written to disk here. A join stores only the host
URL, the membership token and a few routing keywords (see state.JoinRecord).
Retrieved chunk texts live in RAM for one request (a ChunkStore), are put in
the prompt, and are dropped. Answers are generated on THIS machine with THIS
machine's API key.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from rag.corpora import CorpusManager
from rag.db import ChunkMetadata
from rag.params import DEFAULT_INTENSITY
from rag.providers import ProviderConfig
from rag.retrieval import ChunkStore, RetrievalResult, wire_to_result
from rag.state import JoinRecord

# Hybrid retrieval on a cold, very large corpus can take over a minute.
REMOTE_TIMEOUT = float(os.getenv("REMOTE_TIMEOUT", "120"))
LOCAL_POOL_CAP = 50


class RemoteError(Exception):
    """A host request failed (message is safe to show to the user)."""


class HostUnavailable(RemoteError):
    """Host offline / unreachable / timed out."""


class MembershipRevoked(RemoteError):
    """The host no longer accepts this membership (kicked, or host reset)."""


class JoinFailed(RemoteError):
    """Joining was refused (bad / expired code, rate limited, ...)."""


class NoSourcesAvailable(Exception):
    def __init__(self, warnings: List[str]):
        super().__init__("No corpus could be searched. " + " ".join(warnings))
        self.warnings = warnings


def normalize_host_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if not url:
        raise JoinFailed("Enter the host URL (for example http://192.168.1.20:8765).")
    if "://" not in url:
        url = "http://" + url
    return url


def join_host(manager: CorpusManager, host_url: str, otp: str, display_name: str,
              timeout: float = 15.0) -> JoinRecord:
    """Redeem a one-time code with a host and remember the membership token."""
    base = normalize_host_url(host_url)
    try:
        health = httpx.get(f"{base}/v1/health", timeout=timeout)
        health.raise_for_status()
        resp = httpx.post(f"{base}/v1/join", json={"otp": otp.strip(), "display_name": display_name.strip() or "member"},
                          timeout=timeout)
    except httpx.HTTPStatusError:
        raise JoinFailed(f"{base} did not answer like a Hybrid RAG host.")
    except httpx.HTTPError as exc:
        raise JoinFailed(f"Could not reach {base} ({type(exc).__name__}). Check the URL and that the host is running.")
    if resp.status_code == 429:
        raise JoinFailed("Too many failed attempts from this address. Wait a few minutes and try again.")
    if resp.status_code == 401:
        raise JoinFailed("That code is invalid or has expired. Ask the host for a new one.")
    if resp.status_code != 200:
        raise JoinFailed(f"Host refused the join (HTTP {resp.status_code}).")
    data = resp.json()
    return manager.state.add_join(base, data["corpus_name"], display_name.strip() or "member",
                                  data["token"], data.get("partition_keywords") or {})


class RemoteCorpus:
    def __init__(self, join: JoinRecord, timeout: float = REMOTE_TIMEOUT):
        self.join = join
        self.timeout = timeout

    @property
    def name(self) -> str:
        return self.join.corpus_name

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.join.token}"}

    def retrieve(self, query: str, method: str = "hybrid", intensity: float = 0.5, pool: int = 20) -> RetrievalResult:
        try:
            resp = httpx.post(f"{self.join.host_url}/v1/retrieve", headers=self._headers(), timeout=self.timeout,
                              json={"query": query, "method": method, "intensity": intensity, "pool": pool})
        except httpx.HTTPError as exc:
            raise HostUnavailable(f"host unreachable ({type(exc).__name__})")
        if resp.status_code == 401:
            raise MembershipRevoked("the host removed this membership")
        if resp.status_code != 200:
            raise RemoteError(f"host returned HTTP {resp.status_code}")
        return wire_to_result(resp.json())

    def leave(self) -> None:
        """Tell the host we are leaving (best effort; the local token is dropped regardless)."""
        try:
            httpx.post(f"{self.join.host_url}/v1/leave", headers=self._headers(), timeout=5)
        except httpx.HTTPError:
            pass


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def local_chunk_store(db, result: RetrievalResult, limit: int = LOCAL_POOL_CAP) -> ChunkStore:
    """Texts + metadata for a local corpus' top candidates (so they can be fused with remote ones)."""
    ids = [cid for cid, _ in result.ranked[:limit]]
    return ChunkStore(db.get_chunk_texts(ids), db.get_chunk_metadata_batch(ids))


def fuse_results(engine, parts: Sequence[Tuple[str, RetrievalResult]], start: float) -> RetrievalResult:
    """Merge candidate lists from several corpora into one RetrievalResult.

    Every chunk id and partition id is namespaced with its origin label
    ('host', 'mine', ...) so two corpora can never collide, and a selection
    spanning a host and a local corpus counts as cross-domain evidence.
    Two or more non-empty lists are combined with the same RRF + calibration
    the engine uses inside a single corpus; a single list passes through
    untouched, so a host-only search behaves exactly like a local one."""
    store = ChunkStore()
    lists: List[List[Tuple[str, float]]] = []
    for label, res in parts:
        if res.store is None:
            raise ValueError(f"part {label!r} has no ChunkStore")
        ranked: List[Tuple[str, float]] = []
        for cid, score in res.ranked:
            text, meta = res.store.texts.get(cid), res.store.meta.get(cid)
            if not text or meta is None:
                continue
            nid = f"{label}:{cid}"
            ranked.append((nid, score))
            store.texts[nid] = text
            store.meta[nid] = ChunkMetadata(nid, meta.file_path, f"{label}:{meta.partition_id}",
                                            meta.chunk_index, meta.token_count, meta.entity_count)
        if ranked:
            lists.append(ranked)
    if len(lists) > 1:
        fused = engine._fuse_rankings(*lists)
        scores = engine._calibrate_hybrid_confidence([sc for _, sc in fused])
        ranked_all = list(zip((cid for cid, _ in fused), scores))
    else:
        ranked_all = lists[0] if lists else []
    # Each corpus judged the question by its own routing (threshold, chunk budget,
    # cross-domain); the fused result takes the most permissive of those judgments.
    return RetrievalResult(
        ranked=ranked_all,
        is_cross_domain=any(res.is_cross_domain for _, res in parts),
        threshold=min((res.threshold for _, res in parts), default=0.0),
        max_chunks=max((res.max_chunks for _, res in parts), default=5),
        use_hybrid=any(res.use_hybrid for _, res in parts), start=start, store=store,
    )


class FederatedSearch:
    """Runs one question across any mix of local corpora and joined hosts."""

    def __init__(self, manager: CorpusManager, pool: int = 20, remote_timeout: float = REMOTE_TIMEOUT):
        self.manager = manager
        self.pool = pool
        self.remote_timeout = remote_timeout

    def _generation_engine(self, local_ids: Sequence[str]):
        if local_ids:
            return self.manager.get_engine(local_ids[0])[2]
        return self.manager.scratch_engine()

    def run(self, query: str, local_ids: Sequence[str], join_ids: Sequence[int], use_hybrid: bool,
            provider_configs: List[ProviderConfig], method: Optional[str] = None,
            intensity: float = DEFAULT_INTENSITY) -> Dict[str, Any]:
        method = method or ("Hybrid RAG" if use_hybrid else "Standard RAG")
        if not local_ids and not join_ids:
            raise ValueError("Select at least one corpus to search.")

        # Fast path: one local corpus, no hosts -> the unchanged single-corpus pipeline.
        if len(local_ids) == 1 and not join_ids:
            engine = self.manager.get_engine(local_ids[0])[2]
            out = engine._run_rag(query, method, use_hybrid, provider_configs,
                                  params=engine.params_for_intensity(intensity) if use_hybrid else None)
            out["warnings"], out["origins"] = [], {"mine": out["chunks_retrieved"]}
            return out

        start = time.perf_counter()
        warnings: List[str] = []
        engine = self._generation_engine(local_ids)
        parts: List[Tuple[str, RetrievalResult]] = []
        joins = [self.manager.state.get_join(j) for j in join_ids]
        for j in joins:
            if j is None:
                warnings.append("A selected shared corpus no longer exists.")

        def run_local(idx_cid: Tuple[int, str]):
            idx, cid = idx_cid
            db, _, eng = self.manager.get_engine(cid)
            res = eng.retrieve(query, use_hybrid, eng.params_for_intensity(intensity) if use_hybrid else None)
            res.store = local_chunk_store(db, res)
            return (self._label("mine", idx, len(local_ids)), res)

        def run_remote(idx_join):
            idx, j = idx_join
            label = self._label("host", idx, len([x for x in joins if x]))
            try:
                res = RemoteCorpus(j, self.remote_timeout).retrieve(
                    query, "hybrid" if use_hybrid else "standard", intensity=intensity, pool=self.pool)
                return label, res, None
            except MembershipRevoked as exc:
                self.manager.state.set_join_status(j.id, "revoked", str(exc))
                return label, None, f"Shared corpus '{j.corpus_name}' unavailable: {exc}. Rejoin with a new code."
            except RemoteError as exc:
                return label, None, f"Shared corpus '{j.corpus_name}' unavailable: {exc}."

        live_joins = [j for j in joins if j is not None]
        with ThreadPoolExecutor(max_workers=max(1, len(local_ids) + len(live_joins))) as pool:
            local_futures = [pool.submit(run_local, x) for x in enumerate(local_ids)]
            remote_futures = [pool.submit(run_remote, x) for x in enumerate(live_joins)]
            for f in local_futures:
                parts.append(f.result())
            for f in remote_futures:
                label, res, warning = f.result()
                if warning:
                    warnings.append(warning)
                elif res is not None:
                    parts.append((label, res))

        if not parts:
            raise NoSourcesAvailable(warnings)
        fused = fuse_results(engine, parts, start)
        chunk_ids, confidence, path = engine.select(
            query, fused, params=engine.params_for_intensity(intensity) if use_hybrid else None)
        out = engine.answer(query, method, fused, chunk_ids, confidence, path, provider_configs)
        origins: Dict[str, int] = defaultdict(int)
        for cid in chunk_ids:
            origins[cid.split(":", 1)[0]] += 1
        out["origins"], out["warnings"] = dict(origins), warnings
        return out

    @staticmethod
    def _label(base: str, idx: int, total: int) -> str:
        return base if total <= 1 else f"{base}{idx + 1}"
