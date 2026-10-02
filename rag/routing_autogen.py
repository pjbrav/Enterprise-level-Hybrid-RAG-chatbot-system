"""Generate routing keywords for any corpus, so partition routing does not depend
on the built-in Jira/Slack/GitHub list or a hand-written routing_config.yaml.

For every partition the keywords come from three sources:
  1. the partition's own name (a folder called 'hr' routes questions that say 'hr')
  2. its most distinctive terms: TF-IDF with each partition treated as one
     document, so words common to every partition score low
  3. entities (names, products, orgs) that are concentrated in that partition

Terms shared by more than half of the partitions are dropped (they cannot tell
partitions apart). Routing matches keywords as substrings of the lower-cased
question, so very short tokens get a trailing space ('hr ') to avoid matching
inside longer words.
"""

from __future__ import annotations

import re
from typing import Dict, List

from rag.text import _ENTITY_STOPWORDS

GENERIC_NAME_TOKENS = {
    "partition", "general", "documents", "document", "docs", "files", "file", "data", "misc",
    "uploads", "input", "inputs", "source", "sources", "corpus", "pdfs", "pdf", "all", "other",
}
_WORDLIKE = re.compile(r"^[a-z][a-z \-]{3,40}$")


def _format_keyword(token: str) -> str:
    return token if len(token) >= 4 else token + " "


def _name_tokens(partition_id: str) -> List[str]:
    name = partition_id[len("partition_"):] if partition_id.startswith("partition_") else partition_id
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t and t not in GENERIC_NAME_TOKENS and not t.isdigit()]


def _distinctive_terms(texts: Dict[str, str], top_n: int) -> Dict[str, List[str]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return {pid: [] for pid in texts}
    pids = [p for p, t in texts.items() if t.strip()]
    if not pids:
        return {pid: [] for pid in texts}
    # With <3 partitions "appears in most partitions" carries no signal.
    max_df = 0.6 if len(pids) >= 3 else 1.0
    try:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=max_df,
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{3,}\b")
        matrix = vec.fit_transform([texts[p] for p in pids])
    except ValueError:  # empty vocabulary
        return {pid: [] for pid in texts}
    names = vec.get_feature_names_out()
    out: Dict[str, List[str]] = {pid: [] for pid in texts}
    for row, pid in enumerate(pids):
        scores = matrix[row].toarray().ravel()
        order = scores.argsort()[::-1]
        out[pid] = [names[i].lower() for i in order[:top_n] if scores[i] > 0]
    return out


def _concentrated_entities(db, partition_id: str, top_n: int, min_concentration: float = 0.6) -> List[str]:
    """Entities that occur mostly in this partition (>= 60% of all their mentions)."""
    try:
        rows = db._query("SELECT entity, COUNT(*) AS c FROM chunk_entities WHERE partition_id = ? "
                         "GROUP BY entity ORDER BY c DESC LIMIT 60", (partition_id,), fetch=True)
        if not rows:
            return []
        entities = [r["entity"] for r in rows]
        placeholders = ",".join("?" for _ in entities)
        totals = {r["entity"]: r["freq"] for r in db._query(
            f"SELECT entity, freq FROM entity_frequency WHERE entity IN ({placeholders})", entities, fetch=True)}
    except Exception:
        return []
    picked: List[str] = []
    for r in rows:
        entity, count = r["entity"], r["c"]
        total = totals.get(entity, count)
        if (_WORDLIKE.match(entity) and entity not in _ENTITY_STOPWORDS and count >= 2
                and count / max(1, total) >= min_concentration):
            picked.append(entity)
        if len(picked) >= top_n:
            break
    return picked


def generate_partition_keywords(db, sample_per_partition: int = 2000, top_terms: int = 8,
                                top_entities: int = 5) -> Dict[str, List[str]]:
    """Routing keywords for every partition in `db` ({partition_id: [keyword, ...]})."""
    partitions = db.get_partition_ids()
    if not partitions:
        return {}
    texts: Dict[str, str] = {}
    for pid in partitions:
        rows = db._query("SELECT chunk_text FROM chunks WHERE partition_id = ? LIMIT ?",
                         (pid, sample_per_partition), fetch=True)
        texts[pid] = " ".join(r["chunk_text"] for r in rows)
    terms = _distinctive_terms(texts, top_terms)

    per_partition: Dict[str, List[str]] = {}
    for pid in partitions:
        names = _name_tokens(pid)
        candidates = names + terms.get(pid, []) + _concentrated_entities(db, pid, top_entities)
        per_partition[pid] = list(dict.fromkeys(c.strip().lower() for c in candidates if c and c.strip()))

    # A term that most partitions list cannot discriminate between them.
    if len(partitions) >= 4:
        counts: Dict[str, int] = {}
        for kws in per_partition.values():
            for kw in set(kws):
                counts[kw] = counts.get(kw, 0) + 1
        too_common = {kw for kw, n in counts.items() if n > len(partitions) / 2}
        for pid, kws in per_partition.items():
            keep_names = set(_name_tokens(pid))
            per_partition[pid] = [kw for kw in kws if kw in keep_names or kw not in too_common]

    return {pid: [_format_keyword(kw) for kw in kws] for pid, kws in per_partition.items() if kws}
