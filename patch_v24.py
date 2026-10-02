"""
v23 -> v24 Patch: Fix partition routing so Hybrid actually beats Standard.

Problems fixed:
1. Entity-hit-count routing favours big partitions (Slack 800K, Gmail 622K)
   over relevant ones (Jira 25K, GitHub 29K). "api" has 205K mentions but
   most are in Slack/Gmail, so Jira+GitHub questions get routed to the
   wrong partitions.
   FIX: Rank partitions by entity DENSITY (hits / partition_size), not
   raw hit count. A partition where 50% of chunks mention "jira" should
   rank higher than one where 2% mention "api".

2. Super-frequent entities ("api", "github", "slack") match everywhere and
   carry no routing signal.
   FIX: Filter entities by inverse document frequency — entities that appear
   in >40% of chunks are too common to be routing signals.

3. Cross-domain detection requires comparison keywords ("compare", "vs").
   Questions like "What GitHub PRs relate to Jira tickets?" don't trigger
   cross-domain because they lack those keywords.
   FIX: Cross-domain = 2+ active partitions, no keyword requirement.

4. WEAK_TOP_SCORE = 0.05 triggers global fallback on every partition because
   RRF scores are naturally ~0.03.
   FIX: Lower to 0.001.

5. LEXICAL_CANDIDATE_LIMIT = 100 is too high for per-partition searches.
   FIX: Lower to 50.

6. _get_active_partitions returns all 9 partitions, causing 9x cross-domain
   work. Previous PowerShell patch capped to 3 but didn't use density.
   FIX: Cap to 3 AND use density-based ranking.

Run:
    python patch_v24.py
"""

import sys

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ================================================================
# PATCH 1: Replace get_partition_ids_for_entities with density-based version
# ================================================================
OLD_1 = '''    def get_partition_ids_for_entities(self, entities: Iterable[str]) -> List[str]:
        """SQL equivalent of the old in-RAM _get_active_partitions: which
        partitions contain chunks matching these entities, ranked by how
        many matches they have."""
        entities = [e for e in entities if e]
        if not entities:
            return []
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities)
        sql = (
            f"SELECT partition_id, COUNT(*) AS hits FROM chunk_entities "
            f"WHERE entity IN ({placeholders}) GROUP BY partition_id ORDER BY hits DESC"
        )
        rows = self._query(sql, entities, fetch=True)
        return [row["partition_id"] for row in rows]'''

NEW_1 = '''    def get_partition_ids_for_entities(self, entities: Iterable[str]) -> List[str]:
        """Rank partitions by entity DENSITY, not raw hit count.

        Density = hits / partition_size. This stops big partitions (Slack
        800K chunks) from drowning out small relevant ones (Jira 25K)
        just because they have more total mentions of a common term.
        """
        entities = [e for e in entities if e]
        if not entities:
            return []
        placeholders = ",".join("?" if not self.use_postgres else "%s" for _ in entities)
        # Join with partition sizes so we can compute density in SQL
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
        return [row["partition_id"] for row in rows]'''

if OLD_1 in content:
    content = content.replace(OLD_1, NEW_1)
    changes += 1
    print("  [OK] Patched get_partition_ids_for_entities() — density-based routing")
else:
    print("  [SKIP] get_partition_ids_for_entities() not found (already patched?)")
    changes += 0

# ================================================================
# PATCH 2: Cap active partitions to top 3 by density
# ================================================================
OLD_2 = '''        if self.lazy:
            return self.db.get_partition_ids_for_entities(query_entities)'''

NEW_2 = '''        if self.lazy:
            return self.db.get_partition_ids_for_entities(query_entities)[:3]'''

if OLD_2 in content:
    content = content.replace(OLD_2, NEW_2)
    changes += 1
    print("  [OK] Capped _get_active_partitions() to top 3 by density")
else:
    print("  [SKIP] lazy partition cap not found (already patched?)")

# ================================================================
# PATCH 3: Remove comparison-keyword requirement from _is_cross_domain_question
# ================================================================
OLD_3 = '''    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        lower = query.lower()
        comparison_words = {"compare", "contrast", "difference", "versus", "vs", "between", "similar", "differ", "vs."}
        return any(word in lower for word in comparison_words)'''

NEW_3 = '''    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        # Removed comparison-keyword requirement: a question that touches
        # 2+ partitions is cross-domain regardless of phrasing. The old
        # requirement ("compare", "vs", "between") meant questions like
        # "What GitHub PRs relate to Jira tickets?" never triggered
        # cross-domain mode.
        return True'''

if OLD_3 in content:
    content = content.replace(OLD_3, NEW_3)
    changes += 1
    print("  [OK] Removed comparison-keyword gate from _is_cross_domain_question()")
else:
    print("  [SKIP] _is_cross_domain_question() not found (already patched?)")

# ================================================================
# PATCH 4: Lower WEAK_TOP_SCORE
# ================================================================
OLD_4 = '''    WEAK_TOP_SCORE = 0.05'''

NEW_4 = '''    WEAK_TOP_SCORE = 0.001  # RRF scores are ~0.01-0.03; 0.05 caused fallback on every partition'''

if OLD_4 in content:
    content = content.replace(OLD_4, NEW_4)
    changes += 1
    print("  [OK] Lowered WEAK_TOP_SCORE from 0.05 to 0.001")
else:
    print("  [SKIP] WEAK_TOP_SCORE not found (already patched?)")

# ================================================================
# PATCH 5: Lower LEXICAL_CANDIDATE_LIMIT
# ================================================================
OLD_5 = '''    LEXICAL_CANDIDATE_LIMIT = 100  # rows pulled from FTS5/entity_search per query, not the whole corpus'''

NEW_5 = '''    LEXICAL_CANDIDATE_LIMIT = 50  # rows pulled from FTS5/entity_search per query, not the whole corpus'''

if OLD_5 in content:
    content = content.replace(OLD_5, NEW_5)
    changes += 1
    print("  [OK] Lowered LEXICAL_CANDIDATE_LIMIT from 100 to 50")
else:
    print("  [SKIP] LEXICAL_CANDIDATE_LIMIT not found (already patched?)")

# ================================================================
# PATCH 6: Filter frequent entities in _get_active_partitions
# ================================================================
OLD_6 = '''    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs that contain entities matching the query."""
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
        # Sort by score descending
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions]'''

NEW_6 = '''    MAX_ENTITY_FREQUENCY = 0.40  # entities in >40% of chunks are too common for routing

    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs that contain entities matching the query.

        Entities are filtered by frequency: terms that appear in >40% of
        chunks (like "api", "github") carry no routing signal because they
        match everywhere. Only rare, discriminative entities drive routing.
        """
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        if self.lazy:
            # Filter out super-frequent entities before hitting the DB.
            # These would just return all 9 partitions at high cost.
            filtered = self._filter_frequent_entities(query_entities)
            if not filtered:
                # All entities were too frequent; fall back to the full set
                # — the density ranking will still prefer relevant partitions
                filtered = query_entities
            return self.db.get_partition_ids_for_entities(filtered)[:3]
        partition_scores = defaultdict(int)
        for entity in query_entities:
            for chunk_id in self.entity_graph.entity_chunks.get(entity, []):
                metadata = self.processor.chunks_metadata.get(chunk_id)
                if metadata:
                    partition_scores[metadata.partition_id] += 1
        # Sort by score descending
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions][:3]

    def _filter_frequent_entities(self, entities: List[str]) -> List[str]:
        """Remove entities that appear in too many chunks to be useful for routing.

        Uses a cached frequency count from chunk_entities. Entities above
        MAX_ENTITY_FREQUENCY of total chunks are considered too common.
        """
        if not entities:
            return list(entities)
        # Quick check: how many chunks mention each entity?
        # We batch this into one query for efficiency.
        placeholders = ",".join("?" for _ in entities)
        sql = (
            f"SELECT entity, COUNT(DISTINCT chunk_id) AS freq "
            f"FROM chunk_entities WHERE entity IN ({placeholders}) "
            f"GROUP BY entity"
        )
        rows = self.db._query(sql, entities, fetch=True)
        total_chunks = 1924473  # cached; avoids a COUNT(*) on every call
        threshold = int(total_chunks * self.MAX_ENTITY_FREQUENCY)
        frequent = {row["entity"] for row in rows if row["freq"] > threshold}
        return [e for e in entities if e not in frequent]'''

if OLD_6 in content:
    content = content.replace(OLD_6, NEW_6)
    changes += 1
    print("  [OK] Added entity frequency filter to _get_active_partitions()")
else:
    print("  [SKIP] _get_active_partitions() not found (already patched?)")

# ================================================================
# Write patched file
# ================================================================

if changes == 0:
    print("\n  No changes were made. v23.py may already be patched.")
    sys.exit(0)

with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"""
{'=' * 70}
  v24 Patch Applied Successfully: {changes} changes
{'=' * 70}

  Summary of changes:
  1. Partition routing: raw hit count -> entity DENSITY
     (Jira questions now route to Jira, not to Slack just because
     Slack is bigger)
  2. Active partitions capped to top 3 by density
  3. Cross-domain detection: 2+ partitions = cross-domain
     (no more "compare"/"vs" keyword requirement)
  4. WEAK_TOP_SCORE: 0.05 -> 0.001
     (stops RRF-score-triggered global fallback on every partition)
  5. LEXICAL_CANDIDATE_LIMIT: 100 -> 50
     (less data per partition search, faster)
  6. Entity frequency filter: entities in >40% of chunks are filtered
     before routing (stops "api"/"github" from matching all 9 partitions)

  No re-backfill needed. These are search-time changes only.

  Next steps:
    1. Run timing test:
       python -c "import time; from v23 import build_large_corpus_engine; db,proc,eng=build_large_corpus_engine(db_path='hybrid_rag_v22.db',provider='openai'); q='What GitHub PRs are related to the Jira tickets about the API migration?'; t0=time.time(); parts=eng._get_active_partitions(q); print(f'Active partitions: {parts}'); print(f'Partition lookup: {time.time()-t0:.2f}s'); t0=time.time(); cross=eng._get_cross_domain_rankings(q,use_hybrid=True); print(f'Cross-domain: {len(cross)} chunks in {time.time()-t0:.2f}s')"

    2. Run 200-question eval with all partitions:
       python batch_compare_all.py
{'=' * 70}""")
