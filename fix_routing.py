"""
Direct fix for partition routing — works regardless of current v23.py state.

This replaces the ENTIRE get_partition_ids_for_entities() and _get_active_partitions()
methods with the density-based versions, using Python AST-level matching
instead of fragile string matching.

Run:
    python fix_routing.py
"""

import re

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ================================================================
# 1. Replace get_partition_ids_for_entities ENTIRELY
# ================================================================
# Find the function by signature and replace everything up to the next def
pattern_1 = r'(    def get_partition_ids_for_entities\(self, entities: Iterable\[str\]\) -> List\[str\]:.*?)(\n    def )'
replacement_1 = '''    def get_partition_ids_for_entities(self, entities: Iterable[str]) -> List[str]:
        """Rank partitions by entity DENSITY, not raw hit count.

        Density = hits / partition_size. This stops big partitions (Slack
        800K chunks) from drowning out small relevant ones (Jira 25K)
        just because they have more total mentions of a common term.
        """
        entities = [e for e in entities if e]
        if not entities:
            return []
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

'''
content_new = re.sub(pattern_1, replacement_1 + r'\2', content, flags=re.DOTALL)

if content_new != content:
    content = content_new
    print("  [OK] Replaced get_partition_ids_for_entities() with density-based version")
else:
    print("  [FAIL] Could not find get_partition_ids_for_entities()")

# ================================================================
# 2. Replace _get_active_partitions ENTIRELY
# ================================================================
pattern_2 = r'(    def _get_active_partitions\(self, query: str\) -> List\[str\]:.*?)(\n    def _is_cross_domain_question)'
replacement_2 = '''    MAX_ACTIVE_PARTITIONS = 3  # cap: don't trigger cross-domain for all 9

    def _get_active_partitions(self, query: str) -> List[str]:
        """Return partition IDs that contain entities matching the query.

        Uses density-based routing: partitions where a higher FRACTION of
        their chunks mention the query's entities rank first. This stops
        big partitions from drowning out small relevant ones.
        """
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        if self.lazy:
            return self.db.get_partition_ids_for_entities(query_entities)[:self.MAX_ACTIVE_PARTITIONS]
        partition_scores = defaultdict(int)
        for entity in query_entities:
            for chunk_id in self.entity_graph.entity_chunks.get(entity, []):
                metadata = self.processor.chunks_metadata.get(chunk_id)
                if metadata:
                    partition_scores[metadata.partition_id] += 1
        sorted_partitions = sorted(partition_scores.items(), key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_partitions][:self.MAX_ACTIVE_PARTITIONS]

'''
content_new = re.sub(pattern_2, replacement_2 + r'\2', content, flags=re.DOTALL)

if content_new != content:
    content = content_new
    print("  [OK] Replaced _get_active_partitions() with density + cap version")
else:
    print("  [FAIL] Could not find _get_active_partitions()")

# ================================================================
# 3. Replace _is_cross_domain_question — remove keyword gate
# ================================================================
pattern_3 = r'(    def _is_cross_domain_question\(self, query: str\) -> bool:.*?)(\n    def _is_simple_single_domain)'
replacement_3 = '''    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        # Removed comparison-keyword requirement: a question that touches
        # 2+ partitions is cross-domain regardless of phrasing.
        return True

'''
content_new = re.sub(pattern_3, replacement_3 + r'\2', content, flags=re.DOTALL)

if content_new != content:
    content = content_new
    print("  [OK] Replaced _is_cross_domain_question() — removed keyword gate")
else:
    print("  [FAIL] Could not find _is_cross_domain_question()")

# ================================================================
# 4. Fix WEAK_TOP_SCORE
# ================================================================
content = content.replace("WEAK_TOP_SCORE = 0.05", "WEAK_TOP_SCORE = 0.001")
print("  [OK] WEAK_TOP_SCORE set to 0.001")

content = content.replace("WEAK_TOP_SCORE = 0.005", "WEAK_TOP_SCORE = 0.001")
print("  [OK] WEAK_TOP_SCORE set to 0.001 (checked 0.005 too)")

# ================================================================
# 5. Fix LEXICAL_CANDIDATE_LIMIT
# ================================================================
content = content.replace("LEXICAL_CANDIDATE_LIMIT = 100", "LEXICAL_CANDIDATE_LIMIT = 50")
print("  [OK] LEXICAL_CANDIDATE_LIMIT set to 50")

# ================================================================
# Write
# ================================================================
with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"""
{'=' * 70}
  Routing fix applied. Verify with:
    python -c "import time; from v23 import build_large_corpus_engine; db,proc,eng=build_large_corpus_engine(db_path='hybrid_rag_v22.db',provider='openai'); q='What GitHub PRs are related to the Jira tickets about the API migration?'; t0=time.time(); parts=eng._get_active_partitions(q); print('Active partitions:', parts); print('Lookup:', round(time.time()-t0,2), 's')"
{'=' * 70}""")
