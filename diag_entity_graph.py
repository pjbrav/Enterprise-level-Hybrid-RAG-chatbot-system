"""
Diagnostic for chunk_entities table in hybrid_rag_v22.db
Run from the same folder as v23.py:
    python diag_entity_graph.py
"""

import sqlite3
from collections import Counter

DB = "hybrid_rag_v22.db"


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# ---------------------------------------------------------------- TEST 0
section("TEST 0 — Schema check")
for r in cur.execute("PRAGMA table_info(chunk_entities)").fetchall():
    print(f"  {r['name']:20s} {r['type']}")


# ---------------------------------------------------------------- TEST 1
section("TEST 1 — Original cross-partition entities (raw)")
rows = cur.execute("""
    SELECT entity, COUNT(DISTINCT partition_id) AS parts
    FROM chunk_entities
    GROUP BY entity
    HAVING parts > 1
    ORDER BY parts DESC
    LIMIT 20
""").fetchall()
print(f"  Rows returned: {len(rows)}")
for r in rows[:10]:
    print(f"    {r['entity'][:50]:50s}  partitions={r['parts']}")


# ---------------------------------------------------------------- TEST 2
section("TEST 2 — NORMALIZED cross-partition entities (lowercased + trimmed)")
rows = cur.execute("""
    SELECT lower(trim(entity)) AS norm, COUNT(DISTINCT partition_id) AS parts, COUNT(*) AS mentions
    FROM chunk_entities
    GROUP BY norm
    HAVING parts > 1
    ORDER BY parts DESC, mentions DESC
    LIMIT 30
""").fetchall()
print(f"  Rows returned: {len(rows)}")
for r in rows[:15]:
    print(f"    {r['norm'][:50]:50s}  partitions={r['parts']}  mentions={r['mentions']}")


# ---------------------------------------------------------------- TEST 3
section("TEST 3 — Known cross-domain terms")
terms = ['api', 'github', 'slack', 'jira', 'confluence', 'prod', 'staging',
         'aws', 'kubernetes', 'bug', 'deploy', 'migration', 'auth', 'docker',
         'frontend', 'backend', 'database', 'review', 'merge', 'release']
for t in terms:
    rows = cur.execute("""
        SELECT COUNT(DISTINCT partition_id) AS parts, COUNT(*) AS n
        FROM chunk_entities WHERE lower(entity) = ?
    """, (t,)).fetchall()
    parts, n = rows[0]['parts'], rows[0]['n']
    marker = "  <-- CROSS-PARTITION" if parts > 1 else ""
    print(f"    {t:15s}  partitions={parts}  mentions={n}{marker}")


# ---------------------------------------------------------------- TEST 4
section("TEST 4 — Per-partition entity density")
rows = cur.execute("""
    SELECT c.partition_id,
           COUNT(DISTINCT c.chunk_id)               AS chunks,
           COUNT(DISTINCT ce.entity)               AS distinct_entities,
           1.0 * COUNT(ce.entity) / COUNT(DISTINCT c.chunk_id)  AS per_chunk
    FROM chunks c
    LEFT JOIN chunk_entities ce ON c.chunk_id = ce.chunk_id
    GROUP BY c.partition_id
    ORDER BY chunks DESC
""").fetchall()
print(f"  {'Partition':<30s} {'Chunks':>8s} {'Entities':>10s} {'PerChunk':>9s}")
print(f"  {'-' * 30} {'-' * 8} {'-' * 10} {'-' * 9}")
for r in rows:
    print(f"  {str(r['partition_id'])[:30]:<30s} {r['chunks']:>8d} {r['distinct_entities']:>10d} {r['per_chunk']:>9.2f}")


# ---------------------------------------------------------------- TEST 5
section("TEST 5 — Entity extraction coverage (chunks with zero entities)")
row = cur.execute("""
    SELECT COUNT(*) AS total,
           SUM(CASE WHEN ec.cnt = 0 OR ec.cnt IS NULL THEN 1 ELSE 0 END) AS zero_ent,
           AVG(COALESCE(ec.cnt, 0)) AS avg_ent
    FROM chunks c
    LEFT JOIN (SELECT chunk_id, COUNT(*) AS cnt FROM chunk_entities GROUP BY chunk_id) ec
      ON c.chunk_id = ec.chunk_id
""").fetchone()
print(f"  Total chunks:           {row['total']:,}")
print(f"  Chunks with 0 entities: {row['zero_ent']:,}  ({row['zero_ent'] * 100.0 / max(1, row['total']):.1f}%)")
print(f"  Avg entities/chunk:     {row['avg_ent']:.2f}")


# ---------------------------------------------------------------- TEST 6
section("TEST 6 — Top entities per partition (Jira & GitHub focus)")
for pid_filter in ['jira', 'github']:
    rows = cur.execute("""
        SELECT entity, COUNT(*) AS n
        FROM chunk_entities
        WHERE lower(partition_id) LIKE ?
        GROUP BY entity
        ORDER BY n DESC
        LIMIT 10
    """, (f"%{pid_filter}%",)).fetchall()
    print(f"\n  Top entities in '{pid_filter}' partition:")
    if not rows:
        print("    (none found — check partition name in Test 4)")
    for r in rows:
        print(f"    {r['entity'][:50]:50s}  n={r['n']}")


# ---------------------------------------------------------------- TEST 7
section("TEST 7 — Partition name inventory (so you know the exact IDs)")
rows = cur.execute("SELECT DISTINCT partition_id FROM chunks ORDER BY partition_id").fetchall()
for r in rows:
    print(f"    {r['partition_id']}")


# ---------------------------------------------------------------- TEST 8
section("TEST 8 — Cross-partition entity overlap matrix (Jira x GitHub)")
# Get exact partition IDs first
parts = [r['partition_id'] for r in cur.execute("SELECT DISTINCT partition_id FROM chunks ORDER BY partition_id").fetchall()]
jira_ids = [p for p in parts if 'jira' in p.lower()]
gh_ids = [p for p in parts if 'github' in p.lower() or 'git' in p.lower()]
print(f"  Jira partition(s):    {jira_ids}")
print(f"  GitHub partition(s):  {gh_ids}")

if jira_ids and gh_ids:
    jira_ent = {r['entity'] for r in cur.execute(
        "SELECT DISTINCT entity FROM chunk_entities WHERE partition_id = ?", (jira_ids[0],)).fetchall()}
    gh_ent = {r['entity'] for r in cur.execute(
        "SELECT DISTINCT entity FROM chunk_entities WHERE partition_id = ?", (gh_ids[0],)).fetchall()}

    # Raw overlap
    raw_overlap = jira_ent & gh_ent
    # Normalized overlap
    jira_norm = {e.lower().strip() for e in jira_ent}
    gh_norm = {e.lower().strip() for e in gh_ent}
    norm_overlap = jira_norm & gh_norm

    print(f"\n  Raw entities in Jira:     {len(jira_ent):,}")
    print(f"  Raw entities in GitHub:   {len(gh_ent):,}")
    print(f"  Raw overlap (exact match): {len(raw_overlap)}")
    print(f"  Normalized overlap:        {len(norm_overlap)}")
    if norm_overlap:
        print(f"\n  Top 20 normalized shared entities:")
        for e in sorted(norm_overlap)[:20]:
            print(f"    {e}")


# ---------------------------------------------------------------- TEST 9
section("TEST 9 — Case-variant check (same concept, different casing)")
# Check if entities that appear in 1 partition have case variants in another
rows = cur.execute("""
    SELECT lower(trim(entity)) AS norm,
           GROUP_CONCAT(DISTINCT entity) AS variants,
           COUNT(DISTINCT partition_id) AS parts
    FROM chunk_entities
    GROUP BY norm
    HAVING parts > 1
    ORDER BY parts DESC
    LIMIT 20
""").fetchall()
print(f"  Concepts with case/format variants across partitions: {len(rows)}")
for r in rows[:10]:
    variants = r['variants'][:80] if r['variants'] else ''
    print(f"    norm='{r['norm']}'  parts={r['parts']}  variants={variants}")

conn.close()

print(f"\n{'=' * 70}")
print("  Diagnostic complete.")
print(f"{'=' * 70}")
