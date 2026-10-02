"""
Patches v23.py's extract_entities() function to:
1. Lowercase all entities (fixes case-sensitivity bug)
2. Filter template boilerplate and common English words
3. Be stricter with number extraction (no bare decimals)

Also patches entity_search() and get_partition_ids_for_entities()
in DatabaseManager to do case-insensitive matching.

Run from the same folder as v23.py:
    python patch_extract_entities.py

Then re-run the backfill:
    python backfill_entities.py
"""

import re

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# =======================================================
# 1. Replace extract_entities() with a cleaned-up version
# =======================================================

OLD_EXTRACT = '''def extract_entities(text: str) -> Set[str]:
    """Generic entity extraction: capitalized phrases, acronyms, and numbers."""
    candidates = re.findall(r"\\b[A-Z][A-Za-z0-9\\-]*(?:\\s+[A-Z][A-Za-z0-9\\-]*)*\\b", text)
    entities = set()
    for c in candidates:
        c = c.strip()
        if len(c) >= 3:
            entities.add(c)
    number_pattern = r"\\b[\\$€]?\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?\\s*(?:psi|USD|EUR|bar|%|k|M)?\\b"
    numbers = re.findall(number_pattern, text)
    for num in numbers:
        if num.strip():
            entities.add(num.strip())
    return {e for e in entities if len(e) > 2}'''

NEW_EXTRACT = '''# Boilerplate / common English words that the capitalized-phrase regex
# picks up from sentence starts, Jira/GitHub templates, etc. These carry
# no entity-graph signal and just pollute cross-partition matching.
_ENTITY_STOPWORDS = frozenset({
    # Common sentence-starters
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "is", "are", "was", "were", "be", "been", "being", "been",
    "has", "have", "had", "do", "does", "did", "done",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "for", "from", "with", "about", "into", "onto", "over", "under",
    "and", "or", "but", "if", "then", "else", "so", "as", "at", "by",
    "in", "on", "to", "of", "up", "out", "off", "down", "away",
    "yes", "no", "not", "nor", "yet", "ok", "okay",
    "you", "your", "yours", "we", "our", "ours", "us",
    "i", "me", "my", "mine", "they", "them", "their", "theirs",
    "he", "she", "his", "her", "hers", "him",
    "please", "thanks", "thank", "hi", "hey", "hello",
    # Template / boilerplate from Jira, GitHub, Confluence
    "added", "removed", "fixed", "updated", "created", "deleted",
    "customer", "support", "issue", "impact", "environment",
    "observed", "confirmed", "initial", "reported", "assigned",
    "priority", "status", "description", "summary", "comment",
    "motivation", "checklist", "changes", "approach", "context",
    "rollout", "background", "testing", "reviewed", "review",
    "merge", "merged", "pull", "request", "branch",
    "steps", "reproduce", "expected", "actual", "behavior",
    "acceptance", "criteria", "definition", "done",
    # Common words that happen to be capitalized
    "utc", "est", "pst", "cet", "ist",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow",
})


def extract_entities(text: str) -> Set[str]:
    """Generic entity extraction: capitalized phrases, acronyms, and numbers.

    All entities are lowercased so that "GitHub", "Github", and "GITHUB"
    are treated as the same entity. Common boilerplate words (template
    field labels, sentence-starters) are filtered out.
    """
    candidates = re.findall(r"\\b[A-Z][A-Za-z0-9\\-]*(?:\\s+[A-Z][A-Za-z0-9\\-]*)*\\b", text)
    entities = set()
    for c in candidates:
        c = c.strip().lower()
        if len(c) >= 3 and c not in _ENTITY_STOPWORDS:
            # Also filter single-word entities that are all-consonant or
            # all-vowel (unlikely to be real entities)
            entities.add(c)
    # Numbers: keep only meaningful numbers (IDs, percentages, dollar amounts)
    # Skip bare decimals like "0.0", "0.00", "0.0001" which are noise from
    # metrics/telemetry text. Keep numbers with units or >= 3 digits.
    number_pattern = r"\\b[\\$€]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?\\s*(?:psi|USD|EUR|bar|%|k|M)?\\b"
    numbers = re.findall(number_pattern, text)
    for num in numbers:
        num = num.strip()
        if num and len(num) > 2:
            entities.add(num)
    # Keep meaningful standalone numbers (4+ digits — likely IDs, timestamps)
    id_pattern = r"\\b\\d{4,}\\b"
    ids = re.findall(id_pattern, text)
    for id_val in ids:
        entities.add(id_val)
    return {e for e in entities if len(e) > 2}'''


if OLD_EXTRACT not in content:
    # Try to locate the function by its signature
    import sys
    print("ERROR: Could not find the original extract_entities() in v23.py.")
    print("The function may have already been patched, or the formatting differs.")
    print("\nSearching for 'def extract_entities' ...")
    for i, line in enumerate(content.splitlines(), 1):
        if "def extract_entities" in line:
            print(f"  Found at line {i}: {line.strip()}")
    sys.exit(1)

content = content.replace(OLD_EXTRACT, NEW_EXTRACT)

# =======================================================
# 2. Patch entity_search() for case-insensitive matching
# =======================================================

OLD_ENTITY_SEARCH = '''        sql = f"SELECT chunk_id, COUNT(DISTINCT entity) AS matches FROM chunk_entities WHERE entity IN ({placeholders})"
        params: List[Any] = list(entities)'''

NEW_ENTITY_SEARCH = '''        entities_lower = [e.lower() for e in entities]
        sql = f"SELECT chunk_id, COUNT(DISTINCT lower(entity)) AS matches FROM chunk_entities WHERE lower(entity) IN ({placeholders})"
        params: List[Any] = list(entities_lower)'''

if OLD_ENTITY_SEARCH in content:
    content = content.replace(OLD_ENTITY_SEARCH, NEW_ENTITY_SEARCH)
    print("  [OK] Patched entity_search() for case-insensitive matching")
else:
    print("  [SKIP] entity_search() already patched or signature changed")

# =======================================================
# 3. Patch get_partition_ids_for_entities() for case-insensitive matching
# =======================================================

OLD_GET_PARTITIONS = '''        sql = (
            f"SELECT partition_id, COUNT(*) AS hits FROM chunk_entities "
            f"WHERE entity IN ({placeholders}) GROUP BY partition_id ORDER BY hits DESC"
        )
        rows = self._query(sql, entities, fetch=True)'''

NEW_GET_PARTITIONS = '''        entities_lower = [e.lower() for e in entities]
        sql = (
            f"SELECT partition_id, COUNT(*) AS hits FROM chunk_entities "
            f"WHERE lower(entity) IN ({placeholders}) GROUP BY partition_id ORDER BY hits DESC"
        )
        rows = self._query(sql, entities_lower, fetch=True)'''

if OLD_GET_PARTITIONS in content:
    content = content.replace(OLD_GET_PARTITIONS, NEW_GET_PARTITIONS)
    print("  [OK] Patched get_partition_ids_for_entities() for case-insensitive matching")
else:
    print("  [SKIP] get_partition_ids_for_entities() already patched or signature changed")

# =======================================================
# Write the patched file
# =======================================================

# Backup
with open(V23_FILE + ".bak", "w", encoding="utf-8") as f:
    f.write(open(V23_FILE, "r", encoding="utf-8").read())
print(f"  [OK] Backup saved to {V23_FILE}.bak")

with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)
print(f"  [OK] Patched {V23_FILE}")

print(f"""
{'=' * 70}
  Patch complete. Changes made:
  1. extract_entities() now lowercases all entities and filters
     boilerplate/template words
  2. entity_search() now matches case-insensitively
  3. get_partition_ids_for_entities() now matches case-insensitively

  Next steps:
    1. Re-run the backfill to rebuild the entity table:
         python backfill_entities.py

    2. Re-run the diagnostic to verify:
         python diag_entity_graph.py

    3. Re-run your eval:
         python batch_compare.py
{'=' * 70}""")
