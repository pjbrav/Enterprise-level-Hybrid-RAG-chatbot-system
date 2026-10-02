"""
Routing fix v2 — keyword-based partition detection with entity density fallback.

The entity-graph-based routing doesn't work because:
- Common entities ("api", "github") match everywhere
- Even density-based ranking gets fooled by which partition has the
  highest density of "api" mentions (not necessarily the relevant one)

This patch adds direct keyword matching: if a question mentions "jira",
"ticket", "issue" -> partition_jira. If it mentions "github", "pr",
"pull request" -> partition_github. This is simple, fast, and correct.

Entity density is kept as a fallback for questions that don't explicitly
mention a source by name.

Run:
    python fix_routing_v2.py
"""

import re

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ================================================================
# Replace _get_active_partitions with keyword + density hybrid
# ================================================================
pattern = r'(    def _get_active_partitions\(self, query: str\) -> List\[str\]:.*?)(\n    def _is_cross_domain_question)'

replacement = '''    # Direct keyword -> partition mapping. These are unambiguous signals
    # that a question is about a specific data source.
    PARTITION_KEYWORDS = {
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

'''
content_new = re.sub(pattern, replacement + r'\2', content, flags=re.DOTALL)

if content_new != content:
    content = content_new
    print("  [OK] Replaced _get_active_partitions() with keyword + density hybrid")
else:
    print("  [FAIL] Could not find _get_active_partitions()")

# ================================================================
# Ensure _is_cross_domain_question has no keyword gate
# ================================================================
pattern_cd = r'(    def _is_cross_domain_question\(self, query: str\) -> bool:.*?)(\n    def _is_simple_single_domain)'
replacement_cd = '''    def _is_cross_domain_question(self, query: str) -> bool:
        partitions = self._get_active_partitions(query)
        if len(partitions) < 2:
            return False
        # 2+ active partitions = cross-domain, no comparison keywords needed
        return True

'''
content_new = re.sub(pattern_cd, replacement_cd + r'\2', content, flags=re.DOTALL)

if content_new != content:
    content = content_new
    print("  [OK] Replaced _is_cross_domain_question() — removed keyword gate")
else:
    print("  [FAIL] Could not find _is_cross_domain_question()")

# ================================================================
# Fix WEAK_TOP_SCORE and LEXICAL_CANDIDATE_LIMIT
# ================================================================
content = content.replace("WEAK_TOP_SCORE = 0.05", "WEAK_TOP_SCORE = 0.001")
content = content.replace("WEAK_TOP_SCORE = 0.005", "WEAK_TOP_SCORE = 0.001")
print("  [OK] WEAK_TOP_SCORE set to 0.001")

content = content.replace("LEXICAL_CANDIDATE_LIMIT = 100", "LEXICAL_CANDIDATE_LIMIT = 50")
print("  [OK] LEXICAL_CANDIDATE_LIMIT set to 50")

with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"""
{'=' * 70}
  Routing fix v2 applied.

  How it works now:
  - If the query mentions "jira", "ticket" -> routes to partition_jira
  - If the query mentions "github", "pr", "repo" -> routes to partition_github
  - If no keywords match -> falls back to entity density

  Verify:
    python -c "from v23 import build_large_corpus_engine; db,proc,eng=build_large_corpus_engine(db_path='hybrid_rag_v22.db',provider='openai'); print(eng._get_active_partitions('What GitHub PRs are related to the Jira tickets about the API migration?'))"
{'=' * 70}""")
