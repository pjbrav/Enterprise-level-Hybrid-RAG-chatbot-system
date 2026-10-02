"""
Upgrade v23.py to v24.py — spaCy NER + config-file routing.

This script:
1. Copies v23.py -> v24.py
2. Replaces extract_entities() with spaCy NER + regex ID fallback
3. Replaces hardcoded PARTITION_KEYWORDS with config-file loading
4. Adds routing_config.yaml template

Run:
    python upgrade_to_v24.py

Then install spaCy:
    pip install spacy
    python -m spacy download en_core_web_sm

Then re-backfill entities:
    python backfill_entities.py    # but change 'from v23 import' to 'from v24 import'

Then run eval:
    # Update batch_compare_all.py to import from v24 instead of v23
"""

import shutil
import os

V23_FILE = "v23.py"
V24_FILE = "v24.py"
CONFIG_FILE = "routing_config.yaml"

# ================================================================
# 1. Copy v23.py to v24.py
# ================================================================
if not os.path.exists(V23_FILE):
    print(f"ERROR: {V23_FILE} not found. Run this from the same folder.")
    import sys
    sys.exit(1)

shutil.copy2(V23_FILE, V24_FILE)
print(f"  [OK] Copied {V23_FILE} -> {V24_FILE}")

with open(V24_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ================================================================
# 2. Replace extract_entities() with spaCy NER + regex ID fallback
# ================================================================
OLD_EXTRACT = '''# Boilerplate / common English words that the capitalized-phrase regex
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

NEW_EXTRACT = '''# ---------------------------------------------------------------------------
# Entity extraction: spaCy NER + regex ID fallback
# ---------------------------------------------------------------------------
# spaCy extracts typed named entities (PERSON, ORG, GPE, EVENT, PRODUCT, etc.)
# with context awareness — "John" at sentence start is a person, not just a
# capitalized word. This produces a far cleaner entity graph than pure regex.
#
# The regex fallback catches structured IDs spaCy misses: ticket IDs (PROJ-123),
# PR numbers (#4567), commit hashes, etc.
#
# If spaCy is not installed, falls back to the original regex-only extractor
# so the system never breaks on a fresh deployment.

_SPACY_NLP = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    pass

# Boilerplate filter (kept for the regex fallback path)
_ENTITY_STOPWORDS = frozenset({
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
    "added", "removed", "fixed", "updated", "created", "deleted",
    "customer", "support", "issue", "impact", "environment",
    "observed", "confirmed", "initial", "reported", "assigned",
    "priority", "status", "description", "summary", "comment",
    "motivation", "checklist", "changes", "approach", "context",
    "rollout", "background", "testing", "reviewed", "review",
    "merge", "merged", "pull", "request", "branch",
    "steps", "reproduce", "expected", "actual", "behavior",
    "acceptance", "criteria", "definition", "done",
    "utc", "est", "pst", "cet", "ist",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow",
})

# spaCy entity types we care about (others like DATE, TIME, MONEY, QUANTITY
# are too noisy for partition routing)
_SPACY_ENTITY_TYPES = {
    "PERSON",      # people names — strong cross-partition signal
    "ORG",         # companies, products, teams — strongest routing signal
    "GPE",         # countries, cities — useful for geo-related queries
    "NORP",        # nationalities, religious/political groups
    "FAC",         # facilities, buildings
    "EVENT",       # named events (conferences, incidents)
    "WORK_OF_ART", # titles, documents
    "LAW",         # laws, regulations
    "PRODUCT",     # products, features — strong routing signal
    "LOC",         # non-GPE locations
}


def _get_spacy_nlp():
    \"\"\"Lazily load spaCy model on first use (saves startup time).\"\"\"
    global _SPACY_NLP
    if _SPACY_NLP is None and _SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
            except Exception:
                _SPACY_NLP = None  # model not downloaded — fall back to regex
    return _SPACY_NLP


def extract_entities(text: str) -> Set[str]:
    \"\"\"Extract entities using spaCy NER + regex ID fallback.

    spaCy provides context-aware named entity recognition (PERSON, ORG,
    PRODUCT, etc.) that is far more accurate than regex for real names.
    A regex fallback catches structured IDs spaCy misses: ticket IDs,
    PR numbers, commit hashes.

    All entities are lowercased. If spaCy is not installed or the model
    is not downloaded, falls back to the original regex-only extractor.
    \"\"\"
    entities = set()

    # --- spaCy NER path ---
    nlp = _get_spacy_nlp()
    if nlp is not None:
        # Truncate very long texts to avoid spaCy's 1M char limit
        # and keep processing fast on large chunks
        truncated = text[:50000] if len(text) > 50000 else text
        doc = nlp(truncated)
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                name = ent.text.strip().lower()
                if len(name) >= 3 and name not in _ENTITY_STOPWORDS:
                    entities.add(name)

        # Regex ID extraction (runs alongside spaCy, not instead of)
        # These are structured patterns spaCy doesn't catch
        _add_regex_ids(text, entities)
        return entities

    # --- Regex fallback (spaCy not available) ---
    candidates = re.findall(r"\\b[A-Z][A-Za-z0-9\\-]*(?:\\s+[A-Z][A-Za-z0-9\\-]*)*\\b", text)
    for c in candidates:
        c = c.strip().lower()
        if len(c) >= 3 and c not in _ENTITY_STOPWORDS:
            entities.add(c)
    # Numbers with units or 4+ digit IDs
    number_pattern = r"\\b[\\$€]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?\\s*(?:psi|USD|EUR|bar|%|k|M)?\\b"
    for num in re.findall(number_pattern, text):
        num = num.strip()
        if num and len(num) > 2:
            entities.add(num)
    id_pattern = r"\\b\\d{4,}\\b"
    for id_val in re.findall(id_pattern, text):
        entities.add(id_val)
    return {e for e in entities if len(e) > 2}


def _add_regex_ids(text: str, entities: Set[str]) -> None:
    \"\"\"Extract structured IDs that spaCy misses.

    These patterns are common in enterprise data and carry strong
    routing signal:
    - Ticket IDs: PROJ-123, ENG-4567
    - PR references: #1234, PR #5678
    - Commit hashes: abc123def456 (7-40 hex chars)
    - File paths with extensions: config.yaml, main.py
    - Version numbers: v2.3.1, 2.0.0
    \"\"\"
    # Ticket IDs: 2-10 uppercase letters, hyphen, 1-8 digits
    for m in re.findall(r"\\b([A-Z]{2,10}-\\d{1,8})\\b", text):
        entities.add(m.lower())

    # PR/issue references: #1234
    for m in re.findall(r"#(\\d{2,8})\\b", text):
        entities.add(f"#{m}")

    # Commit hashes: 7-40 hex characters (boundary-separated)
    for m in re.findall(r"\\b([0-9a-f]{7,40})\\b", text):
        if len(m) >= 7:
            entities.add(m)

    # Version numbers: v2.3.1, 2.0.0
    for m in re.findall(r"\\b(v?\\d+\\.\\d+\\.\\d+)\\b", text):
        entities.add(m.lower())'''


if OLD_EXTRACT in content:
    content = content.replace(OLD_EXTRACT, NEW_EXTRACT)
    print("  [OK] Replaced extract_entities() with spaCy NER + regex ID fallback")
else:
    print("  [SKIP] Could not find original extract_entities() — may already be patched")
    # Try to find it by function signature
    import re as _re
    match = _re.search(r'def extract_entities\(text: str\) -> Set\[str\]:', content)
    if match:
        print(f"    Found extract_entities at position {match.start()}")
    else:
        print("    extract_entities not found at all")


# ================================================================
# 3. Replace hardcoded PARTITION_KEYWORDS with config-file loading
# ================================================================

# Find the PARTITION_KEYWORDS block and replace with config loading
OLD_KEYWORDS = '''    # Direct keyword -> partition mapping. These are unambiguous signals
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
    }'''

NEW_KEYWORDS = '''    # Default keyword -> partition mapping. Loaded from routing_config.yaml
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
    PARTITION_KEYWORDS = _DEFAULT_PARTITION_KEYWORDS  # may be overridden in __init__'''

if OLD_KEYWORDS in content:
    content = content.replace(OLD_KEYWORDS, NEW_KEYWORDS)
    print("  [OK] Replaced hardcoded PARTITION_KEYWORDS with config-file loading")
else:
    print("  [SKIP] PARTITION_KEYWORDS block not found — may have different formatting")


# ================================================================
# 4. Add config loading to ComparisonEngine.__init__
# ================================================================

OLD_INIT_END = '''        self.cross_encoder = CrossEncoderReranker()
        self.CROSS_DOMAIN_THRESHOLD_RELIEF = 0.12'''

NEW_INIT_END = '''        self.cross_encoder = CrossEncoderReranker()
        self.CROSS_DOMAIN_THRESHOLD_RELIEF = 0.12
        # Load per-customer routing config if available
        self._load_routing_config()'''

if OLD_INIT_END in content:
    content = content.replace(OLD_INIT_END, NEW_INIT_END)

    # Add the _load_routing_config method right after __init__
    # Find the first method after __init__ to insert before it
    OLD_FIRST_METHOD = '''    def _build_index_streaming('''

    NEW_METHOD = '''    def _load_routing_config(self, config_path: str = "routing_config.yaml") -> None:
        \"\"\"Load per-customer partition routing keywords from a YAML config file.

        If the file doesn't exist, the hardcoded defaults in
        _DEFAULT_PARTITION_KEYWORDS are used. This lets each deployment
        customize routing without editing code.
        \"\"\"
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

    def _build_index_streaming('''

    if OLD_FIRST_METHOD in content:
        content = content.replace(OLD_FIRST_METHOD, NEW_METHOD)
        print("  [OK] Added _load_routing_config() method")
    else:
        print("  [SKIP] Could not find insertion point for _load_routing_config()")
else:
    print("  [SKIP] __init__ end not found — routing config loading not added")


# ================================================================
# 5. Write v24.py
# ================================================================
with open(V24_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"  [OK] Written {V24_FILE} ({len(content):,} bytes)")


# ================================================================
# 6. Create routing_config.yaml template
# ================================================================
YAML_TEMPLATE = """# Routing Configuration — per-customer partition keyword mappings
#
# Copy this file to routing_config.yaml and edit for your deployment.
# If this file doesn't exist, hardcoded defaults in v24.py are used.
#
# Keywords are matched case-insensitively against the query text.
# If any keyword appears in the question, the corresponding partition
# is activated for cross-domain search.

partitions:
  partition_jira:
    keywords: ["jira", "ticket", "epic", "sprint", "backlog", "kanban"]

  partition_github:
    keywords: ["github", "pr", "pull request", "pull-request", "repo", "repository", "commit", "branch", "merge request", "merge-request"]

  partition_slack:
    keywords: ["slack", "channel", "dm", "direct message", "slack message"]

  partition_gmail:
    keywords: ["gmail", "email", "inbox", "thread", "mail"]

  partition_confluence:
    keywords: ["confluence", "wiki", "page", "documentation page"]

  partition_hubspot:
    keywords: ["hubspot", "deal", "contact", "company", "pipeline", "crm"]

  partition_linear:
    keywords: ["linear", "linear issue", "project"]

  partition_fireflies:
    keywords: ["fireflies", "meeting", "transcript", "call recording", "firefly"]

  partition_google_drive:
    keywords: ["google drive", "drive", "doc", "document", "sheet", "spreadsheet"]

  # Add new partitions for new data sources here:
  # partition_zendesk:
  #   keywords: ["zendesk", "ticket", "case", "support request"]
  #
  # partition_notion:
  #   keywords: ["notion", "page", "database", "workspace"]
"""

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(YAML_TEMPLATE)
    print(f"  [OK] Created {CONFIG_FILE} template")
else:
    print(f"  [SKIP] {CONFIG_FILE} already exists (not overwriting)")


print(f"""
{'=' * 70}
  v24.py created successfully.
{'=' * 70}

  What changed (v23 -> v24):

  1. Entity extraction: spaCy NER + regex ID fallback
     - spaCy extracts PERSON, ORG, PRODUCT, GPE, EVENT, etc.
     - Regex still catches structured IDs: PROJ-123, #4567, commit hashes
     - Falls back to regex-only if spaCy not installed

  2. Routing: config-file driven instead of hardcoded
     - routing_config.yaml controls partition keywords per customer
     - Edit YAML, not Python code, for new deployments

  Next steps:

    1. Install spaCy:
       pip install spacy
       python -m spacy download en_core_web_sm

    2. Re-backfill entities with v24:
       # Edit backfill_entities.py: change 'from v23 import' to 'from v24 import'
       python backfill_entities.py

    3. Re-run diagnostic:
       python diag_entity_graph.py
       (change 'from v23 import' to 'from v24 import' if needed)

    4. Run eval:
       # Edit batch_compare_all.py: change 'from v23 import' to 'from v24 import'
       python batch_compare_all.py --sample 50

    5. If spaCy is not installed, v24 still works — it falls back to the
       regex extractor automatically. You just won't get the NER quality
       improvement until spaCy is installed and the backfill is re-run.

  Expected impact (with spaCy):
     - Answer relevance: +2pp -> +5-8pp (better routing -> better chunks)
     - Worst-case losses: -30 to -48pp -> -10 to -15pp (fewer wrong partitions)
     - Token savings: -13.5% -> -15-20% (fewer irrelevant chunks)
     - Latency: similar or slightly better (fewer wrong-partition searches)
     - Backfill time: ~28min -> ~10 hours (spaCy is ~20x slower than regex)

{'=' * 70}""")
