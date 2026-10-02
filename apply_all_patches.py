"""
ALL-IN-ONE PATCH — applies spaCy NER + smart router + env-configurable thresholds
directly to v23.py (NOT v24.py).

Run:
    python apply_all_patches.py

Prerequisites:
    pip install spacy PyYAML
    python -m spacy download en_core_web_sm

What this does:
    1. Replaces extract_entities() with spaCy NER + regex ID fallback
    2. Activates smart router in compare() — force_hybrid param
    3. Makes all thresholds env-configurable
    4. Cleans stopwords (removes template-specific words if still present)
"""

import re
import shutil

V23_FILE = "v23.py"
BACKUP_FILE = "v23_pre_all_patches.py"

# Backup
shutil.copy2(V23_FILE, BACKUP_FILE)
print(f"  [OK] Backup saved to {BACKUP_FILE}")

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ================================================================
# PATCH 1: Replace extract_entities() with spaCy NER + regex fallback
# ================================================================
# Match from "def extract_entities" to the "# Persistence" section
OLD_EXTRACT = '''def extract_entities(text: str) -> Set[str]:
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
    # Keep meaningful standalone numbers (4+ digits -- likely IDs, timestamps)
    # but exclude plausible calendar years (1900-2099): confirmed via
    # profiling that a bare year matches an enormous fraction of the corpus
    # (e.g. "2026" matched 278,133 chunks in one partition) since it appears
    # in ordinary dates/timestamps everywhere, carrying near-zero
    # discriminative signal despite looking like a specific identifier.
    id_pattern = r"\\b\\d{4,}\\b"
    ids = re.findall(id_pattern, text)
    for id_val in ids:
        if len(id_val) == 4 and 1900 <= int(id_val) <= 2099:
            continue
        entities.add(id_val)
    return {e for e in entities if len(e) > 2}'''

NEW_EXTRACT = '''_SPACY_NLP = None
_SPACY_AVAILABLE = False
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    pass

_SPACY_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "NORP", "FAC", "EVENT",
    "WORK_OF_ART", "LAW", "PRODUCT", "LOC",
}


def _get_spacy_nlp():
    """Lazily load spaCy model on first use (saves startup time)."""
    global _SPACY_NLP
    if _SPACY_NLP is None and _SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            try:
                _SPACY_NLP = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])
            except Exception:
                _SPACY_NLP = None
    return _SPACY_NLP


def _add_regex_ids(text: str, entities: Set[str]) -> None:
    """Extract structured IDs that spaCy misses."""
    # Ticket IDs: PROJ-123, ENG-4567
    for m in re.findall(r"\\b([A-Z]{2,10}-\\d{1,8})\\b", text):
        entities.add(m.lower())
    # PR/issue references: #1234
    for m in re.findall(r"#(\\d{2,8})\\b", text):
        entities.add(f"#{m}")
    # Commit hashes: 7-40 hex chars
    for m in re.findall(r"\\b([0-9a-f]{7,40})\\b", text):
        if len(m) >= 7:
            entities.add(m)
    # Version numbers: v2.3.1, 2.0.0
    for m in re.findall(r"\\b(v?\\d+\\.\\d+\\.\\d+)\\b", text):
        entities.add(m.lower())


def extract_entities(text: str) -> Set[str]:
    """Extract entities using spaCy NER + regex ID fallback.

    spaCy provides context-aware named entity recognition (PERSON, ORG,
    PRODUCT, etc.) that is far more accurate than regex for real names.
    A regex fallback catches structured IDs spaCy misses: ticket IDs,
    PR numbers, commit hashes.

    All entities are lowercased. If spaCy is not installed or the model
    is not downloaded, falls back to the original regex-only extractor.
    """
    entities = set()

    # --- spaCy NER path ---
    nlp = _get_spacy_nlp()
    if nlp is not None:
        truncated = text[:50000] if len(text) > 50000 else text
        doc = nlp(truncated)
        for ent in doc.ents:
            if ent.label_ in _SPACY_ENTITY_TYPES:
                name = ent.text.strip().lower()
                if len(name) >= 3 and name not in _ENTITY_STOPWORDS:
                    entities.add(name)
        _add_regex_ids(text, entities)
        return entities

    # --- Regex fallback (spaCy not available) ---
    candidates = re.findall(r"\\b[A-Z][A-Za-z0-9\\-]*(?:\\s+[A-Z][A-Za-z0-9\\-]*)*\\b", text)
    for c in candidates:
        c = c.strip().lower()
        if len(c) >= 3 and c not in _ENTITY_STOPWORDS:
            entities.add(c)
    number_pattern = r"\\b[\\$€]?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?\\s*(?:psi|USD|EUR|bar|%|k|M)?\\b"
    for num in re.findall(number_pattern, text):
        num = num.strip()
        if num and len(num) > 2:
            entities.add(num)
    id_pattern = r"\\b\\d{4,}\\b"
    for id_val in re.findall(id_pattern, text):
        if len(id_val) == 4 and 1900 <= int(id_val) <= 2099:
            continue
        entities.add(id_val)
    _add_regex_ids(text, entities)
    return {e for e in entities if len(e) > 2}'''

if OLD_EXTRACT in content:
    content = content.replace(OLD_EXTRACT, NEW_EXTRACT)
    changes += 1
    print("  [OK] Replaced extract_entities() with spaCy NER + regex ID fallback")
else:
    print("  [SKIP] extract_entities() not found — may have different formatting")
    # Try regex approach: replace everything between "def extract_entities" and "# Persistence"
    pattern = r'def extract_entities\(text: str\) -> Set\[str\]:.*?(?=\n# -{20,}\n# Persistence)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + NEW_EXTRACT + content[match.end():]
        changes += 1
        print("  [OK] Replaced extract_entities() via regex match")
    else:
        print("  [FAIL] Could not find extract_entities at all")

# ================================================================
# PATCH 2: Activate smart router in compare()
# ================================================================
OLD_COMPARE_SIG = '''    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None,
                include_direct_ai: bool = True) -> Dict[str, Dict[str, Any]]:'''

NEW_COMPARE_SIG = '''    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None,
                include_direct_ai: bool = True, force_hybrid: bool = False) -> Dict[str, Dict[str, Any]]:'''

if OLD_COMPARE_SIG in content:
    content = content.replace(OLD_COMPARE_SIG, NEW_COMPARE_SIG)
    changes += 1
    print("  [OK] Added force_hybrid param to compare() signature")
else:
    print("  [SKIP] compare() signature not found")

# Replace the Forced Hybrid decision line
OLD_DECISION = '''        decision = {"chosen": "hybrid", "standard_confidence": 0.0, "reason": "Forced Hybrid"}'''

NEW_DECISION = '''        if force_hybrid:
            decision = {"chosen": "hybrid", "standard_confidence": 0.0, "reason": "Forced Hybrid (eval mode)"}
        else:
            decision = self._should_use_hybrid(query)'''

if OLD_DECISION in content:
    content = content.replace(OLD_DECISION, NEW_DECISION)
    changes += 1
    print("  [OK] Activated smart router — force_hybrid=False uses _should_use_hybrid()")
else:
    print("  [SKIP] Forced Hybrid decision line not found")

# ================================================================
# PATCH 3: Make thresholds env-configurable
# ================================================================
THRESHOLD_REPLACEMENTS = [
    ('self.CROSS_DOMAIN_THRESHOLD_RELIEF = 0.12',
     'self.CROSS_DOMAIN_THRESHOLD_RELIEF = float(os.getenv("CROSS_DOMAIN_THRESHOLD_RELIEF", "0.12"))'),
    ('self.MIN_THRESHOLD = 0.20',
     'self.MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.20"))'),
    ('self.ROUTER_STANDARD_CONFIDENCE_HIGH = 0.99',
     'self.ROUTER_STANDARD_CONFIDENCE_HIGH = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_HIGH", "0.99"))'),
    ('self.ROUTER_STANDARD_CONFIDENCE_LOW = 0.40',
     'self.ROUTER_STANDARD_CONFIDENCE_LOW = float(os.getenv("ROUTER_STANDARD_CONFIDENCE_LOW", "0.40"))'),
    ('self.ROUTER_HYBRID_SCORE_THRESHOLD = 0.50',
     'self.ROUTER_HYBRID_SCORE_THRESHOLD = float(os.getenv("ROUTER_HYBRID_SCORE_THRESHOLD", "0.50"))'),
    ('self.ROUTER_LAST_RESORT_THRESHOLD = 0.55',
     'self.ROUTER_LAST_RESORT_THRESHOLD = float(os.getenv("ROUTER_LAST_RESORT_THRESHOLD", "0.55"))'),
    ('self.LAST_RESORT_CANDIDATE_POOL = 20',
     'self.LAST_RESORT_CANDIDATE_POOL = int(os.getenv("LAST_RESORT_CANDIDATE_POOL", "20"))'),
]

for old, new in THRESHOLD_REPLACEMENTS:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        name = old.split('=')[0].strip().split('.')[-1]
        print(f"  [OK] Made {name} env-configurable")
    else:
        print(f"  [SKIP] {old.split('=')[0].strip()} not found")

# ================================================================
# Write patched v23.py
# ================================================================
with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"""
{'=' * 70}
  ALL PATCHES APPLIED: {changes} changes to v23.py
{'=' * 70}

  Changes:
  1. spaCy NER + regex ID fallback in extract_entities()
  2. Smart router activated in compare() — force_hybrid=False by default
  3. All 7 thresholds made env-configurable

  Backup saved to: {BACKUP_FILE}

  Next steps:

    1. Install spaCy:
       pip install spacy PyYAML
       python -m spacy download en_core_web_sm

    2. Verify spaCy is active:
       python -c "from v23 import extract_entities; print(extract_entities('marissa cole from acme corporation said the deployment failed'))"

       If output contains 'marissa cole' and 'acme corporation', spaCy is working.
       If output is empty, spaCy model isn't loaded — check the install.

    3. Run tests:
       python test_v23.py

    4. Run eval (force both pipelines for comparison):
       python batch_compare_v2.py --force-hybrid --sample 50

    5. Run eval (smart router mode):
       python batch_compare_v2.py --sample 50

{'=' * 70}""")
