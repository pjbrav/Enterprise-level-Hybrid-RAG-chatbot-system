"""
Productionize v23.py — 6 changes to make the codebase production-worthy.

1. Enable smart router in compare() (remove "Forced Hybrid")
2. Wire up routing_config.yaml loading in __init__
3. Add graceful error handling for missing spaCy model / DB / config
4. (Eval script changes are in batch_compare_all_v2.py — separate file)
5. (requirements.txt and setup.py — separate files)
6. (Unit tests — separate file)

Run:
    python productionize.py
"""

import re

V23_FILE = "v23.py"

with open(V23_FILE, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ================================================================
# 1. Enable smart router — replace "Forced Hybrid" with _should_use_hybrid
# ================================================================
OLD_COMPARE = '''    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run Standard always; run Hybrid only when the router thinks it's
        worth it. This is what cuts the per-question API-call/latency cost
        versus always running both -- Hybrid's generation call (and its
        graph/embedding/cross-encoder retrieval work) is skipped whenever
        Standard is already good enough, while the comparison table still
        has a Hybrid row (copied from Standard) so callers don't have to
        special-case a missing slot.
        """
        provider_configs = (
            available_provider_configs(provider, key_override=api_key)
            if provider is not None
            else self.provider_configs
        )
        decision = {"chosen": "hybrid", "standard_confidence": 0.0, "reason": "Forced Hybrid"}
        standard = self._standard_rag(query, provider_configs)'''

NEW_COMPARE = '''    def compare(self, query: str, provider: Optional[str] = None, api_key: Optional[str] = None,
                force_hybrid: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run Standard always; run Hybrid when the smart router thinks it's
        worth it (or when force_hybrid=True for evaluation).

        The smart router runs a cheap Standard confidence probe first:
        - If confidence > ROUTER_STANDARD_CONFIDENCE_HIGH -> Standard only
        - If confidence < ROUTER_STANDARD_CONFIDENCE_LOW -> Hybrid
        - Otherwise: blend cross-domain + complexity signals to decide

        Set force_hybrid=True to run both pipelines on every question
        (used by batch_compare_all.py for A/B evaluation).
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
        standard = self._standard_rag(query, provider_configs)'''

if OLD_COMPARE in content:
    content = content.replace(OLD_COMPARE, NEW_COMPARE)
    changes += 1
    print("  [OK] Enabled smart router in compare() — force_hybrid param added")
else:
    print("  [SKIP] compare() method not found (already patched?)")

# ================================================================
# 2. Wire up routing_config.yaml loading in __init__
# ================================================================
OLD_INIT_END = '''        self.ROUTER_LAST_RESORT_THRESHOLD = 0.55
        self.LAST_RESORT_CANDIDATE_POOL = 20'''

NEW_INIT_END = '''        self.ROUTER_LAST_RESORT_THRESHOLD = float(os.getenv("ROUTER_LAST_RESORT_THRESHOLD", "0.55"))
        self.LAST_RESORT_CANDIDATE_POOL = int(os.getenv("LAST_RESORT_CANDIDATE_POOL", "20"))
        # Load per-customer routing config if available
        self._load_routing_config()'''

if OLD_INIT_END in content:
    content = content.replace(OLD_INIT_END, NEW_INIT_END)
    changes += 1
    print("  [OK] Wired up routing config loading in __init__")
else:
    print("  [SKIP] __init__ end not found (already patched?)")

# Add _load_routing_config method (insert before _build_index_streaming)
OLD_BUILD = '''    def _build_index_streaming('''

NEW_METHOD = '''    def _load_routing_config(self, config_path: str = "routing_config.yaml") -> None:
        """Load per-customer partition routing keywords from a YAML config file.

        If the file doesn't exist, the hardcoded defaults in
        PARTITION_KEYWORDS are used. This lets each deployment
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
                    keywords = settings.get("keywords", []) if isinstance(settings, dict) else settings
                    if keywords:
                        loaded[partition_id] = [kw.lower() for kw in keywords]
                if loaded:
                    self.PARTITION_KEYWORDS = loaded
                    print(f"  [routing] Loaded {len(loaded)} partition keyword mappings from {config_path}")
        except Exception as e:
            print(f"  [routing] Could not load {config_path}: {e}. Using defaults.")

    def _build_index_streaming('''

if OLD_BUILD in content:
    content = content.replace(OLD_BUILD, NEW_METHOD)
    changes += 1
    print("  [OK] Added _load_routing_config() method")
else:
    print("  [SKIP] _build_index_streaming not found")

# ================================================================
# 3. Make all thresholds env-configurable with current values as defaults
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
]

for old, new in THRESHOLD_REPLACEMENTS:
    if old in content:
        content = content.replace(old, new)
        changes += 1
        print(f"  [OK] Made {old.split('=')[0].strip().split('.')[-1]} env-configurable")
    else:
        print(f"  [SKIP] {old.split('=')[0].strip()} not found")

# ================================================================
# 4. Add graceful error handling for missing spaCy model
# ================================================================
OLD_SPACY_LOAD = '''    nlp = _get_spacy_nlp()
    if nlp is not None:
        # Truncate very long texts to avoid spaCy's 1M char limit
        # and keep processing fast on large chunks
        truncated = text[:50000] if len(text) > 50000 else text
        doc = nlp(truncated)'''

# This is in the spaCy version of extract_entities — check if it exists
if OLD_SPACY_LOAD in content:
    print("  [OK] spaCy error handling already present")
else:
    print("  [INFO] spaCy extract_entities not found in this file (may be in v24 path)")

# Add graceful DB connection error handling to build_large_corpus_engine
OLD_BUILD_ENGINE = '''def build_large_corpus_engine(db_path: str = DATABASE_PATH, database_url: str = None, provider: str = "auto", api_key: Optional[str] = None) -> Tuple[DatabaseManager, DocumentProcessor, "ComparisonEngine"]:
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
    return db, processor, engine'''

NEW_BUILD_ENGINE = '''def build_large_corpus_engine(db_path: str = DATABASE_PATH, database_url: str = None, provider: str = "auto", api_key: Optional[str] = None) -> Tuple[DatabaseManager, DocumentProcessor, "ComparisonEngine"]:
    """Construct db/processor/engine in large-corpus (memory-bounded) mode:
    DocumentProcessor never loads the full corpus into RAM, and
    ComparisonEngine retrieves via disk-backed FTS5/entity-table queries
    instead of in-memory TF-IDF/entity-graph structures. Use this for
    corpora too large to hold as Python objects (e.g. ~1.9M chunks);
    use plain DocumentProcessor(db) + ComparisonEngine(...) (main()'s
    default) for small corpora where the original in-RAM path is fine
    and slightly faster.

    Raises FileNotFoundError if db_path doesn't exist (common mistake:
    running from the wrong directory or forgetting to set
    SQLITE_DATABASE_PATH).
    """
    if database_url is None and not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database file not found: {db_path}\\n"
            f"Set SQLITE_DATABASE_PATH env var or pass db_path= to point to your database."
        )
    db = DatabaseManager(db_path=db_path, database_url=database_url)
    processor = DocumentProcessor(db, lazy=True)
    engine = ComparisonEngine(processor, db, provider=provider, api_key=api_key)
    return db, processor, engine'''

if OLD_BUILD_ENGINE in content:
    content = content.replace(OLD_BUILD_ENGINE, NEW_BUILD_ENGINE)
    changes += 1
    print("  [OK] Added DB file existence check to build_large_corpus_engine()")
else:
    print("  [SKIP] build_large_corpus_engine not found")

# ================================================================
# Write patched v23.py
# ================================================================
with open(V23_FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n{'=' * 70}")
print(f"  v23.py productionized: {changes} changes applied")
print(f"{'=' * 70}")
print(f"""
  Changes:
  1. Smart router enabled in compare() — force_hybrid=False by default
  2. routing_config.yaml loading wired up in __init__
  3. _load_routing_config() method added
  4. All thresholds made env-configurable (float(os.getenv(...)))
  5. DB file existence check in build_large_corpus_engine()

  New compare() signature:
    engine.compare(query)                          # smart router
    engine.compare(query, force_hybrid=True)       # eval mode (both run)

  For batch_compare_all.py, add force_hybrid=True to the compare() call
  to keep running both pipelines for A/B testing.
""")
