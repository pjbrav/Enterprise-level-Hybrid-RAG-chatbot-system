# Hybrid RAG v24 — Cleanup + Host/Join UI Implementation Plan

> Written 2026-10-02 after a full audit of this folder. It replaces
> `hybrid_rag_ui_v24_457a8137.plan.md` (the Cursor plan). Everything that plan
> asked for is kept here, plus corrections where it conflicted with the code.
> **Implement the phases in order.** Phase 0 (cleanup) goes first because it cuts
> the token cost of every later phase.

---

## Ground rules for the implementer

1. **Don't change retrieval behaviour unless a step says to.** The published eval
   numbers (500 q: +0.9pp answer relevance, −7.7% tokens; 300 q: +3.1pp, −6.3%)
   come from current `v24.py` defaults. Hybrid-intensity slider position **0.5
   must reproduce today's defaults exactly**.
2. After every phase, run `python -m pytest tests -q` and a 5-question smoke run:
   `python scripts/batch_compare_v2.py --sample 5 --force-hybrid` (needs an API key).
3. **Never open `data/`, `EnterpriseRAG_Docs/`, `.venv*`, `*.db` or `*.parquet` to
   read them.** Use `sqlite3` queries or `ls` only.
4. Line numbers below refer to `v24.py` as of 2026-10-02 (2,724 lines). After
   Phase 0.4 they map onto `rag/*.py`. Search by function name, not line number.
5. Make one git commit per phase. If on `main`, create a branch first.

---

## Audit findings (why cleanup comes first)

| Problem | Evidence | Cost |
|---|---|---|
| The local git repo tracks **512,031 files**, including the 16 GB DB, a 1.4 GB parquet, 511,960 raw docs, `.env` (OpenAI key) and `__pycache__` | `git ls-files \| wc -l`; `.gitignore` only lists `.venv*/ venv*/ env/` | `.git` is 3.25 GB. Search tools respect `.gitignore`, so any search can wade through 512k files. |
| No git remote. GitHub is published from a hand-copied `github/` mirror, which has already drifted (router defaults differ, `setup.py`/`requirements.txt` reference v23 vs v24) | `git remote -v` is empty; `diff github/v24.py v24.py` | Two copies of every core file, so every edit has to be made twice and read twice |
| 9 one-off patch scripts that rewrote `v23.py`, all already applied | `apply_all_patches.py`, `patch_v24.py`, `patch_extract_entities.py`, `fix_routing.py`, `fix_routing_v2.py`, `fix_stopwords.py`, `productionize.py`, `upgrade_to_v24.py` all `open("v23.py")` | About 110 KB of stale code that shows up in every grep for engine symbols |
| 4 superseded engine versions | `v22.py`, `v23.py`, `v23.py.bak`, `v23_pre_all_patches.py`, `files.zip` (= v23 + batch_compare_all) | About 470 KB (roughly 120k tokens) of near-duplicate code. Grep hits come back 4–5×. |
| Two 5 GB virtualenvs | `.venv` (Py 3.14, matches `__pycache__/v24.cpython-314.pyc`), `.venv311` (Py 3.11) | 5 GB of disk |
| 12 eval-output CSV/MD files at the root, most from older engines | `comparison_*`, `final_results_500.*` | Root clutter |
| `v24.py` is one 136 KB file | 2,724 lines | Each full read costs about 35k tokens. Every new feature below touches it. |
| Duplicate definitions inside `v24.py` | `_get_spacy_nlp` at :220 and :265, `_add_regex_ids` at :234 and :279, duplicate `_SPACY_*` globals at :206/:251 | The first copy is dead code and confuses readers |

---

## Phase 0 — Folder cleanup (do first)

### 0.1 Safety snapshot (2 min)
- Zip **only the small files being deleted** (not the DB, parquet or docs) to
  `../enterpriselevelcheck_archive_2026-10-02.zip`, *outside* the repo. It will be
  about 1 MB. This is the undo button.
- Ask the user to confirm the delete list below before running any delete.

### 0.2 Delete (already applied or superseded)

| File(s) | Reason |
|---|---|
| `apply_all_patches.py`, `patch_v24.py`, `patch_extract_entities.py`, `fix_routing.py`, `fix_routing_v2.py`, `fix_stopwords.py`, `productionize.py`, `upgrade_to_v24.py` | One-shot source patchers for `v23.py`. All applied, and v24 contains the results. |
| `v22.py`, `v23.py`, `v23.py.bak`, `v23_pre_all_patches.py`, `files.zip` | Superseded engines and backups. Git history and the archive zip keep them. |
| `batch_compare.py`, `batch_compare_all.py` | Superseded by `batch_compare_v2.py` (both import v23) |
| `run_eval.py` (v22), `run_eval_hybrid.py` (v23) | Superseded eval runners |
| `check_partitions.py`, `check_progress.py`, `diag_entity_graph.py` | Ingest-time diagnostics for the v23 build. One-off. |
| `test_v23.py` | Identical to `github/test_v24.py` except the import name. Keep the v24 one. |
| `comparison_report_200_regex.md`, `comparison_report_all.md`, `comparison_results*.csv/.md`, `final_results_500.csv/.md` | Results from older engines that the README no longer cites |
| `push_retry.ps1` | One-off docker-push retry loop |
| `hybrid_rag_ui_v24_457a8137.plan.md` | Replaced by this file |
| `__pycache__/` | Build artefact |
| `data/` (empty), `.cache/` (only a 2-file HF cache stub) | Empty. Recreated as needed. |
| `test.parquet` (1.4 GB) | Raw source that `parquet_to_folders.py` turned into `EnterpriseRAG_Docs/`, which is already ingested into the DB. **Ask the user** whether to delete it or move it out of the repo. |
| **One** of `.venv` / `.venv311` | Keep `.venv` (3.14, which produced the current `.pyc`). First verify: `.venv\Scripts\python -c "import spacy, sentence_transformers, gradio"`. If that fails, keep `.venv311` instead. **Ask before deleting** (5 GB). |

### 0.3 Move (keep, but out of the root)

```
enterpriselevelcheck/
├── app.py                       # Gradio UI
├── v24.py                       # becomes a thin re-export shim in 0.4
├── rag/                         # engine package (0.4) + new modules (Phases 1–5)
├── scripts/
│   ├── batch_compare_v2.py      # eval harness (fix sys.path: insert repo root)
│   ├── backfill_entity_frequency.py   # delete once Phase 1.5 lands
│   ├── make_demo_db.py
│   ├── strip_embeddings_db.py
│   ├── profile_sample.py        # port: `import v23` -> `import v24 as v23` (3 call sites)
│   └── parquet_to_folders.py    # documents how EnterpriseRAG_Docs was built
├── tests/
│   └── test_v24.py              # from github/test_v24.py
├── samples/                     # tiny fixture corpus for upload smoke tests
│   ├── documents/               # the 10 synthetic policy docs (was ./documents)
│   └── evaluation_set/          # level1-3 JSON (was ./evaluation_set)
├── eval_results/
│   ├── final_resultsv2_500.csv (+ restore .md from git: `git show HEAD:final_resultsv2_500.md`)
│   ├── final_resultsv2_300.csv/.md
│   └── test_threshold_080.csv/.md
├── data/                        # gitignored
│   ├── hybrid_rag_v22.db        # same-drive move = instant rename
│   ├── questionaskerbigset.parquet
│   ├── corpora/                 # per-corpus SQLite files (Phase 1)
│   └── state.db                 # control plane (Phase 2)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md   # this file
│   └── Hybrid_RAG_Interview_Prep.pdf  # or move outside the repo (personal)
├── routing_config.example.yaml  # from github/routing_config.yaml
├── requirements.txt  requirements-docker.txt  Dockerfile  docker-compose.yml
├── env.example  setup.py  README.md  LICENSE  .gitignore  CLAUDE.md
```

- **`EnterpriseRAG_Docs/` (2.4 GB, 511,960 files): move outside the repo**, e.g.
  `../datasets/EnterpriseRAG_Docs/`. It is only needed to re-ingest from scratch.
  This is the biggest token win, because no tool will ever see those files again.
- Update the defaults after the move:
  - `DATABASE_PATH` default at `v24.py:80` becomes `data/hybrid_rag_v22.db`. Also set `SQLITE_DATABASE_PATH` in `.env`.
  - The `--db`/`--questions` defaults in `batch_compare_v2.py` point at `data/`.
  - In `app.py` `run_eval`, change the subprocess call to `scripts/batch_compare_v2.py` and drop `cwd="/app"`. Use the repo root via `Path(__file__).parent`.
- Fold the `github/` mirror back into the root, then **delete `github/`**:
  - `README.md`: take `github/README.md`, which has the uncommitted 57-line addition.
  - `LICENSE`, `env.example`, `test_v24.py` → `tests/`.
  - `setup.py`: use the github version (`py_modules=["v24"]`, then `packages=["rag"]` after 0.4).
  - `requirements.txt`: rewrite in Phase 5.
  - `github/v24.py`: discard. The local `v24.py` is newer (router HIGH=2.0, pivot 10.0).
  - `github/routing_config.yaml` → `routing_config.example.yaml`. Keep the local `routing_config.yaml` as the live, gitignored config.

### 0.4 Split `v24.py` into a package (mechanical, no behaviour change)

The goal is that later phases read a 300–600-line module instead of 2,700 lines.

| Module | Moves from v24.py |
|---|---|
| `rag/config.py` | module constants and env reads (:70–:125) |
| `rag/text.py` | `clean_text`, `estimate_tokens`, `stem_token`, `query_terms`, stopwords, spaCy/entity extraction (:120–:343). **Delete the dead first copies** of `_get_spacy_nlp`/`_add_regex_ids`/`_SPACY_*` (:206–:250). |
| `rag/db.py` | `ChunkMetadata`, `DatabaseManager`, `estimate_provider_cost` |
| `rag/ingest.py` | `IngestionSummary`, `_hash_file_streaming`, `DocumentProcessor`, `select_files_dialog` |
| `rag/rankers.py` | `build_tfidf_index`, `vector_rank`, `EntityKnowledgeGraph`, `EmbeddingRanker`, `CrossEncoderReranker` |
| `rag/providers.py` | `ProviderConfig`, `available_provider_configs`, `choose_provider`, and the provider-call body of `_generate_answer` (as `call_provider(config, prompt) -> (text, info)`) |
| `rag/engine.py` | `ComparisonEngine`, `export_markdown`, `build_large_corpus_engine` |
| `rag/cli.py` | `print_status`, `main` |

- `v24.py` becomes `from rag.config import *; from rag.text import *; …` and
  re-exports every public **and** underscore name the tests import
  (`extract_entities`, `_ENTITY_STOPWORDS`, …). Then `scripts/` and `tests/`
  keep working unchanged.
- Acceptance: `pytest tests` passes, and a 5-question `batch_compare_v2.py --seed 7`
  run gives the same `chunk_ids` per question before and after.

### 0.5 Git: make this folder the real GitHub clone

The local repo has one commit, no remote, and holds secrets and huge binaries.
Pushing it would be a disaster, and rewriting it isn't worth the effort. **Ask
the user to confirm**, then:

1. Write the new `.gitignore` (below).
2. Move `.git` outside the repo (`../enterpriselevelcheck_old_git`) instead of deleting it.
3. Run `git init -b main`, `git remote add origin https://github.com/pjbrav/Enterprise-level-Hybrid-RAG-chatbot-system.git`, `git fetch origin`, `git reset --soft origin/main`, and commit the restructure on branch `restructure`. Push only if the user asks.
4. Check that `git ls-files | wc -l` is about 40, not 512k.
5. `.env` was only ever in the *local* commit. Check on GitHub that it was never pushed. If it was, rotate the OpenAI key.

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv*/
venv*/
.pytest_cache/
# Secrets
.env
# Data — never commit
data/
*.db
*.db-*
*.parquet
*.zip
uploads/
EnterpriseRAG_Docs/
# Generated results (keep only curated ones under eval_results/)
web_eval*
comparison_results*
latency_test*
# Local config (example is committed)
routing_config.yaml
# Backups / IDE / OS
*.bak
*_pre_*
.vscode/
.idea/
.DS_Store
Thumbs.db
.cache/
```

### 0.6 Add `CLAUDE.md` (keeps future sessions cheap)

Keep it under 40 lines:
- The layout tree above, and the fact that `v24.py` is a shim and the real code is in `rag/`.
- "Never read `data/`, `*.db`, `*.parquet`. The DB has 1.9M chunks and 34M entity rows. Inspect it with `sqlite3`."
- How to run: `python app.py`, `pytest tests`, `scripts/batch_compare_v2.py --sample 5`.
- Eval-number provenance: `eval_results/final_resultsv2_500.*`.

---

## Phase 1 — Storage: Postgres fixes, pool, per-corpus isolation

### 1.1 Corrections to the Cursor plan
- **Corpus isolation without rewriting every query.** Putting a `corpus_id` column on
  every table would mean touching every SQL statement. Use:
  - **SQLite:** one file per corpus, `data/corpora/<corpus_id>.db`. Deleting a corpus is an unlink. This is the default when `DATABASE_URL` is unset.
  - **Postgres:** one schema per corpus, `corpus_<id>`. A pooled connection runs `SET search_path TO corpus_<id>, public` on checkout. `DatabaseManager(…, schema=…)`.
  The existing `DatabaseManager` code is then reused unchanged per corpus.
- **The control plane** (corpora registry, OTPs, memberships, joins) lives in
  `data/state.db` (SQLite), or in Postgres schema `rag_state` when `DATABASE_URL`
  is set. It is a separate class, `StateStore` in `rag/state.py`, and is never
  mixed with chunk tables.

### 1.2 `DatabaseManager` fixes (`rag/db.py`)
- **PRAGMA bug:** `conn.execute("PRAGMA journal_mode=WAL")` runs on every
  `_query`/`_many`/`save_document` (:381, :397, :535). It fails on Postgres
  because psycopg2 connections have no `.execute`. Run it **once** in `_init_db`
  for SQLite only, and skip it for read-only attachments.
- **Pool:** for Postgres, use `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=int(os.getenv("PG_POOL_MAX", 10)))`.
  Add a `_conn()` context manager that does `getconn`/`putconn` and sets `search_path`.
  SQLite stays connect-per-call (cheap, thread-safe).
- **Read-only attach:** the constructor gets a `read_only: bool` parameter. When it is set, open with
  `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`, skip `_init_db`'s
  CREATE statements, and make `save_document`, `save_embeddings`,
  `save_comparison` and `record_usage` no-ops. `hybrid_rag_v22.db` is attached this way.
- **Postgres FTS** (`fts_search`, :620–:640): the code builds `"a | b | c"` and
  passes it to `plainto_tsquery`, which strips the `|`, so this becomes AND semantics. Change to
  `to_tsquery('english', %s)` with the terms sanitised to `[a-z0-9]+`, each
  suffixed `:*` only when the term is 4 characters or longer, and joined with ` | `.
  Keep the GIN index on `chunk_text_tsv`.
- **Postgres term capping:** create a `term_df(term text primary key, df int)` table,
  filled after each ingest from
  `SELECT word, ndoc FROM ts_stat('SELECT chunk_text_tsv FROM chunks')`.
  `_cap_fts_terms` reads it on both backends. SQLite keeps `fts5vocab`.
- **Entity frequency:** at the end of `DocumentProcessor.process_uploads`, call a new
  `db.rebuild_entity_frequency()` with the logic from `backfill_entity_frequency.py`.
  Then delete that script.

### 1.3 Corpus registry (`rag/state.py`)
```
corpora(id TEXT PK, name, kind TEXT CHECK(kind IN ('hosted','personal')),
        backend TEXT, location TEXT, read_only INT, retained INT,
        partition_keywords_json TEXT, created_at)
```
- `retained=0` (one-shot) corpora are deleted on app exit (`atexit`) and swept on startup.
- `EngineCache` holds an LRU of `corpus_id -> (db, processor, engine)` behind a lock. A corpus is
  invalidated on ingest or delete. Every engine is built with `build_large_corpus_engine(...)`.

**Acceptance:** with `DATABASE_URL` pointing at docker-compose Postgres, ingest
`samples/documents/` into a new corpus and run a hybrid query. No PRAGMA error occurs,
`fts_search(["remote","work"])` returns chunks containing *either* word, and
`entity_frequency` is populated.

---

## Phase 2 — Hosting: OTP, membership, read-only retrieval API

### 2.1 Engine refactor that remote retrieval depends on (`rag/engine.py`)
Today `_run_rag` (:2384) mixes retrieval, `db` lookups and generation, so a
member can't feed host chunks into it. Split it:

```python
@dataclass
class RetrievedChunk:
    chunk_id: str        # namespaced: "host:<id>" or "mine:<id>" once fused
    text: str
    score: float         # calibrated 0-1
    partition_id: str
    source: str          # file path / doc title for display
    origin: str          # "host" | "mine"

@dataclass
class RetrievalResult:
    candidates: List[RetrievedChunk]   # ranked pool, BEFORE _intelligent_retrieve selection
    is_cross_domain: bool
    threshold: float
    max_chunks: int
    path: str                          # standard_rag | hybrid | hybrid_last_resort
    latency: float

def retrieve(self, query, use_hybrid: bool, params: HybridParams, pool: int = 20) -> RetrievalResult
def select(self, query, result: RetrievalResult) -> Tuple[List[RetrievedChunk], float]   # _intelligent_retrieve on texts
def answer(self, query, method, selected, confidence, result, provider_configs) -> Dict   # same dict shape as today
```
- Change `_build_context`, `_calculate_confidence`, `_validate_answer_quality`,
  `_source_summary` and `_calculate_chunk_relevance` to take an optional
  `texts: Dict[str,str]` / `meta: Dict[str,ChunkMetadata]`. When it is given, they skip the DB.
- `_run_rag` becomes `retrieve → select → answer`. It must return identical
  output (same acceptance check as 0.4).

### 2.2 Host API (`rag/hosting.py`, FastAPI + uvicorn)
- **Two ports, deliberately:** Gradio (host controls, keys, library) binds
  **`127.0.0.1:7860`** only. The retrieval API binds `HOST_BIND` (default
  `0.0.0.0:8765`). The LAN must never reach the admin UI.
- Start the API in a daemon thread with `uvicorn.Server(config)` when the user clicks *Start
  hosting*. *Stop hosting* sets `server.should_exit = True`.

| Endpoint | Auth | Body → Response |
|---|---|---|
| `GET /v1/health` | none | `{ok, corpus_name, version}` |
| `POST /v1/join` | OTP | `{otp, display_name}` → `{token, corpus_name, partition_keywords}` |
| `POST /v1/retrieve` | Bearer | `{query ≤2000 chars, method: "hybrid"\|"standard", intensity 0-1, pool ≤50}` → `RetrievalResult` JSON (texts truncated to `CONTEXT_CHARS_PER_CHUNK`) |
| `POST /v1/leave` | Bearer | deletes own membership |

- There are no ingest, delete, list-documents or get-chunk-by-arbitrary-id endpoints. A member
  can only get the top-`pool` chunks for a query.
- The host never calls `save_comparison` on member queries and never generates answers for
  members. The member's own key pays.

### 2.3 OTP and membership (`rag/state.py`)
```
otp_codes(id PK, corpus_id, code_hash, salt, expires_at, max_uses, uses, revoked)
memberships(id PK, corpus_id, display_name, token_hash, ip, joined_at, last_seen)
```
- OTP: `f"{secrets.randbelow(10**6):06d}"`. Store `sha256(salt+code)` and compare with
  `hmac.compare_digest`. Defaults: TTL 30 min, 10 uses.
- **Brute-force guard** (6 digits is only 10⁶ codes): 5 failed `/join` attempts per IP
  per 10 min returns 429. After 50 total failures on a corpus, auto-revoke all codes and show a
  banner in the host UI.
- Token: `secrets.token_urlsafe(32)`. Store only its sha256. Every `/retrieve` updates
  `last_seen`.
- **Kick** deletes the membership row, so the next request gets 401 immediately.
- **Rotate** revokes all unused codes and issues a new one. Existing members stay.
- **Stop hosting** stops the server. Memberships are kept unless the user picks "and remove all members".
- UI copy has to say: traffic is plain HTTP on the LAN, so use it on a trusted network or
  behind a tunnel (e.g. `cloudflared`). The member only needs to be able to reach the host.

**Acceptance:** an httpx/TestClient test covers wrong OTP → 401, 6th bad attempt → 429,
expired OTP → 401, kicked token → 401, and that `/retrieve` returns at most
`pool` chunks.

---

## Phase 3 — Member client and fusion (`rag/remote.py`)

- `joins(id PK, host_url, corpus_name, display_name, token, partition_keywords_json, joined_at)`
  in the **member's** `state.db`. This is the only thing stored for a host. There are no chunk tables
  and no files. (The token is a bearer secret in a local file, which is acceptable for a desktop app. Mention it in the UI.)
- `RemoteCorpus.retrieve(query, method, intensity, pool)` calls `httpx.post(..., timeout=30)` and
  returns a `RetrievalResult` with ids prefixed `host:`. Results stay in RAM for one request only.
- **Fusion** (`fuse(results: List[RetrievalResult]) -> RetrievalResult`): RRF
  (k=60, same as `_fuse_rankings`) over the namespaced candidate lists. Then run
  `engine.select()` on the fused pool, using the `texts`/`meta` dicts from the
  results. That is why 2.1 makes those helpers dict-capable.
- If the host is down or returns 401, show a warning ("shared corpus unavailable: kicked or
  offline") and continue with local corpora only. If the response is 401, mark the join as revoked.
- Sources in the answer are tagged `[host]` or `[mine]`.

**Acceptance (Phase 7 smoke):** after joining, the member's `data/` directory grows by
less than 10 KB (only `state.db` rows).

---

## Phase 4 — Universal routing (any corpus, not just the 9 enterprise sources)

- `PARTITION_KEYWORDS` is a **class** attribute (:1657) that `_load_routing_config`
  overwrites. Make it an instance attribute. Resolution order: YAML override, then
  `corpora.partition_keywords_json`, then `_DEFAULT_PARTITION_KEYWORDS`, which is used
  only if the corpus actually has those `partition_*` ids.
- New `rag/routing_autogen.py`, `generate_partition_keywords(db) -> Dict[str, List[str]]`, runs
  at the end of ingest. For each partition it combines:
  1. folder-derived name tokens
  2. top-8 TF-IDF terms over a sample of at most 2,000 chunks (sklearn `TfidfVectorizer`, `max_df=0.5`)
  3. top-5 spaCy ORG/PRODUCT entities from `entity_frequency`, filtered to entities concentrated in that partition (≥60% of mentions)
  Drop terms that appear in more than half of the partitions.
- Rename opaque `partition_cluster_N` labels to `partition_<top1>_<top2>`. Do this in
  `_auto_partition_documents` (:976) by reusing `_keyword_label` (:956), and use it only for **new**
  corpora. Never rename partitions in the existing 16 GB DB.
- Show the generated keywords in the Library tab and allow edits. Edits save to `partition_keywords_json`.
- `_get_cross_domain_rankings_disk` (:2025): run the per-partition
  `_rank_candidates` calls in a `ThreadPoolExecutor(max_workers=len(active))`. SQLite
  connect-per-call is thread-safe, and the PG pool handles concurrency.
- Neutral prompt: in `_generate_answer` (:2303), change "from research papers" and "the cited papers" to
  "from the retrieved documents". Change the `select_files_dialog` title (:1244) to "Select documents".
- Chunking: add a heading-aware pre-split in `_chunk_document` (:1049). Split on `^#{1,6} `
  markdown headings and on PyMuPDF blocks whose font size is ≥1.2× the median, then apply the
  existing 220-word/25-overlap windows inside each section. This applies to new ingests only.

---

## Phase 5 — Hybrid-intensity slider and providers

### 5.1 Correction to the Cursor plan
The Cursor plan's "Low" row says to lower `ROUTER_STANDARD_CONFIDENCE_HIGH` so that Standard
is skipped more. **The router never runs in the UI.** Production calls
`compare_single("hybrid_rag")` and Testing forces both pipelines, so router thresholds
have no effect there. **The slider only controls how much work Hybrid does.**

### 5.2 `HybridParams` (`rag/params.py`), passed per request
The engine is shared across Gradio threads, so **do not mutate engine attributes**
(that is a race between concurrent users). Pass `params` through
`retrieve → _get_cross_domain_rankings_disk → _rank_candidates_disk`
and `_last_resort_rerank`.

```python
@dataclass(frozen=True)
class HybridParams:
    lexical_limit: int          # replaces class const LEXICAL_CANDIDATE_LIMIT (30)
    max_partitions: int         # replaces hardcoded [:3] in _get_active_partitions
    last_resort_threshold: float  # ROUTER_LAST_RESORT_THRESHOLD (0.40); 0.0 = never rerank
    last_resort_pool: int       # LAST_RESORT_CANDIDATE_POOL (20)

def params_for_intensity(s: float) -> HybridParams
```

| s | lexical_limit | max_partitions | last_resort_threshold | last_resort_pool |
|---|---|---|---|---|
| 0.0 | 15 | 1 | 0.0 (never) | 10 |
| **0.5** | **30** | **3** | **0.40** | **20** |  ← must equal today's defaults |
| 1.0 | 80 | 3 | 0.65 | 40 |

Interpolate piecewise-linearly between those rows, and round the integers.
`params_for_intensity(0.5)` must equal `HybridParams.from_env()`. Add a test for that.
The Standard pipeline ignores params.

### 5.3 Providers (`rag/providers.py`)
- **`key_override` bug** (:1450): the override applies only when `preferred == name`, so
  with `preferred="auto"` a pasted key is silently ignored. Remove "auto" from the UI.
  The user always picks a provider, and the override applies to that one. Keep the env fallback for the rest.
- Add a first-class `ollama` provider: an OpenAI-compatible client,
  `base_url=os.getenv("OLLAMA_BASE_URL","http://localhost:11434/v1")`, `api_key="ollama"`,
  `model=os.getenv("OLLAMA_MODEL","llama3.1")`. It is "configured" whenever a model is set. No key is needed.
- Let a per-request `model` override come from the UI's model textbox.
- `app.py` currently writes the user's key into `os.environ["OPENAI_API_KEY"]` (app.py:27, :92).
  **Remove that.** It leaks one user's key to every other session. Keys are passed
  explicitly and never written to disk.
- Remove the `sk-` prefix check (app.py:22).

---

## Phase 6 — Gradio rewrite (`app.py`)

Tabs: **Search | My library | Host | Join | Evaluate | About**

- **Global (top bar):** provider dropdown (OpenAI / Anthropic / Gemini / Ollama / OpenAI-compatible),
  API key (password, `gr.State` only), model textbox, base-URL textbox (shown for compatible only).
- **Search:**
  - Corpus checklist: my local corpora plus active joins, tagged `[host] name`.
  - Mode radio: **Production** runs Hybrid only. **Testing** runs Standard and Hybrid **in parallel**
    (`ThreadPoolExecutor(2)`), with no copy of Standard into the Hybrid slot.
  - Hybrid-intensity slider (0–1, default 0.5, labelled "faster ← → more thorough").
  - Testing shows two columns. Each has the answer, latency, tokens, token savings %,
    retrieval confidence, chunk relevance, answer relevance, chunk count and sources.
    Label the relevance numbers "lexical relevance (proxy)". **Never call them "accuracy".**
- **My library:** upload files or a ZIP, then choose Retain or One-shot, then
  `process_uploads` with a progress bar. Show an `IngestionSummary.as_markdown()`.
  The corpus table (name, docs, chunks, partitions, keywords, retained) has Delete,
  which drops the corpus file or schema. There is also an "Attach existing SQLite DB (read-only)"
  file path box for `data/hybrid_rag_v22.db`.
- **Host:** pick a corpus (hosted kind), then Start/Stop hosting. Show the join URL using
  the LAN IP from `socket.gethostbyname(socket.gethostname())`, with a fallback to the UDP-connect trick.
  Add Issue OTP (shows the code and expiry), Rotate, and a members table (name, IP,
  joined, last seen) with a Kick button per row (a `gr.Dataframe` plus row select plus a Kick button),
  and the failure-count banner. Note: "Windows will ask to allow Python through the firewall."
- **Join:** host URL, OTP and display name, then a Join button. A joins table has Leave. Leaving deletes
  the local token only.
- **Evaluate:** keep the existing batch-eval button (Testing mode on the attached DB).
- `app.launch(server_name=os.getenv("GRADIO_BIND","127.0.0.1"), server_port=7860)`.
  Docker sets `GRADIO_BIND=0.0.0.0`.
- Remove the startup `build_large_corpus_engine()` global (app.py:12). Engines come from
  `EngineCache` lazily, so the app starts instantly even without the 16 GB DB.

---

## Phase 7 — Packaging and smoke test

- `requirements.txt`: numpy, pandas, pyarrow, scikit-learn, spacy, sentence-transformers,
  PyYAML, openai, **anthropic, google-genai, psycopg2-binary, pymupdf, gradio>=4,
  fastapi, uvicorn, httpx**, plus pytest in `requirements-dev.txt`. Make `requirements-docker.txt` match.
- `env.example`: `DATABASE_URL=` (blank = SQLite), `SQLITE_DATABASE_PATH=data/hybrid_rag_v22.db`,
  `HOST_BIND=0.0.0.0:8765`, `GRADIO_BIND=127.0.0.1`, `PG_POOL_MAX=10`,
  `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, provider keys and models, and the current router defaults (HIGH=2.0,
  LOW=0.42, LAST_RESORT=0.40, `LEXICAL_CALIBRATION_PIVOT=10.0`). There is no `ADMIN_SETUP_CODE`.
- `docker-compose.yml`: a `pgvector/pgvector:pg16` service with a volume and healthcheck, and an
  `app` service that publishes 7860 and 8765 and mounts `./data:/app/data`.
- **Dockerfile:** remove `COPY hybrid_rag_v22.db` (it bakes 16 GB into the image, which is why
  `push_retry.ps1` existed). Mount `data/` as a volume instead. Change the
  `COPY` line to `COPY app.py v24.py routing_config.example.yaml /app/` plus `COPY rag/ scripts/`.
- **Two-process smoke test** (`tests/smoke_two_instances.md`, run manually, or scripted with
  two `DATA_DIR`s on ports 7860/7861 and 8765/8766):
  1. Instance A ingests `samples/documents/` (retain), hosts it, and issues an OTP.
  2. Instance B joins with no files. `du data/` grows by less than 10 KB, and a query returns `[host]` sources.
  3. B uploads its own file and queries. Sources are mixed `[host]` and `[mine]`. A's corpus file is unchanged (compare mtime and sha).
  4. A kicks B. B's next query shows the "shared corpus unavailable" banner and still answers from `[mine]`.
  5. A wrong OTP 6 times in a row returns 429.
  6. Testing mode shows two columns. Latency is about the max of the two, not their sum.

---

## Out of scope (unchanged from the Cursor plan)
Syncing or shipping the 16 GB DB to members; SSO or email; host-paid LLM keys; NAT
traversal or a cloud relay; file types beyond pdf/txt/md/py/js/ipynb/zip;
auto-migrating `hybrid_rag_v22.db` into Postgres (optional later
`scripts/migrate_sqlite_to_pg.py`); LLM-as-judge (optional later, Testing only).

## Phase checklist
- [ ] 0 Cleanup, package split, git re-init, CLAUDE.md
- [ ] 1 Storage: PRAGMA, pool, FTS, read-only attach, per-corpus isolation, entity_frequency
- [ ] 2 retrieve/select/answer split, host API, OTP and members
- [ ] 3 Member client, fusion, offline handling
- [ ] 4 Auto keywords, cluster labels, parallel partitions, neutral prompt, heading chunking
- [ ] 5 HybridParams slider, providers (Ollama, key override, no env leak)
- [ ] 6 Gradio rewrite
- [ ] 7 Requirements, env, compose, Dockerfile, two-instance smoke test
