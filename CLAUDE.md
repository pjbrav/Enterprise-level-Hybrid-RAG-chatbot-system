# Hybrid RAG v24

Hybrid retrieval (FTS + entity graph + RRF, optional cross-encoder rerank) that answers
questions over a large document DB with fewer LLM tokens than standard RAG.
Roadmap: `docs/IMPLEMENTATION_PLAN.md` (all phases implemented; see each phase's STATUS note for deviations).

## Layout
- `rag/` - the engine. `config` (env + constants) -> `text` (tokens, entities) -> `db` -> `ingest`,
  `rankers`, `providers` -> `engine` (`ComparisonEngine`) -> `cli`. `state` (corpora registry, `data/state.db`)
  and `corpora` (`CorpusManager`: create/attach/ingest/delete + engine cache) sit on top. `retrieval` holds the
  retrieve->select->answer data types (`RetrievalResult`, `ChunkStore`, wire format); `hosting` is the read-only
  FastAPI host API + OTP/membership (`HostServer`); `remote` is the member side (`join_host`, `RemoteCorpus`,
  `FederatedSearch`: search any mix of local corpora and joined hosts; host chunks stay in RAM, never on disk).
- One corpus = one SQLite file (`data/corpora/<id>.db`) or one Postgres schema (`corpus_<id>`); attached DBs are read-only.
- `v24.py` - thin compatibility shim re-exporting `rag.*`. Edit `rag/`, never `v24.py`.
- `app.py` - Gradio UI (layout only; logic in `rag/app_controller.py`). `scripts/` - eval harness, DB tools, two-process
  smoke test. `tests/` - unit/integration tests. Slider = `rag/params.py` (per-request, never set on the shared engine).
- `samples/` - tiny fixture docs + eval question sets. `eval_results/` - curated results the README cites.
- `data/` (gitignored) - `hybrid_rag_v22.db` (16 GB, 1.9M chunks, 34M entity rows), `questionaskerbigset.parquet`
  (500 eval questions), future `corpora/` and `state.db`.
- Raw source docs live OUTSIDE the repo at `../datasets/EnterpriseRAG_Docs/` (512k files).

## Do not
- Read or glob `data/`, `*.db`, `*.parquet`, `../datasets/`. Inspect the DB with `sqlite3` queries only.
- Mutate engine attributes per request (the engine is shared across threads); pass per-request params.
- Change retrieval defaults without re-running the eval: published numbers (500 q: +0.9pp answer relevance,
  -7.7% tokens) come from the current defaults in `rag/engine.py` / `rag/config.py`.

## Run
```
.venv\Scripts\python app.py                                   # UI on :7860
.venv\Scripts\python -m unittest discover -s tests -t .       # all tests (pytest also works once installed)
$env:TEST_DATABASE_URL="postgresql://postgres@127.0.0.1:5544/postgres"   # optional: also run the Postgres tests
.venv\Scripts\python scripts\smoke_two_instances.py            # real host + member processes, no LLM calls
.venv\Scripts\python scripts\batch_compare_v2.py --force-hybrid --sample 5   # needs an API key in .env
```
Windows console: set `PYTHONUTF8=1` (the engine prints emoji).

## Gotchas
- `.env` is loaded at import, so real API keys may be in `os.environ`: tests must never reach a real LLM
  (pass `provider_configs=[]`, or clear the keys / set `GRADIO_BIND=0.0.0.0`).
- Postgres tests need a throwaway server; never point `TEST_DATABASE_URL` at a database you care about.
- Shell heredocs with big Python blocks break in this environment: write a script file, then run it.
