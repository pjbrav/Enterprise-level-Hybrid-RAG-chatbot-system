# Hybrid RAG v24

Hybrid retrieval (FTS + entity graph + RRF, optional cross-encoder rerank) that answers
questions over a large document DB with fewer LLM tokens than standard RAG.
Roadmap: `docs/IMPLEMENTATION_PLAN.md` (Phase 0 cleanup done; Phases 1-7 = host/join UI).

## Layout
- `rag/` - the engine. `config` (env + constants) -> `text` (tokens, entities) -> `db` -> `ingest`,
  `rankers`, `providers` -> `engine` (`ComparisonEngine`) -> `cli`.
- `v24.py` - thin compatibility shim re-exporting `rag.*`. Edit `rag/`, never `v24.py`.
- `app.py` - Gradio UI. `scripts/` - eval harness and DB maintenance. `tests/` - unit tests.
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
.venv\Scripts\python -m pytest tests -q                       # unit tests (no DB needed)
.venv\Scripts\python scripts\batch_compare_v2.py --force-hybrid --sample 5   # needs an API key in .env
```
Windows console: set `PYTHONUTF8=1` (the engine prints emoji).
