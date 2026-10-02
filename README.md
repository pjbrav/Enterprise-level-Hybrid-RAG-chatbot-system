# Hybrid RAG — Enterprise Search You Can Host and Share

Upload a large document set once, then ask questions over it with **fewer LLM tokens and better
evidence** than standard RAG. Anyone can **host** a corpus from their own machine and let others
query it remotely with a one-time code, without those people ever receiving the files or the database.

Hybrid RAG combines FTS lexical search, an entity knowledge graph, Reciprocal Rank Fusion (RRF),
optional cross-encoder reranking and partition-based routing for multi-source data.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp env.example .env                                     # optional: API keys, Postgres, ports
python app.py                                           # http://127.0.0.1:7860
```

In the UI:

1. **My library** — upload files or a ZIP (pdf, txt, md, py, js, ipynb). Routing keywords are generated for you.
2. **Search** — tick a corpus, ask a question. Pick your provider (OpenAI, Anthropic, Gemini, Ollama, or any
   OpenAI-compatible server) and paste a key; it stays in memory and is never saved.
   - **Production** runs Hybrid only. **Testing** runs Standard and Hybrid side by side, in parallel, with latency,
     tokens, token savings and relevance proxies for each.
   - **Hybrid intensity** (0 → 1) sets how much work Hybrid does. 0.5 is the configuration the published results used.

## Share a corpus (host / join)

Any machine can be a host. Hosting is per corpus, not a special install.

- **Host tab:** pick a corpus, *Start hosting*, then *Issue one-time code*. Give the URL and the 6-digit code to a colleague.
  You see who joined, can *kick* anyone (their access stops immediately), *rotate* codes, and stop hosting.
- **Join tab:** enter the host URL, the code and your name. **Nothing is downloaded** — only a membership token is stored
  (a few KB). Tick the shared corpus under Search; results are tagged `[host]` and your own documents `[mine]`, and both
  can be searched together.
- Members receive only the top few chunks for each question, held in memory for that request. They cannot add, change or
  delete anything. Answers are generated on the member's machine with the member's own API key, so the host is never billed.
- If the host is offline or has kicked you, search keeps working on your own documents and tells you why.

Security notes: traffic is **plain HTTP** — use a trusted network or a tunnel. The one-time code is the only secret; wrong
codes are rate limited per address, and a burst of failures revokes all outstanding codes. The admin UI binds to
`127.0.0.1`; only the host API (port 8765) is exposed. When the UI is bound to a network address, the server's own API
keys are not used (visitors must type theirs) unless `UI_ALLOW_SERVER_KEYS=1`.

## Architecture

```
Query → spaCy NER → Partition Routing → FTS (BM25) + Entity Graph Search   (per partition, in parallel)
                                        ↓
                                   RRF Fusion
                                        ↓
                    (rarely) Cross-encoder + embedding rerank
                                        ↓
                              Top Chunks → LLM → Answer
```

**Standard RAG:** query → FTS search → top chunks → LLM → answer.
**Hybrid RAG:** query → entity extraction → partition routing → per-partition FTS + entity graph → RRF fusion →
optional rerank → LLM → answer.

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Lexical search | SQLite FTS5 / PostgreSQL tsvector | Term-frequency ranking |
| Entity extraction | spaCy NER + regex IDs | Named entity → chunk mapping |
| Entity graph | `chunk_entities` table | Cross-partition entity matching |
| Rank fusion | RRF (k=60) | Merge lexical + entity rankings; also merges host and local results |
| Reranking | cross-encoder/ms-marco-TinyBERT | Last-resort semantic scoring |
| Routing | Generated per-corpus keywords (+ optional YAML) | Choose which partitions to search |
| Storage | One SQLite file or one Postgres schema per corpus | Isolation; delete = drop |

## Results (500-question evaluation)

| Metric | Standard | Hybrid | Difference |
|--------|----------|--------|------------|
| Answer relevance | 65.2% | 66.1% | +0.9 pp |
| Token usage | 459,014 | 423,874 | -7.7% |
| Latency | 9.4s | 17.1s | +7.7s |
| Win rate (answer rel) | 36% | 39% | +3 pp |

Hybrid wins 39% of questions vs Standard's 36%, while saving 7.7% tokens. Relevance figures are lexical-overlap
proxies, not accuracy. Raw outputs are in `eval_results/`.

## Dataset

- 1,924,473 chunks across 9 enterprise data sources
- 34M entity rows in the knowledge graph
- 500 evaluation questions across 10 question types
- Sources: Slack, Gmail, GitHub, Jira, Confluence, HubSpot, Linear, Fireflies, Google Drive

**Dataset source:** [ADD HUGGINGFACE LINK HERE]

To use the existing database, put it at `data/hybrid_rag_v22.db` and attach it read-only under *My library → Attach*
(it is never copied or modified).

## Configuration

Everything is optional and documented in [`env.example`](env.example): storage backend, ports, provider keys and models
(including Ollama), and the retrieval thresholds. Per-corpus routing keywords are generated at ingest and editable in the
UI; `routing_config.yaml` (copy [`routing_config.example.yaml`](routing_config.example.yaml)) overrides them.

### PostgreSQL / Docker

```bash
docker compose up --build        # app on :7860 + PostgreSQL with pgvector; data in ./data
```

The image contains code only. Corpora live in Postgres (one schema each); mount `./data` to attach large SQLite
databases. Set `DATABASE_URL` yourself to use an existing server.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q                       # unit tests (Postgres tests need TEST_DATABASE_URL)
python scripts/smoke_two_instances.py           # real two-process host/join scenario, no LLM calls
python scripts/batch_compare_v2.py --force-hybrid --sample 50     # evaluation harness (needs a provider key)
```

## Files

| Path | Purpose |
|------|---------|
| `app.py` | Gradio UI (layout only) |
| `rag/` | Engine and services: `engine`, `db`, `ingest`, `rankers`, `providers`, `params` (slider), `corpora` (library), `hosting` + `remote` (share/join), `state`, `routing_autogen`, `app_controller` |
| `v24.py` | Compatibility shim re-exporting the engine (`from v24 import ...` still works) |
| `scripts/` | Evaluation harness, database tools, two-process smoke test |
| `tests/` | Unit and integration tests |
| `samples/` | Small fixture documents and evaluation question sets |
| `eval_results/` | Curated evaluation outputs |
| `docs/` | Implementation plan and notes |

## License

MIT
