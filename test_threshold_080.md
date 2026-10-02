# Standard vs Hybrid RAG — Batch Comparison
Run on 2026-09-14T13:18:09.134146
Total questions: 200
Database: `hybrid_rag_v22.db`
Questions file: `questionaskerbigset.parquet`
Provider: openai
Mode: smart router

## Summary
| Metric | Standard | Hybrid | Difference |
|--------|----------|--------|------------|
| Chunk relevance | 35.0% | 35.0% | +0.0 pp |
| Answer relevance | 65.5% | 65.5% | +0.0 pp |
| Avg latency | 8.1s | 8.1s | +0.0s |
| Total tokens | 177,830 | 177,830 | +0 |

- **Hybrid wins (answer rel):** 0/200 (0%)
- **Standard wins:** 0/200 (0%)
- **Ties:** 200/200 (100%)
- **Cross-domain triggered:** 0/200