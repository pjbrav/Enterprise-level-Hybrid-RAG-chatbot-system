# Standard vs Hybrid RAG — Batch Comparison
Run on 2026-08-31T19:54:16.127828
Total questions: 500
Database: `hybrid_rag_v22.db`
Questions file: `questionaskerbigset.parquet`
Provider: openai
Mode: forced hybrid

## Summary
| Metric | Standard | Hybrid | Difference |
|--------|----------|--------|------------|
| Chunk relevance | 32.9% | 26.6% | -6.3 pp |
| Answer relevance | 30.0% | 26.3% | -3.7 pp |
| Avg latency | 2.2s | 24.1s | +21.9s |
| Total tokens | 460,231 | 424,985 | -35,246 |

- **Hybrid wins (answer rel):** 118/500 (24%)
- **Standard wins:** 200/500 (40%)
- **Ties:** 182/500 (36%)
- **Cross-domain triggered:** 140/500