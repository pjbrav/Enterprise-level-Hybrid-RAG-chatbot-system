# Standard vs Hybrid RAG — Batch Comparison
Run on 2026-09-10T13:29:45.802837
Total questions: 300
Database: `hybrid_rag_v22.db`
Questions file: `questionaskerbigset.parquet`
Provider: openai
Mode: forced hybrid

## Summary
| Metric | Standard | Hybrid | Difference |
|--------|----------|--------|------------|
| Chunk relevance | 33.8% | 26.9% | -6.9 pp |
| Answer relevance | 64.2% | 67.4% | +3.1 pp |
| Avg latency | 9.2s | 13.5s | +4.4s |
| Total tokens | 275,604 | 258,109 | -17,495 |

- **Hybrid wins (answer rel):** 146/300 (49%)
- **Standard wins:** 102/300 (34%)
- **Ties:** 52/300 (17%)
- **Cross-domain triggered:** 79/300