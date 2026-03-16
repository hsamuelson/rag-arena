# Round 4: Final Push

**Date:** 2026-03-09 16:05
**Benchmark:** HotpotQA (multi-hop, 50 samples, 491 docs)
**Backend:** Hybrid BM25+Dense

## Rankings

| Rank | Hypothesis | nDCG@10 | MRR | R@K | Latency |
|------|-----------|---------|-----|-----|---------|
| 1 | ce-title-multiwindow-0.15 | 0.8558 | 0.9667 | 0.8900 | 454ms |
| 2 | ce-title-multiwindow-0.2 | 0.8496 | 0.9567 | 0.8900 | 440ms |
| 3 | ce-title-boost-0.3 | 0.8449 | 0.9567 | 0.8900 | 322ms |
| 4 | ce-title-boost-0.2 | 0.8441 | 0.9567 | 0.8900 | 325ms |
| 5 | ce-title-boost-0.25 | 0.8437 | 0.9567 | 0.8900 | 322ms |
| 6 | cross-encoder | 0.8347 | 0.9367 | 0.8900 | 276ms |
| 7 | ce-multihop-iterative | 0.8344 | 0.9367 | 0.8900 | 364ms |
| 8 | ce-multihop-iterative | 0.8320 | 0.9367 | 0.8900 | 365ms |
| 9 | ce-ensemble-rrf-2m | 0.8228 | 0.9200 | 0.8900 | 304ms |
| 10 | ce-ensemble-avg-2m | 0.8209 | 0.9200 | 0.8900 | 260ms |