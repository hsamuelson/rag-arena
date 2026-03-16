# Gold Standard Baselines: HotpotQA

**Date:** 2026-03-09 15:48
**Benchmark:** HotpotQA (multi-hop, 50 samples)

## Rankings (sorted by nDCG@10)

| Rank | Backend + Hypothesis | nDCG@10 | MRR | R@K | Latency |
|------|---------------------|---------|-----|-----|---------|
| 1 | hybrid-bm25-dense + cross-encoder | 0.8347 | 0.9367 | 0.8900 | 202ms |
| 2 | hybrid-bm25-dense + flat-baseline | 0.7300 | 0.8006 | 0.8900 | 33ms |
| 3 | direct-embeddings + cross-encoder | 0.6820 | 0.7757 | 0.7300 | 421ms |
| 4 | bm25 + flat-baseline | 0.6787 | 0.8060 | 0.8000 | 1ms |
| 5 | hybrid-bm25-dense + anti-hubness-0.4p | 0.6612 | 0.6656 | 0.8900 | 33ms |
| 6 | hybrid-bm25-dense + pure-csls-5k | 0.6563 | 0.6590 | 0.8900 | 33ms |
| 7 | direct-embeddings + pure-csls-5k | 0.6144 | 0.6925 | 0.7300 | 32ms |
| 8 | direct-embeddings + flat-baseline | 0.5850 | 0.6407 | 0.7300 | 32ms |

## Published Baselines (for reference)
| Method | HotpotQA nDCG@10 |
|--------|-----------------|
| BM25 (published) | 0.603 |
| DPR | 0.391 |
| ColBERTv2 | 0.667 |
| E5-large-v2 | 0.633 |
| SPLADE-v3 | 0.692 |
| BM25+CE (published) | 0.707 |