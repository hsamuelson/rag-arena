# Round 6: Top LLM-Powered Hypotheses

**Date:** 2026-03-09 19:32
**Samples:** 200
**Benchmark:** HotpotQA
**Backend:** Hybrid BM25+Dense
**LLM:** Qwen 3.5 122B via Ollama
**Stats:** Bootstrap 95% CI, paired permutation test (5000 perms)

## Results (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | 95% CI | vs CE | vs DP30 | p-value | Sig? | Time |
|------|-----------|---------|--------|-------|---------|---------|------|------|
| 1 | deep-pool-50-ce | 0.8561 | [0.8328, 0.8793] | +8.8% | +2.5% | 0.0000 | YES | 213s |
| 2 | llm-query-decomp-ce | 0.8532 | [0.8279, 0.8773] | +8.4% | +2.1% | 0.0000 | YES | 3651s |
| 3 | deep-pool-ircot-ce | 0.8514 | [0.8264, 0.8752] | +8.2% | +1.9% | 0.0000 | YES | 1528s |
| 4 | deep-pool-30-ce | 0.8352 | [0.8068, 0.8621] | +6.1% | baseline | 0.0000 | YES | 148s |
| 5 | ircot-simplified | 0.8149 | [0.7851, 0.8452] | +3.5% | -2.4% | 0.0002 | YES | 1375s |
| 6 | cross-encoder (baseline) | 0.7872 | [0.7526, 0.8186] | baseline | -5.8% | - | - | 70s |

## Key Questions Answered

1. **Does deeper pooling keep helping?** (pool-50 vs pool-30)
2. **Does LLM multi-hop retrieval beat deep pooling?** (IRCoT vs pool-30)
3. **Do they stack?** (deep-pool + IRCoT vs either alone)