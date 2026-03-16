# Round 5: Novel Approaches with Statistical Validation

**Date:** 2026-03-09 17:07
**Samples:** 200 (statistically powered)
**Benchmark:** HotpotQA (~2000 docs)
**Backend:** Hybrid BM25+Dense
**Statistical tests:** Bootstrap 95% CI, paired permutation test (5000 perms)

## Results (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | 95% CI | vs CE | p-value | Sig? |
|------|-----------|---------|--------|-------|---------|------|
| 1 | deep-pool-30-ce | 0.8352 | [0.8068, 0.8621] | +6.1% | 0.0000 | YES |
| 2 | ce-cross-doc-5k | 0.7919 | [0.7583, 0.8249] | +0.6% | 0.0750 | no |
| 3 | ce-coverage-greedy-0.2 | 0.7913 | [0.7559, 0.8227] | +0.5% | 0.0318 | YES |
| 4 | ce-with-context | 0.7912 | [0.7568, 0.8218] | +0.5% | 0.1142 | no |
| 5 | ce-MiniLM-L-12 | 0.7891 | [0.7556, 0.8200] | +0.2% | 0.3276 | no |
| 6 | ce-query-type-adaptive | 0.7885 | [0.7546, 0.8200] | +0.2% | 0.0296 | YES |
| 7 | ce-multihop-iterative | 0.7880 | [0.7542, 0.8194] | +0.1% | 0.2984 | no |
| 8 | ce-keyword-focused-0.5 | 0.7873 | [0.7531, 0.8183] | +0.0% | 0.5088 | no |
| 9 | cross-encoder (baseline) | 0.7872 | [0.7526, 0.8186] | baseline | - | - |
| 10 | ce-answer-extraction | 0.7865 | [0.7520, 0.8170] | -0.1% | 1.0000 | no |
| 11 | bm25-boosted-ce-0.15 | 0.7830 | [0.7498, 0.8122] | -0.5% | 1.0000 | no |
| 12 | ce-negative-feedback | 0.7769 | [0.7443, 0.8071] | -1.3% | 1.0000 | no |
| 13 | ce-sentence-max | 0.7652 | [0.7305, 0.7978] | -2.8% | 1.0000 | no |
| 14 | query-decomp-ce-mean | 0.7521 | [0.7212, 0.7837] | -4.5% | 1.0000 | no |
| 15 | query-decomp-ce-max | 0.7397 | [0.7060, 0.7725] | -6.0% | 1.0000 | no |
| 16 | ce-sentence-top2 | 0.7225 | [0.6845, 0.7570] | -8.2% | 1.0000 | no |

## Interpretation

### Statistically Significant Improvements (p < 0.05)

1. **Deep Pool CE (+6.1%, p<0.0001)** — THE winner. Retrieving 30 candidates instead of 10,
   then CE reranking to top-10. This confirms: **recall is the bottleneck**. With a deeper
   candidate pool, CE finds more relevant documents. This is the only approach with a large,
   practically significant improvement.

2. **Coverage-Greedy (+0.5%, p=0.032)** — Statistically significant but tiny effect.
   Query-term coverage-based greedy selection adds marginal signal.

3. **Query-Type Adaptive (+0.2%, p=0.030)** — Barely significant, negligible effect.
   Adapting scoring by query type (bridge vs comparison vs factoid) helps slightly.

### Not Significant (noise)

All other approaches (12 of 15) are statistically indistinguishable from plain CE:
- Larger model (MiniLM-L-12): +0.2%, p=0.33
- Document context: +0.5%, p=0.11
- Cross-document scoring: +0.6%, p=0.075
- Keyword focusing: +0.0%, p=0.51
- Multi-hop iterative: +0.1%, p=0.30

### Methods that HURT

- Query decomposition (heuristic, no LLM): -4.5% to -6.0%
- Sentence-level scoring: -2.8% to -8.2%
- Negative feedback: -1.3%
- BM25 boost: -0.5%

### Key Takeaway

**The only way to meaningfully improve over CE is to give it more candidates.**
Scoring strategy tweaks are noise. The bottleneck is recall, not ranking quality.
To go beyond deep-pool-30, the next frontier is multi-hop retrieval with LLM
(IRCoT: +8-15% published) — but this requires an LLM in the loop.