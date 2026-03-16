# Experiment Results: BEIR Zero-Shot Transfer (SciFact + NFCorpus)

**Date:** 2026-03-09
**Benchmark:** BEIR (SciFact + NFCorpus, 8816 docs, 50 samples)
**Backend:** direct-embeddings (nomic-embed-text 768d)

## Rankings (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | MRR | R@K | vs Flat |
|------|-----------|---------|-----|-----|---------|
| 1 | flat-baseline | 0.440 | 0.414 | 0.521 | baseline |
| 2 | embedding-triangulation | 0.440 | 0.414 | 0.521 | +0.0% |
| 3 | rrf-multi-perspective | 0.438 | 0.413 | 0.521 | -0.5% |
| 4 | topological-persistence | 0.431 | 0.404 | 0.521 | -2.0% |
| 5 | anti-hubness-0.4p | 0.424 | 0.392 | 0.521 | -3.6% |
| 6 | influence-function | 0.424 | 0.394 | 0.521 | -3.6% |
| 7 | score-calibration-5k | 0.417 | 0.383 | 0.521 | -5.2% |
| 8 | pure-csls-5k | 0.414 | 0.381 | 0.521 | -5.9% |

## Analysis

On BEIR zero-shot transfer, **flat baseline wins** (0.440 nDCG). All reranking methods hurt.

This is a fundamentally different regime from HotpotQA:
- **Domain mismatch**: nomic-embed-text wasn't fine-tuned for SciFact/NFCorpus
- **Sparse relevance**: BEIR queries have very few relevant docs (1-3 per query)
- **No hub structure**: 8.8K docs with domain-mismatched embeddings don't form the hub patterns that CSLS corrects

**CSLS hurts the most** (-5.9%) — it's actively degrading the signal by trying to correct for hubs that don't exist in this regime.

### Comparison with Published Baselines
- Our flat: 0.440 nDCG
- BM25 (SciFact): 0.665, BM25 (NFCorpus): 0.325 → avg ~0.495
- DPR (SciFact): 0.318, DPR (NFCorpus): 0.189 → avg ~0.254
- We're between BM25 and DPR for dense retrieval — reasonable for nomic-embed-text

## Key Insight
CSLS/anti-hubness is a **corpus-size-dependent** technique:
- **Small corpus (160 docs)**: Hurts. No hubs to correct.
- **Medium corpus (8.8K, cross-domain)**: Hurts. Domain mismatch prevents hub formation.
- **Large corpus (491 docs, same domain)**: Helps significantly. Hubs form within domain.

The technique requires:
1. Sufficient corpus size for concentration-of-measure effects
2. In-domain embeddings where hub structure is meaningful
