# Experiment Results: Synthetic Benchmark (Small Corpus)

**Date:** 2026-03-09
**Benchmark:** Synthetic Multi-hop (160 docs, 60 samples, 1/2/3-hop)
**Backend:** direct-embeddings (nomic-embed-text 768d)

## Rankings (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | MRR | R@K |
|------|-----------|---------|-----|-----|
| 1 | flat-baseline | 0.727 | 0.648 | 1.000 |
| 2 | embedding-triangulation | 0.727 | 0.650 | 1.000 |
| 3 | rrf-multi-perspective | 0.725 | 0.655 | 1.000 |
| 4 | mahalanobis-retrieval | 0.725 | 0.639 | 1.000 |
| 5 | anti-hubness-0.4p | 0.710 | 0.653 | 1.000 |
| 6 | topological-persistence | 0.709 | 0.631 | 1.000 |
| 7 | influence-function | 0.708 | 0.649 | 1.000 |
| 8 | score-calibration-5k | 0.687 | 0.618 | 1.000 |

## Analysis

At 160 docs (well below embedding capacity ceiling):
- **Perfect recall** (R@K=1.000) for all methods — the corpus is small enough that all relevant docs are retrieved
- **Flat baseline wins** (0.727) — reranking adds noise when the initial ranking is already good
- **Anti-hubness HURTS** (-2.3%) — at 160 docs there are no real hubs to correct for
- **Score calibration hurts more** (-5.5%) — local density is uniform in a small corpus

**Key insight:** CSLS/anti-hubness is a scaling correction. It helps exactly when the corpus is large enough for concentration-of-measure effects to create hub documents. On small corpora, the original cosine ranking is hard to beat.
