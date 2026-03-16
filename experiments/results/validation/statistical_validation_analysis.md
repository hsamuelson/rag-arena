# Statistical Validation: Honest Assessment

**Date:** 2026-03-09
**Critical finding: Most CE tweaks are noise, not signal.**

## What's Real vs Noise

### REAL improvements (statistically significant at p<0.05)
| Technique | nDCG@10 (n=200) | vs Baseline | p-value | Effect |
|-----------|----------------|------------|---------|--------|
| Hybrid > Dense retrieval | 0.660 vs ~0.52* | +25%+ | <0.0001 | Large |
| Cross-encoder > Flat ranking | 0.787 vs 0.660 | +19.2% | <0.0001 | Medium-Large |

### NOISE (NOT statistically significant)
| Technique | nDCG@10 (n=200) | vs CE | p-value | Cohen's d |
|-----------|----------------|-------|---------|-----------|
| CE + Title Boost | 0.786 | -0.1% | 0.59 | -0.015 |
| CE Title+MultiWindow | 0.790 | +0.3% | 0.29 | 0.041 |
| CE Multi-window max | 0.789 | +0.2% | 0.25 | 0.048 |

## Why n=50 Was Misleading

At n=50 (our original experiments), random variation in sample selection created apparent
differences of 1-3%. With the same methods at n=200:
- Apparent gains shrink from +2.5% to +0.3%
- p-values are far from significant (0.25-0.59)
- Cohen's d is negligible (<0.05)
- The 95% confidence intervals completely overlap

**Lesson:** For nDCG evaluation, n=50 is insufficient to detect real differences smaller than ~5%.
We need n=200+ AND p<0.05 to trust any claimed improvement.

## What This Means for Next Steps

To genuinely improve over plain CE (0.787 nDCG on our benchmark), we need approaches that are
**fundamentally different**, not scoring tweaks:

1. **Improve first-stage recall** — hybrid already gives 0.890, but missing 11% of relevant docs
   caps all downstream performance. Need recall > 0.95.

2. **Multi-hop reasoning** — HotpotQA requires info from 2+ documents. Iterative retrieval
   (retrieve → read → retrieve again) is the only way to find both bridge and answer docs.

3. **Larger reranking models** — ms-marco-MiniLM-L-6-v2 is 22M params. DeBERTa-v3-base (184M)
   or LLM-based rerankers (7B+) could provide meaningful accuracy gains.

4. **Document expansion** — doc2query or contextual retrieval (add LLM-generated context before
   embedding) improves recall at indexing time.

5. **Query decomposition** — break multi-hop questions into sub-questions, retrieve independently,
   merge. This directly addresses HotpotQA's multi-hop nature.

## Statistical Requirements Going Forward

All future experiments must:
- Use n >= 200 samples
- Report 95% bootstrap CIs
- Report paired permutation test p-values vs CE baseline
- Only claim improvement if p < 0.05 AND effect size > 0.2 (small)
