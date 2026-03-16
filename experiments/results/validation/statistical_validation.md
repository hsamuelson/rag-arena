# Statistical Validation Results

**Date:** 2026-03-09 16:34
**Samples:** 200 (4x previous experiments)
**Bootstrap:** 1000 resamples, 95% CI
**Permutation test:** 10000 permutations, one-sided

## Results

| Hypothesis | Mean nDCG@10 | 95% CI | vs CE baseline | p-value | Significant? |
|-----------|-------------|--------|---------------|---------|-------------|
| flat-baseline | 0.6604 | [0.6220, 0.6981] | -16.1% | 0.0000 | Yes |
| cross-encoder | 0.7872 | [0.7526, 0.8186] | baseline | - | - |
| ce-title-boost-0.2 | 0.7864 | [0.7517, 0.8178] | -0.1% | 0.5901 | No |
| ce-title-multiwindow-0.15 | 0.7897 | [0.7547, 0.8195] | +0.3% | 0.2863 | No |
| ce-multi-window-max | 0.7889 | [0.7540, 0.8183] | +0.2% | 0.2546 | No |

## Effect Sizes (Cohen's d vs CE baseline)

- **flat-baseline**: d = -0.646 (medium)
- **ce-title-boost-0.2**: d = -0.015 (negligible)
- **ce-title-multiwindow-0.15**: d = 0.041 (negligible)
- **ce-multi-window-max**: d = 0.048 (negligible)

## Per-Query nDCG Distribution

- **flat-baseline**: min=0.000 Q1=0.447 median=0.624 Q3=0.888 max=1.000
- **cross-encoder**: min=0.000 Q1=0.613 median=0.850 Q3=1.000 max=1.000
- **ce-title-boost-0.2**: min=0.000 Q1=0.613 median=0.850 Q3=1.000 max=1.000
- **ce-title-multiwindow-0.15**: min=0.000 Q1=0.613 median=0.850 Q3=1.000 max=1.000
- **ce-multi-window-max**: min=0.000 Q1=0.613 median=0.850 Q3=1.000 max=1.000