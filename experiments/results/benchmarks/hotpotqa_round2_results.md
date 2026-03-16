# Experiment Results: HotpotQA Round 2 (Hybrids + Tuning)

**Date:** 2026-03-09
**Benchmark:** HotpotQA (multi-hop, 491 docs, 50 samples)
**Goal:** Beat anti-hubness-0.4p (0.613 nDCG) via hybrids and tuning

## Rankings (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | MRR | vs Flat | vs Round 1 Best |
|------|-----------|---------|-----|---------|-----------------|
| 1 | **pure-csls-5k** | **0.614** | **0.693** | +5.0% | **NEW BEST** (+0.2%) |
| 2 | **anti-hubness-0.6p** | **0.614** | **0.693** | +5.0% | tied |
| 3 | **anti-hubness-0.8p** | **0.614** | **0.693** | +5.0% | tied |
| 4 | anti-hubness-0.4p | 0.613 | 0.691 | +4.8% | Round 1 champ |
| 5 | pure-csls-3k | 0.611 | 0.682 | +4.4% | |
| 6 | pure-csls-7k | 0.611 | 0.688 | +4.4% | |
| 7 | hybrid-csls-topo-calibrated | 0.606 | 0.678 | +3.6% | |
| 8 | hybrid-antihub-influence-0.7w | 0.605 | 0.677 | +3.4% | |
| 9 | hybrid-rrf-top5-30k | 0.596 | 0.654 | +1.9% | |
| 10 | hybrid-rrf-top5-60k | 0.594 | 0.654 | +1.5% | |
| 11 | flat-baseline | 0.585 | 0.641 | baseline | |
| 12 | hybrid-antihub-influence-0.5w | 0.576 | 0.616 | -1.5% | |

## Key Findings

### NEW CHAMPION: Pure CSLS with k=5
- **Pure CSLS (0.614)** marginally beats the blended anti-hubness (0.613)
- The signal IS the ranking: `CSLS(q,d) = 2*cos(q,d) - mean_k(cos(d, NN_k(d)))`
- No blending with original scores needed — CSLS is self-contained
- k=5 is the sweet spot (k=3 and k=7 both slightly worse)

### Anti-hubness tuning plateaus at 0.614
- Increasing penalty from 0.4 to 0.6 or 0.8 gives 0.614 — same as pure CSLS
- At penalty=0.8, it's basically pure CSLS anyway (dominates the blend)
- This confirms: **CSLS is the active ingredient**

### Hybrid fusion is counterproductive
- Every hybrid scores LOWER than pure CSLS
- Adding influence function to CSLS: 0.576 (at 0.5w) — much worse!
- Triple fusion (CSLS + topo + calibration): 0.606 — still worse
- RRF of top 5 signals: 0.594-0.596 — even worse
- **Conclusion: Ensembling dilutes the CSLS signal**

### The lesson
CSLS works because it corrects for neighbourhood density bias. When you add other signals,
you're adding noise. The influence function signal is somewhat correlated with CSLS (both
relate to how a document sits in its neighbourhood), so fusing them adds redundancy, not
complementary information.

## Comparison: Novel vs Memory Baselines vs Flat

| Category | Best Method | nDCG@10 | vs Flat |
|----------|-----------|---------|---------|
| Pure post-retrieval | pure-csls-5k | 0.614 | +5.0% |
| Original hypothesis | anti-hubness-0.4p | 0.613 | +4.8% |
| Novel hypothesis | topological-persistence | 0.595 | +1.7% |
| Memory baseline | rrf-multi-perspective | 0.593 | +1.4% |
| Hybrid | hybrid-csls-topo-calibrated | 0.606 | +3.6% |
| Flat baseline | flat-baseline | 0.585 | 0% |
