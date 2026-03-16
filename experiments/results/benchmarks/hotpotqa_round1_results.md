# Experiment Results: HotpotQA Round 1

**Date:** 2026-03-09
**Benchmark:** HotpotQA (multi-hop retrieval, 491 docs ingested, 50 samples)
**Backend:** direct-embeddings (nomic-embed-text 768d via Ollama)
**Metric:** Retrieval-only (skip-LLM), nDCG@10 primary metric

## Rankings (sorted by nDCG@10)

| Rank | Hypothesis | nDCG@10 | MRR | R@K | Type | vs Flat |
|------|-----------|---------|-----|-----|------|---------|
| 1 | **anti-hubness-0.4p** | **0.613** | **0.691** | 0.730 | Original | **+4.8%** |
| 2 | topological-persistence-30.0r | 0.595 | 0.643 | 0.730 | Novel #12 | +1.7% |
| 3 | influence-function | 0.594 | 0.654 | 0.730 | Novel #15 | +1.5% |
| 4 | score-calibration-5k | 0.594 | 0.650 | 0.730 | Original | +1.5% |
| 5 | rrf-multi-perspective | 0.593 | 0.653 | 0.730 | Memory Baseline | +1.4% |
| 6 | embedding-triangulation | 0.586 | 0.644 | 0.730 | Novel #20 | +0.2% |
| 7 | **flat-baseline** | **0.585** | **0.641** | 0.730 | Control | baseline |
| 8 | optimal-transport-0.1e | 0.585 | 0.641 | 0.730 | Novel #10 | +0.0% |
| 9 | anchor-expansion | 0.585 | 0.641 | 0.730 | Novel #18 | +0.0% |
| 10 | cross-encoder-proxy | 0.584 | 0.640 | 0.730 | Novel #4 | -0.2% |
| 11 | hierarchical-cluster | 0.584 | 0.643 | 0.730 | Memory Baseline | -0.2% |
| 12 | isotropy-enhancement | 0.582 | 0.639 | 0.730 | Novel #1 | -0.5% |
| 13 | mahalanobis-retrieval | 0.602 | 0.675 | 0.730 | Novel #5 | +2.9% |
| 14 | query-drift-a0.6 | 0.577 | 0.616 | 0.730 | Novel #2 | -1.4% |
| 15 | leverage-scores-0.6a | 0.572 | 0.619 | 0.730 | Novel #9 | -2.2% |
| 16 | graph-community | 0.571 | 0.634 | 0.730 | Memory Baseline | -2.4% |
| 17 | kernel-herding-autos | 0.571 | 0.634 | 0.730 | Novel #13 | -2.4% |
| 18 | dpp-selection-0.7w | 0.568 | 0.634 | 0.730 | Original | -2.9% |
| 19 | mean-bias-correction | 0.564 | 0.596 | 0.730 | Original | -3.6% |
| 20 | random-projection-ensemble | 0.561 | 0.590 | 0.730 | Novel #16 | -4.1% |
| 21 | curvature-aware | 0.558 | 0.585 | 0.730 | Novel #17 | -4.6% |
| 22 | info-bottleneck-1.0b | 0.556 | 0.620 | 0.730 | Novel #7 | -5.0% |
| 23 | capacity-partition-5ps | 0.555 | 0.612 | 0.730 | Original | -5.1% |
| 24 | submodular-coverage-0.3q | 0.553 | 0.616 | 0.730 | Novel #11 | -5.5% |
| 25 | mutual-information | 0.549 | 0.586 | 0.730 | Novel #19 | -6.2% |
| 26 | variance-reduction | 0.507 | 0.552 | 0.730 | Novel #14 | -13.3% |
| 27 | adaptive-context-window | 0.474 | 0.621 | 0.490 | Novel #3 | -19.0% |
| 28 | density-peaks-3p | 0.461 | 0.430 | 0.730 | Novel #6 | -21.2% |
| 29 | spectral-rerank-5b | 0.437 | 0.376 | 0.730 | Novel #8 | -25.3% |

## Key Findings

### Top Performers (beat flat baseline)
1. **Anti-hubness** (0.613) — Clear winner, +4.8% over flat. CSLS correction for hub documents is the single most effective reranking strategy.
2. **Topological persistence** (0.595) — Novel hypothesis! Union-Find filtration scoring works well. Documents creating long-lived connected components are genuinely relevant.
3. **Influence function** (0.594) — Novel hypothesis! Leave-one-out centroid perturbation identifies documents whose removal hurts query alignment.
4. **Score calibration** (0.594) — Z-score normalization by local density continues to be strong.
5. **RRF multi-perspective** (0.593) — Memory retrieval baseline. Multi-view fusion (raw + centered + centroid) provides modest but consistent improvement.
6. **Mahalanobis** (0.602) — Novel hypothesis! Correlation-aware distance metric beats cosine. 3rd best nDCG overall.
7. **Embedding triangulation** (0.586) — Novel hypothesis, marginal improvement.

### Memory Retrieval Baselines Performance
- **RRF Multi-Perspective**: 0.593 nDCG (+1.4%) — Best memory baseline, competitive
- **Hierarchical Cluster (RAPTOR-style)**: 0.584 nDCG (-0.2%) — Close to flat
- **Graph Community (GraphRAG-style)**: 0.571 nDCG (-2.4%) — Diversity hurts precision

### Hypotheses That Hurt Performance
- **Spectral reranking** (-25.3%) — Fiedler vector sampling destroys relevance ordering
- **Density peaks** (-21.2%) — Selecting cluster centers != selecting relevant docs
- **Adaptive context window** (-19.0%) — Reducing result count (R@K=0.490) hurts recall badly
- **Variance reduction** (-13.3%) — Control variate approach too aggressive

### Observations
1. **Recall is identical (0.730)** for all except adaptive-context-window — reranking doesn't change recall
2. **Anti-hubness dominates** — hub correction is the key insight for embedding retrieval
3. **Novel hypotheses show promise**: topological persistence and influence function are new ideas that beat established methods
4. **Diversity-focused methods hurt**: DPP, graph-community, spectral-rerank — diversity trades off against relevance
5. **Mahalanobis is expensive but effective** — 8.4s/query but strong results

## Next Steps
- Run Round 2 with tuned hyperparameters on top performers
- Combine best ideas (anti-hubness + influence function + topological persistence)
- Test on synthetic benchmark for small-corpus behavior
- Create hybrid hypotheses combining top techniques
