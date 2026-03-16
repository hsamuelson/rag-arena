#!/usr/bin/env python3
"""SOTA Retrieval Strategy Benchmark — Equal Footing Comparison.

Benchmarks 12 retrieval strategies on FiQA (200 queries, nDCG@10),
all using the same nomic-embed-text + BM25 hybrid backend with top-50
candidate retrieval. The only variable is the reranking strategy.

Fairness rules:
  - Same retriever: nomic-embed-text (768-dim) + BM25 hybrid, RRF fusion
  - Same benchmark: FiQA 57K docs, 200 queries, nDCG@10
  - Same seed: 42 for reproducibility
  - Rerankers are the variable under test

Strategies (3 tiers):
  Tier 1 — Drop-in rerankers:
    1. Flat baseline (no reranking)
    2. CE L-6 (default, top-10)
    3. CE L-12 (top-50 pool)
    4. BGE-v2-m3 (top-50 pool)
    5. Mxbai-rerank-base-v2 (top-50 pool)

  Tier 2 — Strategy innovations:
    6. MRAM-v1.5 (sentence retrieval + CE L-12)
    7. MRAM + BGE reranker
    8. Late interaction (MaxSim reranking)
    9. Late interaction + MRAM

  Tier 3 — LLM & Ensemble:
    10. Multi-reranker ensemble (RRF of CE-L12 + BGE + MaxSim)
    11. LLM listwise (qwen3.5:122b, top-20)
    12. LLM pointwise (qwen3.5:122b, top-20)

Published BEIR FiQA baselines for context:
  - BM25:      0.236 nDCG@10
  - DPR:       0.112 nDCG@10
  - E5-large:  0.370 nDCG@10
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.hybrid_backend import HybridBackend
from arena.benchmarks.beir_subset import BEIRSubsetBenchmark
from arena.metrics.scoring import ndcg_at_k, compute_scorecard

# Published BEIR baselines for FiQA (nDCG@10)
BEIR_BASELINES = {
    "BM25": 0.236,
    "DPR": 0.112,
    "E5-large": 0.370,
}


def compute_per_query_ndcg(per_question_results, k=10):
    ndcgs = []
    for q in per_question_results:
        retrieved = q["retrieved_ids"][:k]
        relevant = q["relevant_ids"]
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    return np.array(ndcgs)


def bootstrap_ci(scores, n_bootstrap=2000, ci=0.95):
    rng = np.random.RandomState(42)
    means = sorted(scores[rng.randint(0, len(scores), size=len(scores))].mean()
                   for _ in range(n_bootstrap))
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def paired_wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    try:
        from scipy.stats import wilcoxon
        diffs = scores_a - scores_b
        # Remove zeros (ties)
        diffs = diffs[diffs != 0]
        if len(diffs) < 10:
            return 1.0
        stat, p_val = wilcoxon(diffs)
        return float(p_val)
    except ImportError:
        # Fallback to permutation test
        return paired_permutation_test(scores_a, scores_b)


def paired_permutation_test(scores_a, scores_b, n_perms=5000):
    rng = np.random.RandomState(42)
    observed = scores_a.mean() - scores_b.mean()
    diffs = scores_a - scores_b
    count = sum(1 for _ in range(n_perms)
                if (diffs * rng.choice([-1, 1], size=len(diffs))).mean() >= observed)
    return count / n_perms


def run_hypothesis(name, hyp, samples, backend, config):
    from tqdm import tqdm

    per_question = []
    iterator = tqdm(samples, desc=name)

    for sample in iterator:
        t0 = time.time()
        results, embeddings = backend.retrieve_with_embeddings(sample.question, config.top_k)
        query_emb = None
        try:
            query_emb = backend.embed_query(sample.question)
        except Exception:
            pass

        hyp_result = hyp.apply(sample.question, results, embeddings, query_emb)
        latency_ms = (time.time() - t0) * 1000

        per_question.append({
            "question_id": sample.question_id,
            "question": sample.question,
            "prediction": "",
            "ground_truth": sample.ground_truth,
            "category": sample.category,
            "retrieved_ids": [r.doc_id for r in hyp_result.results],
            "relevant_ids": sample.corpus_doc_ids,
            "latency_ms": latency_ms,
            "hypothesis_metadata": hyp_result.metadata,
        })

    scorecard = compute_scorecard(per_question, k=config.top_k)
    return per_question, scorecard


def build_hypotheses(backend):
    """Build all 12 strategies organized by tier."""
    from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
    from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
    from arena.hypotheses.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis
    from arena.hypotheses.bge_reranker import BGERerankerHypothesis
    from arena.hypotheses.mxbai_reranker import MxbaiRerankerHypothesis
    from arena.hypotheses.multi_resolution import MultiResolutionHypothesis
    from arena.hypotheses.mram_bge_reranker import MRAMBGERerankerHypothesis
    from arena.hypotheses.late_interaction_reranker import LateInteractionRerankerHypothesis
    from arena.hypotheses.late_interaction_mram import LateInteractionMRAMHypothesis
    from arena.hypotheses.multi_reranker_ensemble import MultiRerankerEnsembleHypothesis
    from arena.hypotheses.llm_listwise_reranker import LLMListwiseRerankerHypothesis
    from arena.hypotheses.llm_pointwise_reranker import LLMPointwiseRerankerHypothesis

    tiers = {
        "Tier 1: Drop-in Rerankers": [],
        "Tier 2: Strategy Innovations": [],
        "Tier 3: LLM & Ensemble": [],
    }

    # --- Tier 1 ---
    flat = FlatBaselineHypothesis()
    tiers["Tier 1: Drop-in Rerankers"].append(("Flat baseline", flat))

    ce_l6 = CrossEncoderRerankerHypothesis()
    tiers["Tier 1: Drop-in Rerankers"].append(("CE L-6 (top-10)", ce_l6))

    ce_l12 = DeepPool50CEL12Hypothesis()
    ce_l12.set_backend(backend)
    tiers["Tier 1: Drop-in Rerankers"].append(("CE L-12 (top-50)", ce_l12))

    bge = BGERerankerHypothesis()
    bge.set_backend(backend)
    tiers["Tier 1: Drop-in Rerankers"].append(("BGE-v2-m3 (top-50)", bge))

    mxbai = MxbaiRerankerHypothesis()
    mxbai.set_backend(backend)
    tiers["Tier 1: Drop-in Rerankers"].append(("Mxbai-rerank (top-50)", mxbai))

    # --- Tier 2 ---
    mram = MultiResolutionHypothesis()
    mram.set_backend(backend)
    tiers["Tier 2: Strategy Innovations"].append(("MRAM-v1.5", mram))

    mram_bge = MRAMBGERerankerHypothesis()
    mram_bge.set_backend(backend)
    tiers["Tier 2: Strategy Innovations"].append(("MRAM + BGE", mram_bge))

    late = LateInteractionRerankerHypothesis()
    late.set_backend(backend)
    tiers["Tier 2: Strategy Innovations"].append(("Late interaction", late))

    late_mram = LateInteractionMRAMHypothesis()
    late_mram.set_backend(backend)
    tiers["Tier 2: Strategy Innovations"].append(("Late interaction + MRAM", late_mram))

    # --- Tier 3 ---
    ensemble = MultiRerankerEnsembleHypothesis()
    ensemble.set_backend(backend)
    tiers["Tier 3: LLM & Ensemble"].append(("Multi-reranker ensemble", ensemble))

    llm_list = LLMListwiseRerankerHypothesis()
    llm_list.set_backend(backend)
    tiers["Tier 3: LLM & Ensemble"].append(("LLM listwise", llm_list))

    llm_point = LLMPointwiseRerankerHypothesis()
    llm_point.set_backend(backend)
    tiers["Tier 3: LLM & Ensemble"].append(("LLM pointwise", llm_point))

    return tiers


def print_tier_results(tier_name, tier_results, baseline_ndcgs):
    """Print comparison table for a tier."""
    print(f"\n{'─' * 70}")
    print(f"  {tier_name}")
    print(f"{'─' * 70}")
    print(f"  {'Strategy':<30s}  {'nDCG@10':>8s}  {'vs CE-L6':>8s}  {'95% CI':>18s}  {'p-val':>8s}")
    print(f"  {'─' * 76}")

    for name, data in tier_results.items():
        ndcg = data["mean_ndcg"]
        ci = bootstrap_ci(data["ndcgs"])
        if baseline_ndcgs is not None:
            diff_pct = (ndcg - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            if ndcg >= baseline_ndcgs.mean():
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {name:<30s}  {ndcg:8.4f}  {diff_pct:+7.1f}%  [{ci[0]:.4f}, {ci[1]:.4f}]  {p_val:.4f} {sig}")
        else:
            print(f"  {name:<30s}  {ndcg:8.4f}  {'—':>8s}  [{ci[0]:.4f}, {ci[1]:.4f}]")


def generate_report(all_results, output_dir):
    """Generate markdown report with full results."""
    lines = [
        "# SOTA Retrieval Strategy Benchmark — Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Benchmark:** FiQA (200 queries, nDCG@10)",
        f"**Backend:** nomic-embed-text (768-dim) + BM25 hybrid, RRF fusion, top-50 candidates",
        "",
        "## Published BEIR Baselines",
        "",
        "| Method | nDCG@10 |",
        "|--------|---------|",
    ]
    for name, score in BEIR_BASELINES.items():
        lines.append(f"| {name} | {score:.3f} |")

    lines.extend(["", "## Results", ""])
    lines.append("| # | Strategy | nDCG@10 | vs CE-L6 | 95% CI | p-value | Sig |")
    lines.append("|---|----------|---------|----------|--------|---------|-----|")

    baseline_ndcg = all_results.get("CE L-6 (top-10)", {}).get("mean_ndcg", 0)
    baseline_ndcgs = all_results.get("CE L-6 (top-10)", {}).get("ndcgs")

    for i, (name, data) in enumerate(all_results.items(), 1):
        ndcg = data["mean_ndcg"]
        ci = bootstrap_ci(data["ndcgs"])
        if baseline_ndcgs is not None and name != "Flat baseline":
            diff_pct = (ndcg - baseline_ndcg) / baseline_ndcg * 100
            if ndcg >= baseline_ndcg:
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(
                f"| {i} | {name} | {ndcg:.4f} | {diff_pct:+.1f}% | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {p_val:.4f} | {sig} |"
            )
        else:
            lines.append(f"| {i} | {name} | {ndcg:.4f} | — | [{ci[0]:.4f}, {ci[1]:.4f}] | — | — |")

    # Holm-Bonferroni correction
    lines.extend(["", "## Statistical Notes", ""])
    lines.append("- **Test:** Paired Wilcoxon signed-rank test vs CE L-6 baseline")
    lines.append("- **Correction:** Holm-Bonferroni for 11 comparisons (threshold starts at 0.05/11 = 0.0045)")
    lines.append("- **CI:** Bootstrap 95% confidence interval (2000 resamples, seed=42)")

    # Holm-Bonferroni analysis
    if baseline_ndcgs is not None:
        p_values = []
        for name, data in all_results.items():
            if name in ("Flat baseline", "CE L-6 (top-10)"):
                continue
            ndcg = data["mean_ndcg"]
            if ndcg >= baseline_ndcg:
                p = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            p_values.append((name, p, data["mean_ndcg"]))

        p_values.sort(key=lambda x: x[1])
        lines.extend(["", "### Holm-Bonferroni Corrected Results", ""])
        lines.append("| Rank | Strategy | p-value | Threshold | Significant? |")
        lines.append("|------|----------|---------|-----------|-------------|")
        n_tests = len(p_values)
        for rank, (name, p, ndcg) in enumerate(p_values):
            threshold = 0.05 / (n_tests - rank)
            sig = "Yes" if p < threshold else "No"
            lines.append(f"| {rank+1} | {name} | {p:.6f} | {threshold:.6f} | {sig} |")

    # Top 3 strategies
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_ndcg"], reverse=True)
    lines.extend(["", "## Top 3 Strategies", ""])
    for i, (name, data) in enumerate(sorted_results[:3], 1):
        lines.append(f"{i}. **{name}**: {data['mean_ndcg']:.4f} nDCG@10")

    # Composability check
    lines.extend(["", "## Composability Analysis", ""])
    mram_score = all_results.get("MRAM-v1.5", {}).get("mean_ndcg", 0)
    bge_score = all_results.get("BGE-v2-m3 (top-50)", {}).get("mean_ndcg", 0)
    mram_bge_score = all_results.get("MRAM + BGE", {}).get("mean_ndcg", 0)
    if mram_score and bge_score and mram_bge_score:
        if mram_bge_score > max(mram_score, bge_score):
            lines.append(f"- MRAM + BGE ({mram_bge_score:.4f}) > max(MRAM={mram_score:.4f}, BGE={bge_score:.4f}) -> **Composable!**")
        else:
            lines.append(f"- MRAM + BGE ({mram_bge_score:.4f}) vs max(MRAM={mram_score:.4f}, BGE={bge_score:.4f}) -> Not composable")

    late_score = all_results.get("Late interaction", {}).get("mean_ndcg", 0)
    ce_l12_score = all_results.get("CE L-12 (top-50)", {}).get("mean_ndcg", 0)
    if late_score and ce_l12_score:
        if late_score > ce_l12_score:
            lines.append(f"- Late interaction ({late_score:.4f}) > CE L-12 ({ce_l12_score:.4f}) -> **ColBERT-style wins!**")
        else:
            lines.append(f"- Late interaction ({late_score:.4f}) vs CE L-12 ({ce_l12_score:.4f}) -> CE reranking still better")

    report = "\n".join(lines)
    with open(output_dir / "sota_benchmark_report.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_dir}/sota_benchmark_report.md")


def main():
    np.random.seed(42)

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 200

    print("=" * 70)
    print("SOTA RETRIEVAL STRATEGY BENCHMARK — EQUAL FOOTING")
    print("=" * 70)
    print(f"Backend: nomic-embed-text + BM25 hybrid, RRF fusion")
    print(f"Benchmark: FiQA, {max_samples} queries, nDCG@10")
    print(f"Seed: 42")

    # Load FiQA
    print("\n### Loading FiQA...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # Ingest
    print(f"\n### Ingesting {len(corpus)} documents...")
    backend = HybridBackend(config)
    t0 = time.time()
    backend.ingest(corpus)
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    # Build all strategies
    tiers = build_hypotheses(backend)

    all_results = {}  # ordered dict of name -> data
    baseline_ndcgs = None  # CE L-6 baseline for comparison

    # Run tier by tier
    for tier_name, strategies in tiers.items():
        print(f"\n{'#' * 70}")
        print(f"  {tier_name}")
        print(f"{'#' * 70}")

        tier_results = {}

        for name, hyp in strategies:
            print(f"\n{'=' * 60}")
            print(f"  Running: {name}")
            print(f"{'=' * 60}")

            start = time.time()
            try:
                per_question, scorecard = run_hypothesis(name, hyp, samples, backend, config)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
            elapsed = time.time() - start
            ndcgs = compute_per_query_ndcg(per_question)

            data = {
                "per_question": per_question,
                "scorecard": scorecard,
                "ndcgs": ndcgs,
                "mean_ndcg": float(ndcgs.mean()),
                "elapsed": elapsed,
            }
            all_results[name] = data
            tier_results[name] = data

            print(f"  nDCG@10={ndcgs.mean():.4f}  R@K={scorecard.recall_at_k:.3f}  "
                  f"MRR={scorecard.mrr:.3f}  ({elapsed:.1f}s)")

            # Track baseline
            if name == "CE L-6 (top-10)":
                baseline_ndcgs = ndcgs

            # Save incrementally
            raw = {n: d["ndcgs"].tolist() for n, d in all_results.items()}
            with open(output_dir / "sota_benchmark_raw.json", "w") as f:
                json.dump(raw, f, indent=2)

        # Print tier summary
        print_tier_results(tier_name, tier_results, baseline_ndcgs)

    # ─── Final Summary ───
    print(f"\n{'#' * 70}")
    print("FINAL RESULTS — ALL STRATEGIES")
    print(f"{'#' * 70}")

    print(f"\n  Published baselines (FiQA nDCG@10):")
    for name, score in BEIR_BASELINES.items():
        print(f"    {name:20s}: {score:.4f}")

    print(f"\n  Our results:")
    for name, data in all_results.items():
        ndcg = data["mean_ndcg"]
        vs_e5 = (ndcg - BEIR_BASELINES["E5-large"]) / BEIR_BASELINES["E5-large"] * 100
        print(f"    {name:<30s}: {ndcg:.4f}  ({vs_e5:+.1f}% vs E5-large, {data['elapsed']:.0f}s)")

    # Statistical tests vs CE L-6 with Holm-Bonferroni
    if baseline_ndcgs is not None:
        print(f"\n  Statistical comparisons vs CE L-6 (Holm-Bonferroni corrected):")
        p_values = []
        for name, data in all_results.items():
            if name in ("Flat baseline", "CE L-6 (top-10)"):
                continue
            diff = (data["mean_ndcg"] - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            if data["mean_ndcg"] >= baseline_ndcgs.mean():
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            ci = bootstrap_ci(data["ndcgs"])
            p_values.append((name, diff, p_val, ci))

        # Sort by p-value for Holm-Bonferroni
        p_values.sort(key=lambda x: x[2])
        n_tests = len(p_values)
        for rank, (name, diff, p_val, ci) in enumerate(p_values):
            threshold = 0.05 / (n_tests - rank)
            sig = "***" if p_val < threshold / 10 else "**" if p_val < threshold / 2 else "*" if p_val < threshold else "ns"
            print(f"    {name:<30s}: {diff:+.1f}% (p={p_val:.4f} {sig}), 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Top 3
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_ndcg"], reverse=True)
    print(f"\n  Top 3 strategies (generation-0 candidates for random walk):")
    for i, (name, data) in enumerate(sorted_results[:3], 1):
        print(f"    {i}. {name}: {data['mean_ndcg']:.4f}")

    # Save final results
    raw = {name: data["ndcgs"].tolist() for name, data in all_results.items()}
    with open(output_dir / "sota_benchmark_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    summary = {
        "date": datetime.now().isoformat(),
        "benchmark": "FiQA",
        "n_queries": max_samples,
        "seed": 42,
        "backend": "nomic-embed-text + BM25 hybrid, RRF fusion, top-50",
        "beir_baselines": BEIR_BASELINES,
        "results": {
            name: {
                "ndcg_at_10": data["mean_ndcg"],
                "elapsed_s": data["elapsed"],
                "vs_e5_large_pct": (data["mean_ndcg"] - BEIR_BASELINES["E5-large"]) / BEIR_BASELINES["E5-large"] * 100,
            }
            for name, data in all_results.items()
        },
    }
    with open(output_dir / "sota_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    generate_report(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}/sota_benchmark_*.json")


if __name__ == "__main__":
    main()
