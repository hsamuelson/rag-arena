#!/usr/bin/env python3
"""Run all 23 novel hypotheses (3 memory baselines + 20 novel) against HotpotQA.

This is the overnight experiment runner. It:
1. Loads HotpotQA (66K docs, multi-hop retrieval)
2. Runs flat baseline + 3 memory retrieval baselines + 20 novel hypotheses
3. Saves results to experiments/results/ as JSON and Markdown
4. Prints comparison table sorted by nDCG@10
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.direct_embeddings import DirectEmbeddingBackend
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.benchmarks.synthetic_multihop import SyntheticMultiHopBenchmark
from arena.hypotheses import *
from arena.runners.arena_runner import ArenaRunner

# All hypotheses to test
HYPOTHESES = {
    # Control
    "flat": FlatBaselineHypothesis,
    # Memory retrieval baselines (3)
    "rrf-multi": RRFMultiPerspectiveHypothesis,
    "hierarchical": HierarchicalClusterHypothesis,
    "graph-community": GraphCommunityHypothesis,
    # Novel hypotheses (20)
    "isotropy": IsotropyEnhancementHypothesis,
    "query-drift": QueryDriftCorrectionHypothesis,
    "adaptive-window": AdaptiveContextWindowHypothesis,
    "cross-proxy": CrossEncoderProxyHypothesis,
    "mahalanobis": MahalanobisRetrievalHypothesis,
    "density-peak": DensityPeakSelectionHypothesis,
    "info-bottleneck": InformationBottleneckHypothesis,
    "spectral-rerank": SpectralRerankingHypothesis,
    "leverage-score": LeverageScoreSamplingHypothesis,
    "optimal-transport": OptimalTransportHypothesis,
    "submodular": SubmodularCoverageHypothesis,
    "topo-persistence": TopologicalPersistenceHypothesis,
    "kernel-herding": KernelHerdingHypothesis,
    "variance-reduce": VarianceReductionHypothesis,
    "influence-fn": InfluenceFunctionHypothesis,
    "rand-proj-ensemble": RandomProjectionEnsembleHypothesis,
    "curvature": CurvatureAwareHypothesis,
    "anchor-expand": AnchorExpansionHypothesis,
    "mutual-info": MutualInformationHypothesis,
    "triangulation": EmbeddingTriangulationHypothesis,
}

# Also include the best from original scaling set
ORIGINAL_BEST = {
    "capacity-partition": CapacityPartitionHypothesis,
    "anti-hubness": AntiHubnessHypothesis,
    "score-calibration": ScoreCalibrationHypothesis,
    "dpp": DPPSelectionHypothesis,
    "mean-bias": MeanBiasCorrectionHypothesis,
}


def write_markdown_results(results, output_path, benchmark_name, elapsed_secs):
    """Write results to a Markdown file for easy reading."""
    # Sort by nDCG descending
    sorted_results = sorted(results, key=lambda r: r.scorecard.ndcg_at_k, reverse=True)

    lines = [
        f"# Experiment Results: {benchmark_name}",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Duration:** {elapsed_secs:.0f}s ({elapsed_secs/60:.1f}min)",
        f"**Samples per hypothesis:** {sorted_results[0].scorecard.num_samples if sorted_results else 'N/A'}",
        f"",
        f"## Rankings (sorted by nDCG@10)",
        f"",
        f"| Rank | Hypothesis | nDCG@10 | MRR | Recall@K | Precision@K | vs Flat |",
        f"|------|-----------|---------|-----|----------|-------------|---------|",
    ]

    flat_ndcg = None
    for r in sorted_results:
        if "flat" in r.hypothesis_name and "baseline" in r.hypothesis_name.lower():
            flat_ndcg = r.scorecard.ndcg_at_k
            break
    if flat_ndcg is None:
        for r in sorted_results:
            if r.hypothesis_name == "flat-baseline":
                flat_ndcg = r.scorecard.ndcg_at_k
                break

    for rank, r in enumerate(sorted_results, 1):
        sc = r.scorecard
        if flat_ndcg and flat_ndcg > 0:
            delta = ((sc.ndcg_at_k - flat_ndcg) / flat_ndcg) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "baseline"
        lines.append(
            f"| {rank} | {r.hypothesis_name} | {sc.ndcg_at_k:.4f} | "
            f"{sc.mrr:.4f} | {sc.recall_at_k:.4f} | {sc.precision_at_k:.4f} | {delta_str} |"
        )

    # Add category breakdown
    lines.extend([
        "",
        "## Category Breakdown",
        "",
    ])
    for r in sorted_results[:5]:
        sc = r.scorecard
        lines.append(f"### {r.hypothesis_name} (nDCG: {sc.ndcg_at_k:.4f})")
        if sc.category_scores:
            for cat, cat_scores in sc.category_scores.items():
                lines.append(f"  - **{cat}**: recall={cat_scores.get('recall_at_k', 'N/A'):.4f}, n={cat_scores.get('num_samples', '?')}")
        lines.append("")

    # Write top 3 analysis
    lines.extend([
        "## Analysis",
        "",
        f"**Best:** {sorted_results[0].hypothesis_name} (nDCG={sorted_results[0].scorecard.ndcg_at_k:.4f})" if sorted_results else "",
        f"**Worst:** {sorted_results[-1].hypothesis_name} (nDCG={sorted_results[-1].scorecard.ndcg_at_k:.4f})" if sorted_results else "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown results written to {output_path}")


def run_experiment(benchmark_cls, benchmark_name, max_samples, config, hyp_dict, output_dir):
    """Run a single benchmark experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {benchmark_name} ({max_samples} samples)")
    print(f"Hypotheses: {len(hyp_dict)}")
    print(f"{'='*70}")

    backend = DirectEmbeddingBackend(config)

    if benchmark_name == "hotpotqa":
        benchmark = benchmark_cls(max_samples=max_samples)
    else:
        benchmark = benchmark_cls()

    hypotheses = [cls() for cls in hyp_dict.values()]

    runner = ArenaRunner(config)
    start = time.time()

    results = runner.run_arena(
        benchmark=benchmark,
        backend=backend,
        hypotheses=hypotheses,
        max_samples=max_samples,
        skip_llm=True,
        verbose=True,
    )

    elapsed = time.time() - start

    # Save JSON
    json_path = output_dir / f"{benchmark_name}_novel_results.json"
    runner.save_results(results, json_path)
    print(f"\nJSON saved to {json_path}")

    # Save Markdown
    md_path = output_dir / f"{benchmark_name}_novel_results.md"
    write_markdown_results(results, md_path, benchmark_name, elapsed)

    # Print comparison
    runner.print_comparison(results)

    return results


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine novel + best original
    all_hyps = {**HYPOTHESES, **ORIGINAL_BEST}
    print(f"Total hypotheses to test: {len(all_hyps)}")

    all_results = []

    # Run 1: HotpotQA (66K docs — where scaling effects show)
    hotpot_results = run_experiment(
        HotpotQABenchmark, "hotpotqa", 50, config, all_hyps, output_dir
    )
    all_results.extend(hotpot_results)

    # Run 2: Synthetic (small corpus — tests baseline behavior)
    synth_results = run_experiment(
        SyntheticMultiHopBenchmark, "synthetic", None, config, all_hyps, output_dir
    )
    all_results.extend(synth_results)

    # Grand summary
    print(f"\n{'#'*70}")
    print("GRAND SUMMARY (all benchmarks)")
    print(f"{'#'*70}")

    # Write grand markdown
    md_path = output_dir / "grand_summary.md"
    write_markdown_results(all_results, md_path, "All Benchmarks", 0)

    print(f"\nAll results saved to {output_dir}/")
    print("Experiment complete!")


if __name__ == "__main__":
    main()
