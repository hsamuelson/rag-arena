#!/usr/bin/env python3
"""Beat the gold standard: novel CE-based hypotheses vs Hybrid+CE baseline.

Tests 9 novel approaches against the hybrid+cross-encoder baseline (0.835 nDCG@10).
All use the hybrid BM25+Dense backend for maximum first-stage recall.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.hybrid_backend import HybridBackend
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.ce_score_fusion import CEScoreFusionHypothesis
from arena.hypotheses.ce_iterative import CEIterativeHypothesis
from arena.hypotheses.ce_segmented import CESegmentedHypothesis
from arena.hypotheses.ce_pairwise import CEPairwiseHypothesis
from arena.hypotheses.ce_calibrated import CECalibratedHypothesis
from arena.hypotheses.ce_diversity_bonus import CEDiversityBonusHypothesis
from arena.hypotheses.ce_reciprocal_neighbor import CEReciprocalNeighborHypothesis
from arena.hypotheses.ce_multi_window import CEMultiWindowHypothesis
from arena.hypotheses.ce_title_boost import CETitleBoostHypothesis
from arena.runners.arena_runner import ArenaRunner


def run_hypothesis(name, backend, hypothesis, benchmark, runner, max_samples):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    start = time.time()
    results = runner.run_arena(
        benchmark=benchmark,
        backend=backend,
        hypotheses=[hypothesis],
        max_samples=max_samples,
        skip_llm=True,
        verbose=True,
    )
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    return results


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 50
    benchmark = HotpotQABenchmark(max_samples=max_samples)
    runner = ArenaRunner(config)
    hybrid = HybridBackend(config)

    all_results = []

    # --- Baselines ---
    print("\n" + "#" * 60)
    print("BASELINES")
    print("#" * 60)

    r = run_hypothesis("Hybrid + Flat (control)", hybrid, FlatBaselineHypothesis(), benchmark, runner, max_samples)
    all_results.extend(r)

    r = run_hypothesis("Hybrid + Cross-encoder (gold standard)", hybrid, CrossEncoderRerankerHypothesis(), benchmark, runner, max_samples)
    all_results.extend(r)

    # --- Novel CE hypotheses ---
    print("\n" + "#" * 60)
    print("NOVEL CE-BASED HYPOTHESES")
    print("#" * 60)

    novel_hypotheses = [
        ("CE + Score Fusion (alpha=0.7)", CEScoreFusionHypothesis(alpha=0.7)),
        ("CE + Score Fusion (alpha=0.85)", CEScoreFusionHypothesis(alpha=0.85)),
        ("CE Iterative PRF", CEIterativeHypothesis()),
        ("CE Segmented (max)", CESegmentedHypothesis(aggregation="max")),
        ("CE Pairwise (2-pass)", CEPairwiseHypothesis(n_passes=2)),
        ("CE Calibrated (t=1.0)", CECalibratedHypothesis(temperature=1.0)),
        ("CE + Diversity (lambda=0.85)", CEDiversityBonusHypothesis(lambda_param=0.85)),
        ("CE + Diversity (lambda=0.95)", CEDiversityBonusHypothesis(lambda_param=0.95)),
        ("CE + Reciprocal NN", CEReciprocalNeighborHypothesis()),
        ("CE Multi-window (max)", CEMultiWindowHypothesis(aggregation="max")),
        ("CE Multi-window (weighted)", CEMultiWindowHypothesis(aggregation="weighted")),
        ("CE + Title Boost (0.2)", CETitleBoostHypothesis(title_weight=0.2)),
        ("CE + Title Boost (0.1)", CETitleBoostHypothesis(title_weight=0.1)),
    ]

    for name, hyp in novel_hypotheses:
        r = run_hypothesis(name, hybrid, hyp, benchmark, runner, max_samples)
        all_results.extend(r)

    # --- Grand comparison ---
    print(f"\n{'#'*60}")
    print("ROUND 3: BEAT THE GOLD STANDARD")
    print(f"{'#'*60}")
    runner.print_comparison(all_results)

    # Save
    runner.save_results(all_results, output_dir / "round3_beat_gold_standard.json")

    # Markdown report
    sorted_results = sorted(all_results, key=lambda r: r.scorecard.ndcg_at_k, reverse=True)
    gold_ndcg = None
    for r in sorted_results:
        if r.hypothesis_name == "cross-encoder":
            gold_ndcg = r.scorecard.ndcg_at_k
            break

    lines = [
        "# Round 3: Beat the Gold Standard",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Benchmark:** HotpotQA (multi-hop, {max_samples} samples, 491 docs)",
        f"**Backend:** Hybrid BM25+Dense (RRF fusion)",
        f"**Gold Standard:** Hybrid + Cross-encoder = {gold_ndcg:.4f} nDCG@10" if gold_ndcg else "",
        "",
        "## Rankings (sorted by nDCG@10)",
        "",
        "| Rank | Hypothesis | nDCG@10 | MRR | R@K | Latency | vs Gold |",
        "|------|-----------|---------|-----|-----|---------|---------|",
    ]
    for rank, r in enumerate(sorted_results, 1):
        sc = r.scorecard
        vs_gold = f"{((sc.ndcg_at_k / gold_ndcg) - 1) * 100:+.1f}%" if gold_ndcg else "N/A"
        marker = " **" if sc.ndcg_at_k > (gold_ndcg or 0) else ""
        lines.append(
            f"| {rank} | {r.hypothesis_name}{marker} | "
            f"{sc.ndcg_at_k:.4f} | {sc.mrr:.4f} | {sc.recall_at_k:.4f} | {sc.latency_ms:.0f}ms | {vs_gold} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "*(To be filled after results)*",
    ])

    with open(output_dir / "round3_beat_gold_standard.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_dir}/round3_beat_gold_standard.*")


if __name__ == "__main__":
    main()
