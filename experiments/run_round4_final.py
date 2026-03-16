#!/usr/bin/env python3
"""Round 4: Final push — ensemble, multi-hop, and combined best approaches.

Target to beat: CE + Title Boost = 0.844 nDCG@10
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.hybrid_backend import HybridBackend
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.ce_title_boost import CETitleBoostHypothesis
from arena.hypotheses.ce_ensemble import CEEnsembleHypothesis
from arena.hypotheses.ce_multihop_iterative import CEMultihopIterativeHypothesis
from arena.hypotheses.ce_title_multiwindow import CETitleMultiWindowHypothesis
from arena.runners.arena_runner import ArenaRunner


def run_hypothesis(name, backend, hypothesis, benchmark, runner, max_samples):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    start = time.time()
    results = runner.run_arena(
        benchmark=benchmark, backend=backend, hypotheses=[hypothesis],
        max_samples=max_samples, skip_llm=True, verbose=True,
    )
    print(f"  Completed in {time.time() - start:.1f}s")
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

    # Baselines
    r = run_hypothesis("CE baseline", hybrid, CrossEncoderRerankerHypothesis(), benchmark, runner, max_samples)
    all_results.extend(r)

    r = run_hypothesis("CE + Title Boost (current best)", hybrid, CETitleBoostHypothesis(title_weight=0.2), benchmark, runner, max_samples)
    all_results.extend(r)

    # New approaches
    hypotheses = [
        ("CE Ensemble (RRF, 2 models)", CEEnsembleHypothesis(fusion="rrf")),
        ("CE Ensemble (avg, 2 models)", CEEnsembleHypothesis(fusion="avg")),
        ("CE Multi-hop Iterative", CEMultihopIterativeHypothesis()),
        ("CE Multi-hop (bridge=0.25)", CEMultihopIterativeHypothesis(bridge_weight=0.25)),
        ("CE Title+MultiWindow (0.2)", CETitleMultiWindowHypothesis(title_weight=0.2)),
        ("CE Title+MultiWindow (0.15)", CETitleMultiWindowHypothesis(title_weight=0.15)),
        ("CE Title Boost (0.25)", CETitleBoostHypothesis(title_weight=0.25)),
        ("CE Title Boost (0.3)", CETitleBoostHypothesis(title_weight=0.3)),
    ]

    for name, hyp in hypotheses:
        r = run_hypothesis(name, hybrid, hyp, benchmark, runner, max_samples)
        all_results.extend(r)

    # Grand comparison
    print(f"\n{'#'*60}")
    print("ROUND 4: FINAL RESULTS")
    print(f"{'#'*60}")
    runner.print_comparison(all_results)
    runner.save_results(all_results, output_dir / "round4_final.json")

    # Markdown
    sorted_results = sorted(all_results, key=lambda r: r.scorecard.ndcg_at_k, reverse=True)
    best = sorted_results[0].scorecard.ndcg_at_k

    lines = [
        "# Round 4: Final Push",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Benchmark:** HotpotQA (multi-hop, {max_samples} samples, 491 docs)",
        f"**Backend:** Hybrid BM25+Dense",
        "",
        "## Rankings",
        "",
        "| Rank | Hypothesis | nDCG@10 | MRR | R@K | Latency |",
        "|------|-----------|---------|-----|-----|---------|",
    ]
    for rank, r in enumerate(sorted_results, 1):
        sc = r.scorecard
        lines.append(f"| {rank} | {r.hypothesis_name} | {sc.ndcg_at_k:.4f} | {sc.mrr:.4f} | {sc.recall_at_k:.4f} | {sc.latency_ms:.0f}ms |")

    with open(output_dir / "round4_final.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_dir}/round4_final.*")


if __name__ == "__main__":
    main()
