#!/usr/bin/env python3
"""Statistical validation: are our improvements real or noise?

Tests:
1. Increase sample size to 200 (4x previous 50)
2. Bootstrap confidence intervals (1000 resamples)
3. Paired permutation test vs cross-encoder baseline
4. Report p-values and 95% CIs for each hypothesis

This answers: "Is the +2.5% improvement statistically significant?"
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
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.ce_title_boost import CETitleBoostHypothesis
from arena.hypotheses.ce_title_multiwindow import CETitleMultiWindowHypothesis
from arena.hypotheses.ce_multi_window import CEMultiWindowHypothesis
from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
from arena.runners.arena_runner import ArenaRunner


def compute_per_query_ndcg(per_question_results, k=10):
    """Compute nDCG for each individual query."""
    from arena.metrics.scoring import ndcg_at_k
    ndcgs = []
    for q in per_question_results:
        retrieved = q["retrieved_ids"][:k]
        relevant = q["relevant_ids"]
        ndcg = ndcg_at_k(retrieved, relevant, k)
        ndcgs.append(ndcg)
    return np.array(ndcgs)


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    rng = np.random.RandomState(42)
    means = []
    n = len(scores)
    for _ in range(n_bootstrap):
        sample = scores[rng.randint(0, n, size=n)]
        means.append(sample.mean())
    means = sorted(means)
    lower = means[int((1 - ci) / 2 * n_bootstrap)]
    upper = means[int((1 + ci) / 2 * n_bootstrap)]
    return lower, upper


def paired_permutation_test(scores_a, scores_b, n_perms=10000):
    """Paired permutation test: is mean(a) > mean(b) significantly?"""
    rng = np.random.RandomState(42)
    observed_diff = scores_a.mean() - scores_b.mean()
    diffs = scores_a - scores_b
    count = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = (diffs * signs).mean()
        if perm_diff >= observed_diff:
            count += 1
    return count / n_perms  # p-value


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 200  # 4x previous for statistical power
    print(f"Running statistical validation with {max_samples} samples...")
    print("This will take a while due to cross-encoder inference.\n")

    benchmark = HotpotQABenchmark(max_samples=max_samples)
    runner = ArenaRunner(config)
    hybrid = HybridBackend(config)

    # Run hypotheses we want to validate
    hypotheses_to_test = [
        ("flat-baseline", FlatBaselineHypothesis()),
        ("cross-encoder", CrossEncoderRerankerHypothesis()),
        ("ce-title-boost-0.2", CETitleBoostHypothesis(title_weight=0.2)),
        ("ce-title-multiwindow-0.15", CETitleMultiWindowHypothesis(title_weight=0.15)),
        ("ce-multi-window-max", CEMultiWindowHypothesis(aggregation="max")),
    ]

    all_experiment_results = {}
    all_per_query_ndcgs = {}

    for name, hyp in hypotheses_to_test:
        print(f"\n{'='*60}")
        print(f"Running: {name} ({max_samples} samples)")
        print(f"{'='*60}")

        start = time.time()
        results = runner.run_arena(
            benchmark=benchmark,
            backend=hybrid,
            hypotheses=[hyp],
            max_samples=max_samples,
            skip_llm=True,
            verbose=True,
        )
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")

        result = results[0]
        all_experiment_results[name] = result

        # Compute per-query nDCG
        ndcgs = compute_per_query_ndcg(result.per_question)
        all_per_query_ndcgs[name] = ndcgs

    # --- Statistical analysis ---
    print(f"\n{'#'*70}")
    print("STATISTICAL VALIDATION")
    print(f"{'#'*70}")

    baseline_name = "cross-encoder"
    baseline_ndcgs = all_per_query_ndcgs[baseline_name]

    lines = [
        "# Statistical Validation Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {max_samples} (4x previous experiments)",
        f"**Bootstrap:** 1000 resamples, 95% CI",
        f"**Permutation test:** 10000 permutations, one-sided",
        "",
        "## Results",
        "",
        "| Hypothesis | Mean nDCG@10 | 95% CI | vs CE baseline | p-value | Significant? |",
        "|-----------|-------------|--------|---------------|---------|-------------|",
    ]

    print(f"\n{'Hypothesis':<30} {'Mean nDCG':>10} {'95% CI':>20} {'vs CE':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 90)

    for name, ndcgs in all_per_query_ndcgs.items():
        mean_ndcg = ndcgs.mean()
        ci_low, ci_high = bootstrap_ci(ndcgs)
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"

        if name == baseline_name:
            diff_str = "baseline"
            p_str = "-"
            sig_str = "-"
        else:
            diff = mean_ndcg - baseline_ndcgs.mean()
            diff_pct = diff / baseline_ndcgs.mean() * 100
            diff_str = f"{diff_pct:+.1f}%"

            if name == "flat-baseline":
                p_val = paired_permutation_test(baseline_ndcgs, ndcgs)
                p_str = f"{p_val:.4f}"
                sig_str = "Yes" if p_val < 0.05 else "No"
            else:
                p_val = paired_permutation_test(ndcgs, baseline_ndcgs)
                p_str = f"{p_val:.4f}"
                sig_str = "Yes" if p_val < 0.05 else "No"

        print(f"{name:<30} {mean_ndcg:>10.4f} {ci_str:>20} {diff_str:>8} {p_str:>10} {sig_str:>6}")
        lines.append(f"| {name} | {mean_ndcg:.4f} | {ci_str} | {diff_str} | {p_str} | {sig_str} |")

    # Effect size (Cohen's d)
    lines.extend(["", "## Effect Sizes (Cohen's d vs CE baseline)", ""])
    print(f"\n{'Hypothesis':<30} {'Cohen d':>10} {'Interpretation':>15}")
    print("-" * 60)

    for name, ndcgs in all_per_query_ndcgs.items():
        if name == baseline_name:
            continue
        diff = ndcgs - baseline_ndcgs
        d = diff.mean() / (diff.std() + 1e-12)
        if abs(d) < 0.2:
            interp = "negligible"
        elif abs(d) < 0.5:
            interp = "small"
        elif abs(d) < 0.8:
            interp = "medium"
        else:
            interp = "large"
        print(f"{name:<30} {d:>10.3f} {interp:>15}")
        lines.append(f"- **{name}**: d = {d:.3f} ({interp})")

    # Sample-level distribution
    lines.extend(["", "## Per-Query nDCG Distribution", ""])
    for name, ndcgs in all_per_query_ndcgs.items():
        percentiles = np.percentile(ndcgs, [0, 25, 50, 75, 100])
        lines.append(f"- **{name}**: min={percentiles[0]:.3f} Q1={percentiles[1]:.3f} "
                     f"median={percentiles[2]:.3f} Q3={percentiles[3]:.3f} max={percentiles[4]:.3f}")

    with open(output_dir / "statistical_validation.md", "w") as f:
        f.write("\n".join(lines))

    # Save raw per-query data
    raw_data = {name: ndcgs.tolist() for name, ndcgs in all_per_query_ndcgs.items()}
    with open(output_dir / "statistical_validation_raw.json", "w") as f:
        json.dump(raw_data, f, indent=2)

    print(f"\nResults saved to {output_dir}/statistical_validation.*")


if __name__ == "__main__":
    main()
