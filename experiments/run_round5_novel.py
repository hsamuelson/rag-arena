#!/usr/bin/env python3
"""Round 5: 15 novel approaches with proper statistical validation.

All experiments at n=200 with bootstrap CIs and permutation tests.
No more n=50 noise-as-signal mistakes.
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
from arena.metrics.scoring import ndcg_at_k
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.query_decomp_ce import QueryDecompCEHypothesis
from arena.hypotheses.ce_larger_model import CELargerModelHypothesis
from arena.hypotheses.ce_with_context import CEWithContextHypothesis
from arena.hypotheses.ce_sentence_level import CESentenceLevelHypothesis
from arena.hypotheses.ce_keyword_focused import CEKeywordFocusedHypothesis
from arena.hypotheses.ce_answer_extraction import CEAnswerExtractionHypothesis
from arena.hypotheses.bm25_boosted_ce import BM25BoostedCEHypothesis
from arena.hypotheses.ce_negative_feedback import CENegativeFeedbackHypothesis
from arena.hypotheses.deep_pool_ce import DeepPoolCEHypothesis
from arena.hypotheses.ce_cross_doc import CECrossDocHypothesis
from arena.hypotheses.ce_query_type_adaptive import CEQueryTypeAdaptiveHypothesis
from arena.hypotheses.ce_coverage_greedy import CECoverageGreedyHypothesis
from arena.hypotheses.ce_multihop_iterative import CEMultihopIterativeHypothesis
from arena.hypotheses.ce_ensemble import CEEnsembleHypothesis
from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
from arena.runners.arena_runner import ArenaRunner


def compute_per_query_ndcg(per_question_results, k=10):
    ndcgs = []
    for q in per_question_results:
        retrieved = q["retrieved_ids"][:k]
        relevant = q["relevant_ids"]
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    return np.array(ndcgs)


def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)
    means = sorted(scores[rng.randint(0, len(scores), size=len(scores))].mean()
                   for _ in range(n_bootstrap))
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def paired_permutation_test(scores_a, scores_b, n_perms=5000):
    rng = np.random.RandomState(42)
    observed = scores_a.mean() - scores_b.mean()
    diffs = scores_a - scores_b
    count = sum(1 for _ in range(n_perms)
                if (diffs * rng.choice([-1, 1], size=len(diffs))).mean() >= observed)
    return count / n_perms


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 200
    print(f"Round 5: {max_samples} samples, proper statistical validation\n")

    benchmark = HotpotQABenchmark(max_samples=max_samples)
    runner = ArenaRunner(config)
    hybrid = HybridBackend(config)

    # All hypotheses to test
    hypotheses = [
        ("cross-encoder (baseline)", CrossEncoderRerankerHypothesis()),
        ("query-decomp-ce-max", QueryDecompCEHypothesis(aggregation="max")),
        ("query-decomp-ce-mean", QueryDecompCEHypothesis(aggregation="mean")),
        ("ce-MiniLM-L-12", CELargerModelHypothesis("cross-encoder/ms-marco-MiniLM-L-12-v2")),
        ("ce-with-context", CEWithContextHypothesis()),
        ("ce-sentence-max", CESentenceLevelHypothesis(aggregation="max")),
        ("ce-sentence-top2", CESentenceLevelHypothesis(aggregation="top2mean")),
        ("ce-keyword-focused-0.5", CEKeywordFocusedHypothesis(focus_weight=0.5)),
        ("ce-answer-extraction", CEAnswerExtractionHypothesis()),
        ("bm25-boosted-ce-0.15", BM25BoostedCEHypothesis(bm25_weight=0.15)),
        ("ce-negative-feedback", CENegativeFeedbackHypothesis()),
        ("ce-cross-doc-5k", CECrossDocHypothesis(top_k_pairs=5)),
        ("ce-query-type-adaptive", CEQueryTypeAdaptiveHypothesis()),
        ("ce-coverage-greedy-0.2", CECoverageGreedyHypothesis(coverage_weight=0.2)),
        ("ce-multihop-iterative", CEMultihopIterativeHypothesis()),
    ]

    # Also test deep pool (requires backend injection)
    deep_pool = DeepPoolCEHypothesis(pool_size=30, final_k=10)
    deep_pool.set_backend(hybrid)
    hypotheses.append(("deep-pool-30-ce", deep_pool))

    results_data = {}

    for name, hyp in hypotheses:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        start = time.time()
        results = runner.run_arena(
            benchmark=benchmark, backend=hybrid, hypotheses=[hyp],
            max_samples=max_samples, skip_llm=True, verbose=True,
        )
        elapsed = time.time() - start

        result = results[0]
        ndcgs = compute_per_query_ndcg(result.per_question)
        results_data[name] = {
            "result": result,
            "ndcgs": ndcgs,
            "mean_ndcg": ndcgs.mean(),
            "elapsed": elapsed,
        }
        print(f"  nDCG={ndcgs.mean():.4f} ({elapsed:.1f}s)")

    # --- Statistical comparison ---
    print(f"\n{'#'*70}")
    print("ROUND 5: STATISTICAL COMPARISON (n=200)")
    print(f"{'#'*70}")

    baseline = results_data["cross-encoder (baseline)"]
    baseline_ndcgs = baseline["ndcgs"]

    print(f"\n{'Hypothesis':<35} {'nDCG':>7} {'95% CI':>22} {'vs CE':>8} {'p-val':>7} {'Sig':>5}")
    print("-" * 90)

    md_lines = [
        "# Round 5: Novel Approaches with Statistical Validation",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {max_samples} (statistically powered)",
        f"**Benchmark:** HotpotQA ({benchmark._n_docs if hasattr(benchmark, '_n_docs') else '~2000'} docs)",
        f"**Backend:** Hybrid BM25+Dense",
        f"**Statistical tests:** Bootstrap 95% CI, paired permutation test (5000 perms)",
        "",
        "## Results (sorted by nDCG@10)",
        "",
        "| Rank | Hypothesis | nDCG@10 | 95% CI | vs CE | p-value | Sig? |",
        "|------|-----------|---------|--------|-------|---------|------|",
    ]

    sorted_names = sorted(results_data.keys(), key=lambda n: results_data[n]["mean_ndcg"], reverse=True)

    for rank, name in enumerate(sorted_names, 1):
        data = results_data[name]
        ndcgs = data["ndcgs"]
        mean = ndcgs.mean()
        ci_lo, ci_hi = bootstrap_ci(ndcgs)
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"

        if name == "cross-encoder (baseline)":
            diff_str = "baseline"
            p_str = "-"
            sig_str = "-"
        else:
            diff = (mean - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            diff_str = f"{diff:+.1f}%"
            if mean >= baseline_ndcgs.mean():
                p_val = paired_permutation_test(ndcgs, baseline_ndcgs)
            else:
                p_val = paired_permutation_test(baseline_ndcgs, ndcgs)
                p_val = 1.0  # mark as not beating baseline
            p_str = f"{p_val:.4f}"
            sig_str = "YES" if p_val < 0.05 else "no"

        print(f"{name:<35} {mean:>7.4f} {ci_str:>22} {diff_str:>8} {p_str:>7} {sig_str:>5}")
        md_lines.append(f"| {rank} | {name} | {mean:.4f} | {ci_str} | {diff_str} | {p_str} | {sig_str} |")

    # Save results
    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "Only results with p < 0.05 AND positive vs CE difference are genuine improvements.",
        "Everything else is noise at this sample size.",
    ])

    with open(output_dir / "round5_novel.md", "w") as f:
        f.write("\n".join(md_lines))

    # Save raw data
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round5_novel_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Save experiment results
    all_results = [data["result"] for data in results_data.values()]
    runner.save_results(all_results, output_dir / "round5_novel.json")

    print(f"\nResults saved to {output_dir}/round5_novel.*")


if __name__ == "__main__":
    main()
