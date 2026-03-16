#!/usr/bin/env python3
"""Round 6 (Top picks): 4 best hypotheses + baselines.

Strategy:
- Phase 1 (fast): deep-pool-50-ce in parallel with baselines — ~5 min
- Phase 2 (LLM): ircot-simplified, llm-query-decomp-ce, deep-pool-ircot-ce — sequential

Total estimated: ~3-4 hours instead of 11.
"""

import json
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.hybrid_backend import HybridBackend
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.metrics.scoring import ndcg_at_k
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.deep_pool_ce import DeepPoolCEHypothesis
from arena.hypotheses.deep_pool_50_ce import DeepPool50CEHypothesis
from arena.hypotheses.llm_query_decomp_ce import LLMQueryDecompCEHypothesis
from arena.hypotheses.ircot_simplified import IRCoTSimplifiedHypothesis
from arena.hypotheses.deep_pool_ircot_ce import DeepPoolIRCoTCEHypothesis
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


def run_single_hypothesis(name, hyp, config, benchmark_cls, max_samples, backend_config):
    """Run a single hypothesis in its own thread/process."""
    print(f"\n  Starting: {name}")
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    runner = ArenaRunner(config)
    benchmark = benchmark_cls(max_samples=max_samples)
    backend = HybridBackend(config)

    # Inject backend for deep pool hypotheses
    if hasattr(hyp, 'set_backend'):
        hyp.set_backend(backend)

    start = time.time()
    results = runner.run_arena(
        benchmark=benchmark, backend=backend, hypotheses=[hyp],
        max_samples=max_samples, skip_llm=True, verbose=True,
    )
    elapsed = time.time() - start

    result = results[0]
    ndcgs = compute_per_query_ndcg(result.per_question)
    print(f"\n  Finished: {name} — nDCG={ndcgs.mean():.4f} ({elapsed:.1f}s)")
    return name, result, ndcgs, elapsed


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 200
    print(f"Round 6 (Top Picks): {max_samples} samples\n")
    print("=" * 60)

    results_data = {}

    # ─── Phase 1: Non-LLM hypotheses (can share backend) ───
    print("\n### Phase 1: Non-LLM hypotheses (CE baseline, deep-pool-30, deep-pool-50)")
    print("    Running sequentially (shared CE model for efficiency)\n")

    runner = ArenaRunner(config)
    benchmark = HotpotQABenchmark(max_samples=max_samples)
    hybrid = HybridBackend(config)

    phase1_hyps = [
        ("cross-encoder (baseline)", CrossEncoderRerankerHypothesis()),
        ("deep-pool-30-ce", DeepPoolCEHypothesis(pool_size=30, final_k=10)),
        ("deep-pool-50-ce", DeepPool50CEHypothesis()),
    ]

    for name, hyp in phase1_hyps:
        if hasattr(hyp, 'set_backend'):
            hyp.set_backend(hybrid)

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
            "result": result, "ndcgs": ndcgs,
            "mean_ndcg": ndcgs.mean(), "elapsed": elapsed,
        }
        print(f"  nDCG={ndcgs.mean():.4f} ({elapsed:.1f}s)")
        _save_incremental(results_data, output_dir)

    # ─── Phase 2: LLM hypotheses (sequential, Ollama bottleneck) ───
    print("\n\n### Phase 2: LLM-powered hypotheses (sequential — Ollama serializes)")
    print("    Estimated: ~60 min each × 3 = ~3 hours\n")

    phase2_hyps = [
        ("ircot-simplified", IRCoTSimplifiedHypothesis()),
        ("llm-query-decomp-ce", LLMQueryDecompCEHypothesis()),
        ("deep-pool-ircot-ce", DeepPoolIRCoTCEHypothesis()),
    ]

    for name, hyp in phase2_hyps:
        if hasattr(hyp, 'set_backend'):
            hyp.set_backend(hybrid)

        print(f"\n{'='*60}")
        print(f"  {name} (LLM-powered)")
        print(f"{'='*60}")

        start = time.time()
        try:
            results = runner.run_arena(
                benchmark=benchmark, backend=hybrid, hypotheses=[hyp],
                max_samples=max_samples, skip_llm=True, verbose=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - start

        result = results[0]
        ndcgs = compute_per_query_ndcg(result.per_question)
        results_data[name] = {
            "result": result, "ndcgs": ndcgs,
            "mean_ndcg": ndcgs.mean(), "elapsed": elapsed,
        }
        print(f"  nDCG={ndcgs.mean():.4f} ({elapsed:.1f}s)")
        _save_incremental(results_data, output_dir)

    # ─── Statistical comparison ───
    _print_and_save_comparison(results_data, output_dir, max_samples)


def _save_incremental(results_data, output_dir):
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round6_incremental.json", "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  [Saved incremental results: {len(results_data)} hypotheses so far]")


def _print_and_save_comparison(results_data, output_dir, max_samples):
    print(f"\n{'#'*70}")
    print("ROUND 6: STATISTICAL COMPARISON (n=200)")
    print(f"{'#'*70}")

    baseline = results_data.get("cross-encoder (baseline)")
    deep30 = results_data.get("deep-pool-30-ce")

    if not baseline:
        print("ERROR: CE baseline missing")
        return

    baseline_ndcgs = baseline["ndcgs"]

    print(f"\n{'Hypothesis':<35} {'nDCG':>7} {'95% CI':>22} {'vs CE':>8} {'vs DP30':>8} {'p(vsCE)':>8} {'Sig':>5} {'Time':>8}")
    print("-" * 110)

    md_lines = [
        "# Round 6: Top LLM-Powered Hypotheses",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {max_samples}",
        f"**Benchmark:** HotpotQA",
        f"**Backend:** Hybrid BM25+Dense",
        f"**LLM:** Qwen 3.5 122B via Ollama",
        f"**Stats:** Bootstrap 95% CI, paired permutation test (5000 perms)",
        "",
        "## Results (sorted by nDCG@10)",
        "",
        "| Rank | Hypothesis | nDCG@10 | 95% CI | vs CE | vs DP30 | p-value | Sig? | Time |",
        "|------|-----------|---------|--------|-------|---------|---------|------|------|",
    ]

    sorted_names = sorted(results_data.keys(), key=lambda n: results_data[n]["mean_ndcg"], reverse=True)

    for rank, name in enumerate(sorted_names, 1):
        data = results_data[name]
        ndcgs = data["ndcgs"]
        mean = ndcgs.mean()
        ci_lo, ci_hi = bootstrap_ci(ndcgs)
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        time_str = f"{data['elapsed']:.0f}s"

        # vs CE baseline
        if name == "cross-encoder (baseline)":
            diff_ce = "baseline"
            p_str = "-"
            sig_str = "-"
        else:
            diff_val = (mean - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            diff_ce = f"{diff_val:+.1f}%"
            if mean >= baseline_ndcgs.mean():
                p_val = paired_permutation_test(ndcgs, baseline_ndcgs)
            else:
                p_val = 1.0
            p_str = f"{p_val:.4f}"
            sig_str = "YES" if p_val < 0.05 else "no"

        # vs deep-pool-30
        if deep30 and name not in ("deep-pool-30-ce",):
            dp_ndcgs = deep30["ndcgs"]
            dp_diff = (mean - dp_ndcgs.mean()) / dp_ndcgs.mean() * 100
            diff_dp30 = f"{dp_diff:+.1f}%"
        elif name == "deep-pool-30-ce":
            diff_dp30 = "baseline"
        else:
            diff_dp30 = "N/A"

        print(f"{name:<35} {mean:>7.4f} {ci_str:>22} {diff_ce:>8} {diff_dp30:>8} {p_str:>8} {sig_str:>5} {time_str:>8}")
        md_lines.append(f"| {rank} | {name} | {mean:.4f} | {ci_str} | {diff_ce} | {diff_dp30} | {p_str} | {sig_str} | {time_str} |")

    # Interpretation
    md_lines.extend([
        "",
        "## Key Questions Answered",
        "",
        "1. **Does deeper pooling keep helping?** (pool-50 vs pool-30)",
        "2. **Does LLM multi-hop retrieval beat deep pooling?** (IRCoT vs pool-30)",
        "3. **Do they stack?** (deep-pool + IRCoT vs either alone)",
    ])

    with open(output_dir / "round6_top.md", "w") as f:
        f.write("\n".join(md_lines))

    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round6_top_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nResults saved to {output_dir}/round6_top.*")


if __name__ == "__main__":
    main()
