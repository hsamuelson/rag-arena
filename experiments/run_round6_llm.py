#!/usr/bin/env python3
"""Round 6: LLM-powered hypotheses + deep pool variants.

15 hypotheses at n=200 with statistical validation.
LLM hypotheses use Qwen 3.5 122B via Ollama (~17s/query).

Estimated runtime:
- Non-LLM hypotheses (5): ~5 min each = ~25 min
- LLM hypotheses (10): ~60 min each = ~600 min
- Total: ~10-11 hours

To save time, we run non-LLM first, then LLM hypotheses.
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
from arena.hypotheses.deep_pool_ce import DeepPoolCEHypothesis
from arena.hypotheses.deep_pool_50_ce import DeepPool50CEHypothesis
from arena.hypotheses.deep_pool_100_ce import DeepPool100CEHypothesis
from arena.hypotheses.bm25_text_feature_ce import BM25TextFeatureCEHypothesis
from arena.hypotheses.adaptive_pool_depth_ce import AdaptivePoolDepthCEHypothesis
from arena.hypotheses.two_stage_deep_pool_ce import TwoStageDeepPoolCEHypothesis
from arena.hypotheses.temperature_scaled_ce import TemperatureScaledCEHypothesis
from arena.hypotheses.llm_query_decomp_ce import LLMQueryDecompCEHypothesis
from arena.hypotheses.ircot_simplified import IRCoTSimplifiedHypothesis
from arena.hypotheses.ircot_full import IRCoTFullHypothesis
from arena.hypotheses.llm_bridge_entity_ce import LLMBridgeEntityCEHypothesis
from arena.hypotheses.llm_relevance_judge import LLMRelevanceJudgeHypothesis
from arena.hypotheses.llm_query_expansion_ce import LLMQueryExpansionCEHypothesis
from arena.hypotheses.deep_pool_decomp_ce import DeepPoolDecompCEHypothesis
from arena.hypotheses.deep_pool_ircot_ce import DeepPoolIRCoTCEHypothesis
from arena.hypotheses.llm_query_fusion_ce import LLMQueryFusionCEHypothesis
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
    print(f"Round 6: {max_samples} samples, LLM-powered + deep pool variants\n")

    benchmark = HotpotQABenchmark(max_samples=max_samples)
    runner = ArenaRunner(config)
    hybrid = HybridBackend(config)

    # --- Baselines (re-run CE and deep-pool-30 for fair comparison) ---
    ce_baseline = CrossEncoderRerankerHypothesis()
    deep30 = DeepPoolCEHypothesis(pool_size=30, final_k=10)
    deep30.set_backend(hybrid)

    # --- Non-LLM hypotheses (fast) ---
    deep50 = DeepPool50CEHypothesis()
    deep50.set_backend(hybrid)

    deep100 = DeepPool100CEHypothesis()
    deep100.set_backend(hybrid)

    adaptive = AdaptivePoolDepthCEHypothesis()
    adaptive.set_backend(hybrid)

    two_stage = TwoStageDeepPoolCEHypothesis()
    two_stage.set_backend(hybrid)

    # --- LLM hypotheses (slow) ---
    llm_decomp = LLMQueryDecompCEHypothesis()
    llm_decomp.set_backend(hybrid)

    ircot_simple = IRCoTSimplifiedHypothesis()
    ircot_simple.set_backend(hybrid)

    ircot_full = IRCoTFullHypothesis()
    ircot_full.set_backend(hybrid)

    llm_bridge = LLMBridgeEntityCEHypothesis()
    llm_bridge.set_backend(hybrid)

    llm_expansion = LLMQueryExpansionCEHypothesis()
    llm_expansion.set_backend(hybrid)

    deep_decomp = DeepPoolDecompCEHypothesis()
    deep_decomp.set_backend(hybrid)

    deep_ircot = DeepPoolIRCoTCEHypothesis()
    deep_ircot.set_backend(hybrid)

    llm_fusion = LLMQueryFusionCEHypothesis()
    llm_fusion.set_backend(hybrid)

    llm_judge = LLMRelevanceJudgeHypothesis()

    hypotheses = [
        # Baselines
        ("cross-encoder (baseline)", ce_baseline),
        ("deep-pool-30-ce (baseline)", deep30),
        # Non-LLM (run first, fast)
        ("deep-pool-50-ce", deep50),
        ("deep-pool-100-ce", deep100),
        ("bm25-text-feature-ce", BM25TextFeatureCEHypothesis()),
        ("adaptive-pool-depth-ce", adaptive),
        ("two-stage-deep-pool-ce", two_stage),
        ("temperature-scaled-ce-2.0", TemperatureScaledCEHypothesis(temperature=2.0)),
        # LLM-powered (slow)
        ("llm-query-decomp-ce", llm_decomp),
        ("ircot-simplified", ircot_simple),
        ("ircot-full-2hop", ircot_full),
        ("llm-bridge-entity-ce", llm_bridge),
        ("llm-query-expansion-ce", llm_expansion),
        ("deep-pool-decomp-ce", deep_decomp),
        ("deep-pool-ircot-ce", deep_ircot),
        ("llm-query-fusion-ce", llm_fusion),
        ("llm-relevance-judge", llm_judge),
    ]

    results_data = {}

    for name, hyp in hypotheses:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        start = time.time()
        try:
            results = runner.run_arena(
                benchmark=benchmark, backend=hybrid, hypotheses=[hyp],
                max_samples=max_samples, skip_llm=True, verbose=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
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

        # Save incremental results after each hypothesis
        _save_incremental(results_data, output_dir)

    # --- Statistical comparison ---
    _print_and_save_comparison(results_data, output_dir, max_samples)


def _save_incremental(results_data, output_dir):
    """Save results so far (in case of crash during long LLM runs)."""
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round6_llm_raw_incremental.json", "w") as f:
        json.dump(raw, f, indent=2)


def _print_and_save_comparison(results_data, output_dir, max_samples):
    print(f"\n{'#'*70}")
    print("ROUND 6: STATISTICAL COMPARISON (n=200)")
    print(f"{'#'*70}")

    # Use CE as baseline
    if "cross-encoder (baseline)" not in results_data:
        print("ERROR: CE baseline not found, cannot compare")
        return

    baseline = results_data["cross-encoder (baseline)"]
    baseline_ndcgs = baseline["ndcgs"]

    # Also compare against deep-pool-30
    deep30_data = results_data.get("deep-pool-30-ce (baseline)")

    print(f"\n{'Hypothesis':<35} {'nDCG':>7} {'95% CI':>22} {'vs CE':>8} {'vs DP30':>8} {'p-val':>7} {'Sig':>5} {'Time':>8}")
    print("-" * 105)

    md_lines = [
        "# Round 6: LLM-Powered Hypotheses + Deep Pool Variants",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {max_samples} (statistically powered)",
        f"**Benchmark:** HotpotQA",
        f"**Backend:** Hybrid BM25+Dense",
        f"**LLM:** Qwen 3.5 122B via Ollama",
        f"**Statistical tests:** Bootstrap 95% CI, paired permutation test (5000 perms)",
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

        if name == "cross-encoder (baseline)":
            diff_ce = "baseline"
            p_str = "-"
            sig_str = "-"
        else:
            diff_ce_val = (mean - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            diff_ce = f"{diff_ce_val:+.1f}%"
            if mean >= baseline_ndcgs.mean():
                p_val = paired_permutation_test(ndcgs, baseline_ndcgs)
            else:
                p_val = 1.0
            p_str = f"{p_val:.4f}"
            sig_str = "YES" if p_val < 0.05 else "no"

        # Also compute vs deep-pool-30
        if deep30_data and name != "deep-pool-30-ce (baseline)":
            dp30_ndcgs = deep30_data["ndcgs"]
            diff_dp30_val = (mean - dp30_ndcgs.mean()) / dp30_ndcgs.mean() * 100
            diff_dp30 = f"{diff_dp30_val:+.1f}%"
        elif name == "deep-pool-30-ce (baseline)":
            diff_dp30 = "baseline"
        else:
            diff_dp30 = "N/A"

        print(f"{name:<35} {mean:>7.4f} {ci_str:>22} {diff_ce:>8} {diff_dp30:>8} {p_str:>7} {sig_str:>5} {time_str:>8}")
        md_lines.append(f"| {rank} | {name} | {mean:.4f} | {ci_str} | {diff_ce} | {diff_dp30} | {p_str} | {sig_str} | {time_str} |")

    md_lines.extend([
        "",
        "## Interpretation",
        "",
        "### Key Questions:",
        "1. Do deeper pools (50, 100) continue to improve over pool-30?",
        "2. Do LLM-based multi-hop approaches (IRCoT, decomposition) beat deep-pool?",
        "3. Does combining deep-pool + LLM stack the gains?",
        "",
        "Results with p < 0.05 vs CE baseline AND positive difference are genuine improvements.",
    ])

    with open(output_dir / "round6_llm.md", "w") as f:
        f.write("\n".join(md_lines))

    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round6_llm_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    all_results = [data["result"] for data in results_data.values()]
    runner_tmp = ArenaRunner(ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml"))
    runner_tmp.save_results(all_results, output_dir / "round6_llm.json")

    print(f"\nResults saved to {output_dir}/round6_llm.*")


if __name__ == "__main__":
    main()
