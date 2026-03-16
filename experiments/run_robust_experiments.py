#!/usr/bin/env python3
"""Embedder-Robust Retrieval Experiments.

Tests 6 novel hypotheses on BOTH nomic (137M) and snowflake (335M) embedders.
Goal: find techniques that beat BGE+snowflake (0.4867) on BOTH embedders.

A technique is "robust" if it beats the best single-strategy baseline on
BOTH embedders simultaneously.

Baselines:
  | Baseline              | nomic  | snowflake |
  |-----------------------|--------|-----------|
  | Flat (no reranking)   | 0.2656 | 0.3614    |
  | CE L-12 (top-50)      | 0.3720 | 0.4475    |
  | BGE-v2-m3 (top-50)    | 0.3941 | 0.4867    |
  | LI+MRAM (best comp.)  | 0.4148 | 0.4208    |
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.benchmarks.beir_subset import BEIRSubsetBenchmark
from arena.metrics.scoring import ndcg_at_k, compute_scorecard

# Known baselines per embedder
BASELINES = {
    "nomic": {
        "Flat baseline": 0.2656,
        "CE L-12 (top-50)": 0.3720,
        "BGE-v2-m3 (top-50)": 0.3941,
        "LI+MRAM (best composite)": 0.4148,
    },
    "snowflake": {
        "Flat baseline": 0.3614,
        "CE L-12 (top-50)": 0.4475,
        "BGE-v2-m3 (top-50)": 0.4867,
        "LI+MRAM (best composite)": 0.4208,
    },
}

EMBEDDER_CONFIGS = {
    "nomic": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "trust_remote_code": True,
    },
    "snowflake": {
        "model_name": "Snowflake/snowflake-arctic-embed-l",
        "trust_remote_code": True,
    },
}


def compute_per_query_ndcg(per_question_results, k=10):
    return np.array([
        ndcg_at_k(q["retrieved_ids"][:k], q["relevant_ids"], k)
        for q in per_question_results
    ])


def bootstrap_ci(scores, n_bootstrap=2000, ci=0.95):
    rng = np.random.RandomState(42)
    means = sorted(
        scores[rng.randint(0, len(scores), size=len(scores))].mean()
        for _ in range(n_bootstrap)
    )
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def paired_wilcoxon_test(scores_a, scores_b):
    try:
        from scipy.stats import wilcoxon
        diffs = scores_a - scores_b
        diffs = diffs[diffs != 0]
        if len(diffs) < 10:
            return 1.0
        _, p_val = wilcoxon(diffs)
        return float(p_val)
    except ImportError:
        rng = np.random.RandomState(42)
        observed = scores_a.mean() - scores_b.mean()
        diffs = scores_a - scores_b
        count = sum(
            1 for _ in range(5000)
            if (diffs * rng.choice([-1, 1], size=len(diffs))).mean() >= observed
        )
        return count / 5000


def run_hypothesis(name, hyp, samples, backend, top_k):
    from tqdm import tqdm

    per_question = []
    for sample in tqdm(samples, desc=name):
        t0 = time.time()
        results, embeddings = backend.retrieve_with_embeddings(sample.question, top_k)
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

    scorecard = compute_scorecard(per_question, k=top_k)
    return per_question, scorecard


def build_baselines(backend):
    """Build baseline strategies for comparison."""
    from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
    from arena.hypotheses.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis
    from arena.hypotheses.bge_reranker import BGERerankerHypothesis

    strategies = []

    flat = FlatBaselineHypothesis()
    strategies.append(("Flat baseline", flat))

    ce = DeepPool50CEL12Hypothesis()
    ce.set_backend(backend)
    strategies.append(("CE L-12 (top-50)", ce))

    bge = BGERerankerHypothesis()
    bge.set_backend(backend)
    strategies.append(("BGE-v2-m3 (top-50)", bge))

    return strategies


def build_experiments(backend):
    """Build the 6 new experiment hypotheses."""
    from arena.hypotheses.csls_prefilter_ce import CSLSPrefilterCEHypothesis
    from arena.hypotheses.lid_gated_pool_ce import LIDGatedPoolCEHypothesis
    from arena.hypotheses.hub_aware_deep_pool_ce import HubAwareDeepPoolCEHypothesis
    from arena.hypotheses.gated_mram_ce import GatedMRAMCEHypothesis
    from arena.hypotheses.cross_model_maxsim import CrossModelMaxSimHypothesis
    from arena.hypotheses.routed_reranker import RoutedRerankerHypothesis

    strategies = []

    # Exp 1: CSLS Pre-Filter + CE
    csls = CSLSPrefilterCEHypothesis()
    csls.set_backend(backend)
    strategies.append(("Exp1: CSLS Pre-Filter + CE", csls))

    # Exp 2: LID-Gated Adaptive Pool + CE
    lid = LIDGatedPoolCEHypothesis()
    lid.set_backend(backend)
    strategies.append(("Exp2: LID-Gated Pool + CE", lid))

    # Exp 3: Hub-Aware Deep Pool + CE
    hub = HubAwareDeepPoolCEHypothesis()
    hub.set_backend(backend)
    strategies.append(("Exp3: Hub-Aware Pool + CE", hub))

    # Exp 4: Gated MRAM + CE
    gated = GatedMRAMCEHypothesis()
    gated.set_backend(backend)
    strategies.append(("Exp4: Gated MRAM + CE", gated))

    # Exp 5: Cross-Model MaxSim
    cross = CrossModelMaxSimHypothesis()
    cross.set_backend(backend)
    strategies.append(("Exp5: Cross-Model MaxSim", cross))

    # Exp 6: Routed Reranker
    routed = RoutedRerankerHypothesis()
    routed.set_backend(backend)
    strategies.append(("Exp6: Routed Reranker", routed))

    return strategies


def robustness_analysis(all_embedder_results):
    """Determine which techniques are robust across both embedders."""
    print(f"\n{'#' * 70}")
    print("  ROBUSTNESS ANALYSIS — CROSS-EMBEDDER COMPARISON")
    print(f"{'#' * 70}\n")

    # Best single-strategy baselines per embedder
    best_baselines = {
        "nomic": BASELINES["nomic"]["BGE-v2-m3 (top-50)"],
        "snowflake": BASELINES["snowflake"]["BGE-v2-m3 (top-50)"],
    }

    robust_techniques = []

    # Get experiment names (from any embedder)
    exp_names = set()
    for emb_results in all_embedder_results.values():
        for name in emb_results:
            if name.startswith("Exp"):
                exp_names.add(name)

    for exp_name in sorted(exp_names):
        print(f"  {exp_name}:")
        all_beat = True
        for embedder in ["nomic", "snowflake"]:
            data = all_embedder_results.get(embedder, {}).get(exp_name)
            baseline = best_baselines[embedder]
            if data is None:
                print(f"    {embedder}: MISSING")
                all_beat = False
                continue

            mean_ndcg = data["mean_ndcg"]
            ci = bootstrap_ci(data["ndcgs"])
            delta = mean_ndcg - baseline
            pct = delta / baseline * 100 if baseline > 0 else 0

            status = "BEATS" if mean_ndcg > baseline else "LOSES"
            print(f"    {embedder}: {mean_ndcg:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] "
                  f"vs {baseline:.4f} ({delta:+.4f}, {pct:+.1f}%) → {status}")

            if mean_ndcg <= baseline:
                all_beat = False

        # Statistical significance on snowflake (hardest baseline)
        snow_data = all_embedder_results.get("snowflake", {}).get(exp_name)
        snow_bge = all_embedder_results.get("snowflake", {}).get("BGE-v2-m3 (top-50)")
        if snow_data is not None and snow_bge is not None:
            p_val = paired_wilcoxon_test(snow_data["ndcgs"], snow_bge["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    vs snowflake BGE: p={p_val:.4f} {sig}")

        if all_beat:
            robust_techniques.append(exp_name)
            print(f"    → ROBUST ✓")
        else:
            print(f"    → NOT robust")
        print()

    print(f"  {'=' * 60}")
    if robust_techniques:
        print(f"  ROBUST TECHNIQUES ({len(robust_techniques)}):")
        for t in robust_techniques:
            print(f"    - {t}")
    else:
        print(f"  NO technique beat BGE on BOTH embedders.")
        print(f"  → Strong negative result: component quality dominates architecture.")
    print(f"  {'=' * 60}")

    return robust_techniques


def run_on_embedder(embedder_key, config, corpus, samples, output_dir, run_baselines=True):
    """Run all experiments on a single embedder."""
    emb_config = EMBEDDER_CONFIGS[embedder_key]

    print(f"\n{'=' * 70}")
    print(f"  EMBEDDER: {emb_config['model_name']}")
    print(f"{'=' * 70}")

    # Create backend
    from arena.backends.hybrid_st_backend import HybridSTBackend
    backend = HybridSTBackend(
        config,
        model_name=emb_config["model_name"],
        trust_remote_code=emb_config.get("trust_remote_code", False),
    )

    # Ingest with caching
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    print(f"  Ingesting {len(corpus)} docs (cache: {cache_dir})...")
    t0 = time.time()
    backend.ingest(corpus, cache_dir=cache_dir)
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    all_results = {}

    # Run baselines (for live comparison)
    if run_baselines:
        baselines = build_baselines(backend)
        for name, hyp in baselines:
            print(f"\n  Running baseline: {name}")
            try:
                per_q, scorecard = run_hypothesis(name, hyp, samples, backend, config.top_k)
                ndcgs = compute_per_query_ndcg(per_q)
                ci = bootstrap_ci(ndcgs)
                all_results[name] = {
                    "per_question": per_q,
                    "ndcgs": ndcgs,
                    "mean_ndcg": float(ndcgs.mean()),
                }
                print(f"    nDCG@10 = {ndcgs.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Run experiments
    experiments = build_experiments(backend)
    for name, hyp in experiments:
        print(f"\n  Running: {name}")
        start = time.time()
        try:
            per_q, scorecard = run_hypothesis(name, hyp, samples, backend, config.top_k)
            ndcgs = compute_per_query_ndcg(per_q)
            ci = bootstrap_ci(ndcgs)
            elapsed = time.time() - start
            all_results[name] = {
                "per_question": per_q,
                "ndcgs": ndcgs,
                "mean_ndcg": float(ndcgs.mean()),
                "elapsed": elapsed,
            }
            print(f"    nDCG@10 = {ndcgs.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Save incrementally
        raw = {n: d["ndcgs"].tolist() for n, d in all_results.items()}
        with open(output_dir / f"robust_{embedder_key}_raw.json", "w") as f:
            json.dump(raw, f, indent=2)

    # Print results table
    print(f"\n  {'─' * 60}")
    print(f"  Results for {embedder_key}:")
    print(f"  {'Strategy':<35s}  {'nDCG@10':>8s}  {'vs BGE':>10s}")
    print(f"  {'─' * 60}")

    bge_baseline = BASELINES[embedder_key]["BGE-v2-m3 (top-50)"]
    for name, data in all_results.items():
        ndcg = data["mean_ndcg"]
        delta = ndcg - bge_baseline
        pct = delta / bge_baseline * 100 if bge_baseline > 0 else 0
        print(f"  {name:<35s}  {ndcg:8.4f}  {delta:+.4f} ({pct:+.1f}%)")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Embedder-robust retrieval experiments (6 hypotheses × 2 embedders)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Number of queries (default: 200)",
    )
    parser.add_argument(
        "--embedders", nargs="+", default=["nomic", "snowflake"],
        choices=["nomic", "snowflake"],
        help="Which embedders to test (default: both)",
    )
    parser.add_argument(
        "--experiments", nargs="+", type=int, default=None,
        help="Which experiments to run (1-6, default: all)",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip running baselines (use cached values)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Quick smoke test: 10 queries, nomic only",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results"),
        help="Output directory",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.max_samples = 10
        args.embedders = ["nomic"]
        args.skip_baselines = True
        print("SMOKE TEST MODE: 10 queries, nomic only, no baselines")

    np.random.seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EMBEDDER-ROBUST RETRIEVAL EXPERIMENTS")
    print("=" * 70)
    print(f"Embedders:  {', '.join(args.embedders)}")
    print(f"Queries:    {args.max_samples} on FiQA")
    print(f"Seed:       42")
    print(f"Output:     {output_dir}")
    print()

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")

    # Load FiQA
    print("### Loading FiQA benchmark...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:args.max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # Run on each embedder
    all_embedder_results = {}

    for embedder_key in args.embedders:
        embedder_results = run_on_embedder(
            embedder_key, config, corpus, samples, output_dir,
            run_baselines=not args.skip_baselines,
        )
        all_embedder_results[embedder_key] = embedder_results

    # Cross-embedder robustness analysis
    if len(args.embedders) >= 2:
        robust = robustness_analysis(all_embedder_results)
    else:
        robust = []

    # Save final summary
    summary = {
        "date": datetime.now().isoformat(),
        "goal": "Find embedder-robust technique that beats BGE+snowflake (0.4867)",
        "n_queries": args.max_samples,
        "seed": 42,
        "embedders": args.embedders,
        "baselines": BASELINES,
        "results": {
            embedder: {
                name: {
                    "ndcg_at_10": data["mean_ndcg"],
                    "elapsed_s": data.get("elapsed"),
                }
                for name, data in emb_results.items()
            }
            for embedder, emb_results in all_embedder_results.items()
        },
        "robust_techniques": robust,
    }
    with open(output_dir / "robust_experiments_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/robust_experiments_*.json")


if __name__ == "__main__":
    main()
