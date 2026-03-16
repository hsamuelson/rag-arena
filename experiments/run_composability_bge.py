#!/usr/bin/env python3
"""Composability test using BGE-small-en-v1.5 (33M, 384d) — fast CPU-friendly model.

Tests composability with a different (non-nomic) embedder to validate that
MRAM + reranker > max(MRAM, reranker) holds across embedding models.
"""

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

ORIGINAL = {
    "Flat baseline":          0.2656,
    "CE L-12 (top-50)":       0.3720,
    "BGE-v2-m3 (top-50)":     0.3941,
    "MRAM-v1.5":              0.3789,
    "MRAM + BGE":             0.4056,
    "Late interaction":       0.3976,
    "Late interaction + MRAM": 0.4148,
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


def main():
    np.random.seed(42)
    embed_model = "BAAI/bge-small-en-v1.5"
    max_samples = 200
    cache_dir = str(Path(__file__).parent / "cache")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPOSABILITY TEST — BGE-small-en-v1.5")
    print("=" * 70)
    print(f"Embedder:  {embed_model} (33M, 384d)")
    print(f"Queries:   {max_samples} on FiQA")
    print()

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")

    print("### Loading FiQA benchmark...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    print(f"\n### Creating backend with {embed_model}...")
    from arena.backends.hybrid_st_backend import HybridSTBackend
    backend = HybridSTBackend(config, model_name=embed_model)

    print(f"\n### Ingesting {len(corpus)} documents (cache: {cache_dir})...")
    t0 = time.time()
    backend.ingest(corpus, cache_dir=cache_dir)
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    # Build all strategies
    from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
    from arena.hypotheses.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis
    from arena.hypotheses.bge_reranker import BGERerankerHypothesis
    from arena.hypotheses.multi_resolution import MultiResolutionHypothesis
    from arena.hypotheses.mram_bge_reranker import MRAMBGERerankerHypothesis
    from arena.hypotheses.late_interaction_reranker import LateInteractionRerankerHypothesis
    from arena.hypotheses.late_interaction_mram import LateInteractionMRAMHypothesis
    from sentence_transformers import SentenceTransformer

    strategies = []

    strategies.append(("Flat baseline", FlatBaselineHypothesis()))

    ce = DeepPool50CEL12Hypothesis()
    ce.set_backend(backend)
    strategies.append(("CE L-12 (top-50)", ce))

    bge = BGERerankerHypothesis()
    bge.set_backend(backend)
    strategies.append(("BGE-v2-m3 (top-50)", bge))

    mram = MultiResolutionHypothesis()
    mram.set_backend(backend)
    strategies.append(("MRAM-v1.5", mram))

    mram_bge = MRAMBGERerankerHypothesis()
    mram_bge.set_backend(backend)
    strategies.append(("MRAM + BGE", mram_bge))

    late = LateInteractionRerankerHypothesis()
    late.set_backend(backend)
    late._model = SentenceTransformer(embed_model, trust_remote_code=True)
    strategies.append(("Late interaction", late))

    late_mram = LateInteractionMRAMHypothesis()
    late_mram.set_backend(backend)
    late_mram._st_model = SentenceTransformer(embed_model, trust_remote_code=True)
    strategies.append(("Late interaction + MRAM", late_mram))

    # Check for existing results
    raw_path = output_dir / "composability_bge_small_raw.json"
    existing = {}
    if raw_path.exists():
        with open(raw_path) as f:
            existing = json.load(f)

    all_ndcgs = {k: np.array(v) for k, v in existing.items()}

    for name, hyp in strategies:
        if name in all_ndcgs:
            ci = bootstrap_ci(all_ndcgs[name])
            print(f"\n  [CACHED] {name}: nDCG@10 = {all_ndcgs[name].mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Running: {name}")
        print(f"{'=' * 60}")

        start = time.time()
        try:
            per_question, scorecard = run_hypothesis(
                name, hyp, samples, backend, config.top_k
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - start
        ndcgs = compute_per_query_ndcg(per_question)
        all_ndcgs[name] = ndcgs

        ci = bootstrap_ci(ndcgs)
        print(f"  nDCG@10 = {ndcgs.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  ({elapsed:.1f}s)")

        # Save incrementally
        raw = {n: d.tolist() for n, d in all_ndcgs.items()}
        with open(raw_path, "w") as f:
            json.dump(raw, f, indent=2)

    # Results table
    print(f"\n{'#' * 70}")
    print("  RESULTS COMPARISON (BGE-small-en-v1.5 vs Original nomic-embed-text)")
    print(f"{'#' * 70}\n")
    print(f"  {'Strategy':<28s}  {'Original':>8s}  {'BGE-small':>8s}  {'Delta':>10s}")
    print(f"  {'-' * 60}")
    for name, ndcgs in all_ndcgs.items():
        upgraded = float(ndcgs.mean())
        original = ORIGINAL.get(name)
        if original is not None:
            delta = upgraded - original
            pct = delta / original * 100 if original > 0 else 0
            print(f"  {name:<28s}  {original:8.4f}  {upgraded:8.4f}  {delta:+.4f} ({pct:+.1f}%)")

    # Composability
    print(f"\n{'#' * 70}")
    print("  COMPOSABILITY ANALYSIS")
    print(f"{'#' * 70}\n")

    tests = [
        {"combo": "MRAM + BGE", "components": ["MRAM-v1.5", "BGE-v2-m3 (top-50)"]},
        {"combo": "Late interaction + MRAM", "components": ["Late interaction", "MRAM-v1.5"]},
    ]

    for test in tests:
        combo_name = test["combo"]
        comp_names = test["components"]

        if combo_name not in all_ndcgs or not all(c in all_ndcgs for c in comp_names):
            print(f"  [SKIP] {combo_name}: missing data")
            continue

        combo = all_ndcgs[combo_name]
        comps = [all_ndcgs[c] for c in comp_names]
        combo_mean = float(combo.mean())
        comp_means = [float(a.mean()) for a in comps]
        best = max(comp_means)
        best_name = comp_names[comp_means.index(best)]

        gain = (combo_mean - best) / best * 100 if best > 0 else 0
        p_val = paired_wilcoxon_test(combo, comps[comp_means.index(best)])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        status = "COMPOSABLE" if combo_mean > best else "NOT COMPOSABLE"

        print(f"  {combo_name}:")
        print(f"    Combined:       {combo_mean:.4f}")
        print(f"    Best component: {best:.4f} ({best_name})")
        print(f"    Gain:           {gain:+.1f}% (p={p_val:.4f} {sig})")
        print(f"    Verdict:        {status}")
        print()

    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "embed_model": embed_model,
        "n_queries": max_samples,
        "results": {n: {"ndcg_at_10": float(d.mean())} for n, d in all_ndcgs.items()},
    }
    with open(output_dir / "composability_bge_small_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/composability_bge_small_*.json")


if __name__ == "__main__":
    main()
