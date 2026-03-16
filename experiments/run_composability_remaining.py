#!/usr/bin/env python3
"""Run only the 3 missing composability strategies (MRAM+BGE, LI, LI+MRAM).

Picks up where run_composability_test.py left off — reuses the 4 completed
strategies from composability_test_raw.json and runs only the missing ones.
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

# Load existing results
RESULTS_DIR = Path(__file__).parent / "results"
RAW_PATH = RESULTS_DIR / "composability_test_raw.json"

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
    embed_model = "Snowflake/snowflake-arctic-embed-l"
    max_samples = 200

    # Load existing raw results
    existing_raw = {}
    if RAW_PATH.exists():
        with open(RAW_PATH) as f:
            existing_raw = json.load(f)

    completed = list(existing_raw.keys())
    missing = [n for n in ["MRAM + BGE", "Late interaction", "Late interaction + MRAM"]
               if n not in existing_raw]

    print("=" * 70)
    print("COMPOSABILITY TEST — REMAINING STRATEGIES")
    print("=" * 70)
    print(f"Completed: {completed}")
    print(f"Missing:   {missing}")
    print(f"Embedder:  {embed_model}")
    print()

    if not missing:
        print("All strategies already completed!")
        return

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")

    # Load FiQA
    print("### Loading FiQA benchmark...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # Create backend
    print(f"\n### Creating backend with {embed_model}...")
    from arena.backends.hybrid_st_backend import HybridSTBackend
    backend = HybridSTBackend(config, model_name=embed_model)

    # Ingest (with embedding cache to avoid re-computing)
    cache_dir = Path(__file__).parent / "cache"
    print(f"\n### Ingesting {len(corpus)} documents (cache: {cache_dir})...")
    t0 = time.time()
    backend.ingest(corpus, cache_dir=str(cache_dir))
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    # Build only missing strategies
    strategies = []

    if "MRAM + BGE" in missing:
        from arena.hypotheses.mram_bge_reranker import MRAMBGERerankerHypothesis
        mram_bge = MRAMBGERerankerHypothesis()
        mram_bge.set_backend(backend)
        strategies.append(("MRAM + BGE", mram_bge))

    if "Late interaction" in missing:
        from arena.hypotheses.late_interaction_reranker import LateInteractionRerankerHypothesis
        from sentence_transformers import SentenceTransformer
        late = LateInteractionRerankerHypothesis()
        late.set_backend(backend)
        late._model = SentenceTransformer(embed_model, trust_remote_code=True)
        strategies.append(("Late interaction", late))

    if "Late interaction + MRAM" in missing:
        from arena.hypotheses.late_interaction_mram import LateInteractionMRAMHypothesis
        from sentence_transformers import SentenceTransformer
        late_mram = LateInteractionMRAMHypothesis()
        late_mram.set_backend(backend)
        late_mram._st_model = SentenceTransformer(embed_model, trust_remote_code=True)
        strategies.append(("Late interaction + MRAM", late_mram))

    # Run missing strategies
    all_ndcgs = {k: np.array(v) for k, v in existing_raw.items()}

    for name, hyp in strategies:
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
        with open(RAW_PATH, "w") as f:
            json.dump(raw, f, indent=2)

    # Results table
    print(f"\n{'#' * 70}")
    print("  RESULTS COMPARISON")
    print(f"{'#' * 70}\n")
    print(f"  {'Strategy':<28s}  {'Original':>8s}  {'Upgraded':>8s}  {'Delta':>10s}")
    print(f"  {'-' * 60}")
    for name, ndcgs in all_ndcgs.items():
        upgraded = float(ndcgs.mean())
        original = ORIGINAL.get(name)
        if original is not None:
            delta = upgraded - original
            pct = delta / original * 100 if original > 0 else 0
            print(f"  {name:<28s}  {original:8.4f}  {upgraded:8.4f}  {delta:+.4f} ({pct:+.1f}%)")
        else:
            print(f"  {name:<28s}  {'N/A':>8s}  {upgraded:8.4f}")

    # Composability analysis
    print(f"\n{'#' * 70}")
    print("  COMPOSABILITY ANALYSIS")
    print(f"{'#' * 70}\n")

    tests = [
        {
            "combo": "MRAM + BGE",
            "components": ["MRAM-v1.5", "BGE-v2-m3 (top-50)"],
        },
        {
            "combo": "Late interaction + MRAM",
            "components": ["Late interaction", "MRAM-v1.5"],
        },
    ]

    for test in tests:
        combo_name = test["combo"]
        comp_names = test["components"]

        if combo_name not in all_ndcgs or not all(c in all_ndcgs for c in comp_names):
            print(f"  [SKIP] {combo_name}: missing data")
            continue

        combo_ndcgs = all_ndcgs[combo_name]
        comp_ndcg_arrays = [all_ndcgs[c] for c in comp_names]
        combo_mean = float(combo_ndcgs.mean())
        comp_means = [float(a.mean()) for a in comp_ndcg_arrays]
        best_component = max(comp_means)
        best_name = comp_names[comp_means.index(best_component)]

        gain = (combo_mean - best_component) / best_component * 100 if best_component > 0 else 0
        best_idx = comp_means.index(best_component)
        p_val = paired_wilcoxon_test(combo_ndcgs, comp_ndcg_arrays[best_idx])

        composable = combo_mean > best_component
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        status = "COMPOSABLE" if composable else "NOT COMPOSABLE"

        print(f"  {combo_name}:")
        print(f"    Combined:       {combo_mean:.4f}")
        print(f"    Best component: {best_component:.4f} ({best_name})")
        print(f"    Gain:           {gain:+.1f}% (p={p_val:.4f} {sig})")
        print(f"    Verdict:        {status}")
        print()

    # Save final summary
    summary = {
        "date": datetime.now().isoformat(),
        "question": "Does MRAM composability hold with stronger embedder?",
        "embed_model": embed_model,
        "original_embed_model": "nomic-embed-text (137M)",
        "n_queries": max_samples,
        "seed": 42,
        "results": {
            name: {"ndcg_at_10": float(ndcgs.mean())}
            for name, ndcgs in all_ndcgs.items()
        },
        "original_results": ORIGINAL,
    }
    with open(RESULTS_DIR / "composability_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/composability_test_*.json")


if __name__ == "__main__":
    main()
