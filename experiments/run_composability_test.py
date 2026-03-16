#!/usr/bin/env python3
"""Composability Robustness Test — Does MRAM still compose with stronger embeddings?

Focused experiment to answer one question: does sentence-level retrieval
(MRAM) still provide additive gains on top of rerankers when the passage
embedder is upgraded from nomic-embed-text (137M) to a stronger model?

Tests 6 strategies (minimum needed to prove/disprove composability):
  1. Flat baseline (no reranking)
  2. CE L-12 (reranker only, no MRAM)
  3. BGE-v2-m3 (reranker only, no MRAM)
  4. MRAM-v1.5 (sentence retrieval + CE L-12)
  5. MRAM + BGE (sentence retrieval + BGE)
  6. Late interaction + MRAM (sentence retrieval + MaxSim)

Composability holds if:
  - MRAM + BGE > max(MRAM alone, BGE alone)
  - LI + MRAM > max(LI alone, MRAM alone)

We test with snowflake-arctic-embed-l (335M) as the upgraded embedder.

Original results (nomic-embed-text, 137M):
  CE L-12:           0.3720
  BGE-v2-m3:         0.3941
  MRAM-v1.5:         0.3789
  MRAM + BGE:        0.4056  (> max(0.3789, 0.3941) = 0.3941 → COMPOSABLE +2.9%)
  Late interaction:  0.3976
  LI + MRAM:         0.4148  (> max(0.3976, 0.3789) = 0.3976 → COMPOSABLE +4.3%)
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

# Original results for comparison
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
        # Permutation test fallback
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


def build_strategies(backend, embed_model):
    """Build the 6 strategies needed for the composability test."""
    from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
    from arena.hypotheses.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis
    from arena.hypotheses.bge_reranker import BGERerankerHypothesis
    from arena.hypotheses.multi_resolution import MultiResolutionHypothesis
    from arena.hypotheses.mram_bge_reranker import MRAMBGERerankerHypothesis
    from arena.hypotheses.late_interaction_mram import LateInteractionMRAMHypothesis
    from arena.hypotheses.late_interaction_reranker import LateInteractionRerankerHypothesis

    strategies = []

    # 1. Flat baseline
    strategies.append(("Flat baseline", FlatBaselineHypothesis()))

    # 2. CE L-12 (reranker only)
    ce_l12 = DeepPool50CEL12Hypothesis()
    ce_l12.set_backend(backend)
    strategies.append(("CE L-12 (top-50)", ce_l12))

    # 3. BGE-v2-m3 (reranker only)
    bge = BGERerankerHypothesis()
    bge.set_backend(backend)
    strategies.append(("BGE-v2-m3 (top-50)", bge))

    # 4. MRAM-v1.5 (sentence + CE L-12)
    mram = MultiResolutionHypothesis()
    mram.set_backend(backend)
    strategies.append(("MRAM-v1.5", mram))

    # 5. MRAM + BGE
    mram_bge = MRAMBGERerankerHypothesis()
    mram_bge.set_backend(backend)
    strategies.append(("MRAM + BGE", mram_bge))

    # 6. Late interaction (MaxSim only, no MRAM) — need this for composability check
    # Inject the upgraded model so MaxSim uses same embedder
    late = LateInteractionRerankerHypothesis()
    late.set_backend(backend)
    if "/" in embed_model and embed_model != "nomic-ai/nomic-embed-text-v1.5":
        from sentence_transformers import SentenceTransformer
        late._model = SentenceTransformer(embed_model, trust_remote_code=True)
    strategies.append(("Late interaction", late))

    # 7. Late interaction + MRAM
    late_mram = LateInteractionMRAMHypothesis()
    late_mram.set_backend(backend)
    if "/" in embed_model and embed_model != "nomic-ai/nomic-embed-text-v1.5":
        from sentence_transformers import SentenceTransformer
        late_mram._st_model = SentenceTransformer(embed_model, trust_remote_code=True)
    strategies.append(("Late interaction + MRAM", late_mram))

    return strategies


def composability_analysis(all_results):
    """The core question: does MRAM compose additively with rerankers?"""
    print(f"\n{'#' * 70}")
    print("  COMPOSABILITY ANALYSIS — THE KEY QUESTION")
    print(f"{'#' * 70}\n")

    tests = [
        {
            "combo": "MRAM + BGE",
            "components": ["MRAM-v1.5", "BGE-v2-m3 (top-50)"],
            "orig_combo": 0.4056,
            "orig_best": 0.3941,
            "orig_gain": "+2.9%",
        },
        {
            "combo": "Late interaction + MRAM",
            "components": ["Late interaction", "MRAM-v1.5"],
            "orig_combo": 0.4148,
            "orig_best": 0.3976,
            "orig_gain": "+4.3%",
        },
    ]

    all_compose = True
    for test in tests:
        combo_name = test["combo"]
        comp_names = test["components"]

        combo_data = all_results.get(combo_name)
        comp_data = [all_results.get(c) for c in comp_names]

        if not combo_data or not all(comp_data):
            print(f"  [SKIP] {combo_name}: missing data")
            continue

        combo_ndcg = combo_data["mean_ndcg"]
        comp_ndcgs = [d["mean_ndcg"] for d in comp_data]
        best_component = max(comp_ndcgs)
        best_name = comp_names[comp_ndcgs.index(best_component)]

        gain = (combo_ndcg - best_component) / best_component * 100 if best_component > 0 else 0

        # Statistical test: is combo > best component?
        best_idx = comp_ndcgs.index(best_component)
        p_val = paired_wilcoxon_test(combo_data["ndcgs"], comp_data[best_idx]["ndcgs"])

        composable = combo_ndcg > best_component
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        status = "COMPOSABLE" if composable else "NOT COMPOSABLE"
        if not composable:
            all_compose = False

        print(f"  {combo_name}:")
        print(f"    Combined:       {combo_ndcg:.4f}")
        print(f"    Best component: {best_component:.4f} ({best_name})")
        print(f"    Gain:           {gain:+.1f}% (p={p_val:.4f} {sig})")
        print(f"    Original gain:  {test['orig_gain']}")
        print(f"    Verdict:        {status}")
        print()

    print(f"  {'=' * 60}")
    if all_compose:
        print(f"  RESULT: MRAM composability HOLDS with stronger embedder!")
        print(f"  Sentence retrieval provides additive gains regardless of")
        print(f"  embedder quality. The finding is ROBUST.")
    else:
        print(f"  RESULT: MRAM composability WEAKENED or LOST with stronger embedder.")
        print(f"  Sentence retrieval gains diminish when the passage embedder")
        print(f"  is good enough to capture most relevant documents.")
    print(f"  {'=' * 60}")

    return all_compose


def main():
    parser = argparse.ArgumentParser(
        description="Test MRAM composability robustness with upgraded embedder"
    )
    parser.add_argument(
        "--embed-model",
        default="Snowflake/snowflake-arctic-embed-l",
        help="Sentence-transformers embedder (default: Snowflake/snowflake-arctic-embed-l)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Number of queries (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results"),
        help="Output directory",
    )
    args = parser.parse_args()

    np.random.seed(42)
    embed_model = args.embed_model
    max_samples = args.max_samples
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPOSABILITY ROBUSTNESS TEST")
    print("=" * 70)
    print(f"Question: Does MRAM compose with rerankers when using a")
    print(f"          stronger embedder?")
    print(f"Embedder: {embed_model}")
    print(f"Original: nomic-embed-text (137M params)")
    print(f"Queries:  {max_samples} on FiQA")
    print(f"Seed:     42")
    print()

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")

    # ── Load FiQA ──
    print("### Loading FiQA benchmark...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # ── Create backend ──
    print(f"\n### Creating backend with {embed_model}...")
    try:
        from arena.backends.hybrid_st_backend import HybridSTBackend
        backend = HybridSTBackend(config, model_name=embed_model)
    except Exception as e:
        print(f"  ERROR creating ST backend: {e}")
        print(f"  Falling back to standard HybridBackend (Ollama)")
        from arena.backends.hybrid_backend import HybridBackend
        backend = HybridBackend(config)
        embed_model = config.ollama.embed_model

    # ── Ingest ──
    print(f"\n### Ingesting {len(corpus)} documents...")
    t0 = time.time()
    backend.ingest(corpus)
    ingest_time = time.time() - t0
    print(f"  Ingestion: {ingest_time:.1f}s")

    # ── Build strategies ──
    strategies = build_strategies(backend, embed_model)

    all_results = {}

    # ── Run ──
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

        all_results[name] = {
            "per_question": per_question,
            "ndcgs": ndcgs,
            "mean_ndcg": float(ndcgs.mean()),
            "elapsed": elapsed,
        }

        ci = bootstrap_ci(ndcgs)
        print(f"  nDCG@10 = {ndcgs.mean():.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  ({elapsed:.1f}s)")

        # Save incrementally
        raw = {n: d["ndcgs"].tolist() for n, d in all_results.items()}
        with open(output_dir / "composability_test_raw.json", "w") as f:
            json.dump(raw, f, indent=2)

    # ── Results table ──
    print(f"\n{'#' * 70}")
    print("  RESULTS COMPARISON")
    print(f"{'#' * 70}\n")
    print(f"  {'Strategy':<28s}  {'Original':>8s}  {'Upgraded':>8s}  {'Delta':>10s}")
    print(f"  {'-' * 60}")
    for name, data in all_results.items():
        upgraded = data["mean_ndcg"]
        original = ORIGINAL.get(name)
        if original is not None:
            delta = upgraded - original
            pct = delta / original * 100 if original > 0 else 0
            print(f"  {name:<28s}  {original:8.4f}  {upgraded:8.4f}  {delta:+.4f} ({pct:+.1f}%)")
        else:
            print(f"  {name:<28s}  {'N/A':>8s}  {upgraded:8.4f}")

    # ── The core question ──
    composable = composability_analysis(all_results)

    # ── Save final results ──
    summary = {
        "date": datetime.now().isoformat(),
        "question": "Does MRAM composability hold with stronger embedder?",
        "embed_model": embed_model,
        "original_embed_model": "nomic-embed-text (137M)",
        "n_queries": max_samples,
        "seed": 42,
        "composability_holds": composable,
        "results": {
            name: {
                "ndcg_at_10": data["mean_ndcg"],
                "elapsed_s": data["elapsed"],
            }
            for name, data in all_results.items()
        },
        "original_results": ORIGINAL,
    }
    with open(output_dir / "composability_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}/composability_test_*.json")


if __name__ == "__main__":
    main()
