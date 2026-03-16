#!/usr/bin/env python3
"""MRAM Phase 2: Multi-Resolution + Entity Association on FiQA.

Benchmarks MRAM against published BEIR baselines (not internal methods).

Compares:
  1. CE baseline (hybrid BM25+dense, top-10) — our internal baseline
  2. MRAM Phase 1.5 (sentence+passage, upgraded CE L-12, float32)
  3. MRAM Phase 2 (sentence+passage+entity association, CE L-12)

Published BEIR FiQA baselines for context:
  - BM25:      0.236 nDCG@10
  - DPR:       0.112 nDCG@10
  - E5-large:  0.370 nDCG@10
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
from arena.benchmarks.beir_subset import BEIRSubsetBenchmark
from arena.metrics.scoring import ndcg_at_k, compute_scorecard
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.multi_resolution import MultiResolutionHypothesis
from arena.hypotheses.multi_resolution_v2 import MultiResolutionV2Hypothesis

# Published BEIR baselines for FiQA (nDCG@10)
BEIR_BASELINES = {
    "BM25": 0.236,
    "DPR": 0.112,
    "E5-large": 0.370,
}


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


def run_hypothesis(name, hyp, samples, backend, config):
    from tqdm import tqdm

    per_question = []
    iterator = tqdm(samples, desc=name)

    for sample in iterator:
        t0 = time.time()
        results, embeddings = backend.retrieve_with_embeddings(sample.question, config.top_k)
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

    scorecard = compute_scorecard(per_question, k=config.top_k)
    return per_question, scorecard


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 200

    print("=" * 70)
    print("MRAM PHASE 2: Multi-Resolution + Entity Association")
    print("=" * 70)

    # Load FiQA
    print("\n### Loading FiQA...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # Ingest
    print(f"\n### Ingesting {len(corpus)} documents...")
    backend = HybridBackend(config)
    t0 = time.time()
    backend.ingest(corpus)
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    # Hypotheses — only CE baseline and MRAM variants (no DP50)
    ce = CrossEncoderRerankerHypothesis()
    mram_v15 = MultiResolutionHypothesis()
    mram_v15.set_backend(backend)
    mram_v2 = MultiResolutionV2Hypothesis()
    mram_v2.set_backend(backend)

    hypotheses = [
        ("CE baseline", ce),
        ("MRAM-v1.5", mram_v15),
        ("MRAM-v2 (entity)", mram_v2),
    ]

    results_data = {}
    for name, hyp in hypotheses:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        start = time.time()
        try:
            per_question, scorecard = run_hypothesis(name, hyp, samples, backend, config)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - start
        ndcgs = compute_per_query_ndcg(per_question)

        results_data[name] = {
            "per_question": per_question,
            "scorecard": scorecard,
            "ndcgs": ndcgs,
            "mean_ndcg": float(ndcgs.mean()),
            "elapsed": elapsed,
        }
        print(f"  nDCG@10={ndcgs.mean():.4f}  R@K={scorecard.recall_at_k:.3f}  "
              f"MRR={scorecard.mrr:.3f}  ({elapsed:.1f}s)")

        # Report MRAM-specific metadata
        if name.startswith("MRAM"):
            unique_sent = sum(
                1 for q in per_question
                if q["hypothesis_metadata"].get("unique_from_sentence", 0) > 0
            )
            total_cands = np.mean([
                q["hypothesis_metadata"].get("total_candidates", 0) for q in per_question
            ])
            print(f"  Avg candidates: {total_cands:.0f}")
            print(f"  Queries with unique sentence contributions: {unique_sent}/{len(per_question)}")

            if per_question and "unique_from_entity" in per_question[0].get("hypothesis_metadata", {}):
                unique_ent = sum(
                    1 for q in per_question
                    if q["hypothesis_metadata"].get("unique_from_entity", 0) > 0
                )
                avg_query_ents = np.mean([
                    len(q["hypothesis_metadata"].get("query_entities", []))
                    for q in per_question
                ])
                print(f"  Queries with unique entity contributions: {unique_ent}/{len(per_question)}")
                print(f"  Avg query entities extracted: {avg_query_ents:.1f}")

        # Save incrementally
        raw = {n: d["ndcgs"].tolist() for n, d in results_data.items()}
        with open(output_dir / "mram_phase2_raw.json", "w") as f:
            json.dump(raw, f, indent=2)
        print(f"  (saved incrementally)")

    # ─── Final Results vs Published BEIR Baselines ───
    print(f"\n{'#' * 70}")
    print("MRAM PHASE 2: RESULTS vs PUBLISHED BEIR BASELINES")
    print(f"{'#' * 70}")

    print(f"\n  Published baselines (FiQA nDCG@10):")
    for name, score in BEIR_BASELINES.items():
        print(f"    {name:20s}: {score:.4f}")

    print(f"\n  Our results:")
    for name, data in results_data.items():
        ndcg = data["mean_ndcg"]
        vs_e5 = (ndcg - BEIR_BASELINES["E5-large"]) / BEIR_BASELINES["E5-large"] * 100
        vs_bm25 = (ndcg - BEIR_BASELINES["BM25"]) / BEIR_BASELINES["BM25"] * 100
        print(f"    {name:20s}: {ndcg:.4f}  ({vs_e5:+.1f}% vs E5-large, {vs_bm25:+.1f}% vs BM25)")

    # Statistical tests: MRAM variants vs CE baseline
    ce_data = results_data.get("CE baseline")
    if ce_data:
        print(f"\n  Statistical comparisons vs CE baseline:")
        for name, data in results_data.items():
            if name == "CE baseline":
                continue
            diff = (data["mean_ndcg"] - ce_data["mean_ndcg"]) / ce_data["mean_ndcg"] * 100
            if data["mean_ndcg"] >= ce_data["mean_ndcg"]:
                p_val = paired_permutation_test(data["ndcgs"], ce_data["ndcgs"])
            else:
                p_val = paired_permutation_test(ce_data["ndcgs"], data["ndcgs"])
            ci = bootstrap_ci(data["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"    {name}: {diff:+.2f}% (p={p_val:.4f} {sig}), 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    # MRAM-v2 vs MRAM-v1.5 head-to-head
    v15_data = results_data.get("MRAM-v1.5")
    v2_data = results_data.get("MRAM-v2 (entity)")
    if v15_data and v2_data:
        diff = (v2_data["mean_ndcg"] - v15_data["mean_ndcg"]) / v15_data["mean_ndcg"] * 100
        if v2_data["mean_ndcg"] >= v15_data["mean_ndcg"]:
            p_val = paired_permutation_test(v2_data["ndcgs"], v15_data["ndcgs"])
        else:
            p_val = paired_permutation_test(v15_data["ndcgs"], v2_data["ndcgs"])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n  MRAM-v2 vs MRAM-v1.5: {diff:+.2f}% (p={p_val:.4f} {sig})")
        print(f"    Entity association {'HELPS' if diff > 0 and p_val < 0.05 else 'does not significantly help'}")

    # Save final results
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "mram_phase2_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "benchmark": "FiQA",
        "n_queries": max_samples,
        "beir_baselines": BEIR_BASELINES,
        "results": {
            name: {
                "ndcg_at_10": data["mean_ndcg"],
                "elapsed_s": data["elapsed"],
                "vs_e5_large_pct": (data["mean_ndcg"] - BEIR_BASELINES["E5-large"]) / BEIR_BASELINES["E5-large"] * 100,
            }
            for name, data in results_data.items()
        },
    }
    with open(output_dir / "mram_phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {output_dir}/mram_phase2_*.json")


if __name__ == "__main__":
    main()
