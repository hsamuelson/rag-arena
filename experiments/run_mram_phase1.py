#!/usr/bin/env python3
"""MRAM Phase 1: Multi-Resolution retrieval on FiQA.

Tests whether retrieving at sentence + passage + topic level
simultaneously beats single-resolution retrieval.

Compares:
  1. CE baseline (passage only, top-10)
  2. Deep-pool-50 (passage only, top-50 → CE → top-10)
  3. MRAM Phase 1 (sentence + passage + topic → merge → CE → top-10)

Note: MRAM builds sentence-level index on first use (~20-30 min for 57K docs).
After that, per-query speed should be comparable.
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
from arena.hypotheses.deep_pool_50_ce import DeepPool50CEHypothesis
from arena.hypotheses.multi_resolution import MultiResolutionHypothesis


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
    print("MRAM PHASE 1: Multi-Resolution Retrieval on FiQA")
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

    # Hypotheses
    ce = CrossEncoderRerankerHypothesis()
    dp50 = DeepPool50CEHypothesis()
    dp50.set_backend(backend)
    mram = MultiResolutionHypothesis()
    mram.set_backend(backend)

    hypotheses = [
        ("cross-encoder (baseline)", ce),
        ("deep-pool-50 (baseline)", dp50),
        ("MRAM-phase1", mram),
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

        # Save incrementally after each hypothesis
        raw = {n: d["ndcgs"].tolist() for n, d in results_data.items()}
        with open(output_dir / "mram_phase1_raw.json", "w") as f:
            json.dump(raw, f, indent=2)
        print(f"  (saved incrementally)")

        # For MRAM, report level contribution
        if name == "MRAM-phase1":
            unique_sent = sum(
                1 for q in per_question
                if q["hypothesis_metadata"].get("unique_from_sentence", 0) > 0
            )
            unique_topic = sum(
                1 for q in per_question
                if q["hypothesis_metadata"].get("unique_from_topic", 0) > 0
            )
            total_cands = np.mean([
                q["hypothesis_metadata"].get("total_candidates", 0) for q in per_question
            ])
            print(f"  Avg candidates: {total_cands:.0f}")
            print(f"  Queries with unique sentence contributions: {unique_sent}/{len(per_question)}")
            print(f"  Queries with unique topic contributions: {unique_topic}/{len(per_question)}")

    # Statistical comparison
    print(f"\n{'#' * 70}")
    print("MRAM PHASE 1: RESULTS")
    print(f"{'#' * 70}")

    baseline = results_data.get("cross-encoder (baseline)")
    dp50_data = results_data.get("deep-pool-50 (baseline)")
    mram_data = results_data.get("MRAM-phase1")

    if baseline and dp50_data and mram_data:
        b_ndcg = baseline["ndcgs"]
        d_ndcg = dp50_data["ndcgs"]
        m_ndcg = mram_data["ndcgs"]

        print(f"\n  CE baseline:  {b_ndcg.mean():.4f}")
        print(f"  Deep-pool-50: {d_ndcg.mean():.4f} ({(d_ndcg.mean()-b_ndcg.mean())/b_ndcg.mean()*100:+.1f}% vs CE)")
        print(f"  MRAM Phase 1: {m_ndcg.mean():.4f} ({(m_ndcg.mean()-b_ndcg.mean())/b_ndcg.mean()*100:+.1f}% vs CE)")

        # MRAM vs deep-pool-50
        diff_dp50 = (m_ndcg.mean() - d_ndcg.mean()) / d_ndcg.mean() * 100
        if m_ndcg.mean() >= d_ndcg.mean():
            p_val = paired_permutation_test(m_ndcg, d_ndcg)
        else:
            p_val = paired_permutation_test(d_ndcg, m_ndcg)
        ci = bootstrap_ci(m_ndcg)

        print(f"\n  MRAM vs deep-pool-50: {diff_dp50:+.2f}%  p={p_val:.4f}")
        print(f"  MRAM 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

        if m_ndcg.mean() > d_ndcg.mean() and p_val < 0.05:
            print(f"\n  VERDICT: MRAM Phase 1 BEATS deep-pool-50 (statistically significant)")
        elif m_ndcg.mean() > d_ndcg.mean():
            print(f"\n  VERDICT: MRAM Phase 1 slightly better but NOT statistically significant")
        elif m_ndcg.mean() < d_ndcg.mean() and p_val < 0.05:
            print(f"\n  VERDICT: MRAM Phase 1 is WORSE than deep-pool-50 (significant)")
        else:
            print(f"\n  VERDICT: No significant difference between MRAM and deep-pool-50")

    # Save
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "mram_phase1_raw.json", "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nSaved to {output_dir}/mram_phase1_raw.json")


if __name__ == "__main__":
    main()
