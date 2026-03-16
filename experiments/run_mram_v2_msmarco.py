#!/usr/bin/env python3
"""Quick test: does MRAM-v2 (entity) help at 100K scale?

Runs only MRAM-v2 on MS MARCO 100K, compares against saved CE/v1.5 results.
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
from arena.benchmarks.ms_marco import MSMarcoBenchmark
from arena.metrics.scoring import ndcg_at_k, compute_scorecard
from arena.hypotheses.multi_resolution_v2 import MultiResolutionV2Hypothesis


def compute_per_query_ndcg(per_question_results, k=10):
    ndcgs = []
    for q in per_question_results:
        retrieved = q["retrieved_ids"][:k]
        relevant = q["relevant_ids"]
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    return np.array(ndcgs)


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"

    print("=" * 70)
    print("MRAM-v2 (entity) on MS MARCO 100K — just for giggles")
    print("=" * 70)

    # Load same corpus (same seed)
    print("\n### Loading MS MARCO (same seed=42)...")
    benchmark = MSMarcoBenchmark(max_corpus=100_000, max_queries=200, seed=42)
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:200]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # Ingest
    print(f"\n### Ingesting {len(corpus)} documents...")
    backend = HybridBackend(config)
    t0 = time.time()
    backend.ingest(corpus)
    print(f"  Ingestion: {time.time() - t0:.1f}s")

    # Run MRAM-v2 only
    from tqdm import tqdm

    mram_v2 = MultiResolutionV2Hypothesis()
    mram_v2.set_backend(backend)

    print(f"\n{'=' * 60}")
    print(f"  MRAM-v2 (entity) — 100K docs")
    print(f"{'=' * 60}")

    per_question = []
    start = time.time()
    for sample in tqdm(samples, desc="MRAM-v2 (entity)"):
        t0 = time.time()
        results, embeddings = backend.retrieve_with_embeddings(sample.question, config.top_k)
        query_emb = None
        try:
            query_emb = backend.embed_query(sample.question)
        except Exception:
            pass

        hyp_result = mram_v2.apply(sample.question, results, embeddings, query_emb)
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

    elapsed = time.time() - start
    ndcgs = compute_per_query_ndcg(per_question)

    print(f"\n  nDCG@10={ndcgs.mean():.4f}  ({elapsed:.1f}s)")

    # MRAM-specific stats
    unique_sent = sum(1 for q in per_question
                      if q["hypothesis_metadata"].get("unique_from_sentence", 0) > 0)
    unique_ent = sum(1 for q in per_question
                     if q["hypothesis_metadata"].get("unique_from_entity", 0) > 0)
    total_cands = np.mean([q["hypothesis_metadata"].get("total_candidates", 0)
                           for q in per_question])
    avg_ents = np.mean([len(q["hypothesis_metadata"].get("query_entities", []))
                        for q in per_question])
    print(f"  Avg candidates: {total_cands:.0f}")
    print(f"  Queries with unique sentence contributions: {unique_sent}/{len(per_question)}")
    print(f"  Queries with unique entity contributions: {unique_ent}/{len(per_question)}")
    print(f"  Avg query entities extracted: {avg_ents:.1f}")

    # Load saved results for comparison
    with open(output_dir / "mram_msmarco_raw.json") as f:
        saved = json.load(f)

    ce_ndcgs = np.array(saved["CE baseline"])
    v15_ndcgs = np.array(saved["MRAM-v1.5"])
    v2_ndcgs = ndcgs

    print(f"\n{'#' * 70}")
    print("COMPARISON")
    print(f"{'#' * 70}")
    print(f"  CE baseline:       {ce_ndcgs.mean():.4f}")
    print(f"  MRAM-v1.5:         {v15_ndcgs.mean():.4f}")
    print(f"  MRAM-v2 (entity):  {v2_ndcgs.mean():.4f}")

    # v2 vs v1.5
    diff = v2_ndcgs - v15_ndcgs
    same = (diff == 0).sum()
    better = (diff > 0).sum()
    worse = (diff < 0).sum()
    pct = (v2_ndcgs.mean() - v15_ndcgs.mean()) / v15_ndcgs.mean() * 100
    print(f"\n  v2 vs v1.5: {pct:+.2f}%")
    print(f"  Same: {same}, Better: {better}, Worse: {worse}")

    from scipy.stats import wilcoxon
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        stat, p = wilcoxon(nonzero)
        print(f"  Wilcoxon: p={p:.4f}")
    else:
        print(f"  All identical — no test needed")

    # Save combined raw
    saved["MRAM-v2 (entity)"] = v2_ndcgs.tolist()
    with open(output_dir / "mram_msmarco_raw.json", "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nSaved to {output_dir}/mram_msmarco_raw.json")


if __name__ == "__main__":
    main()
