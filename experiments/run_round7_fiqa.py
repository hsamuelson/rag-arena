#!/usr/bin/env python3
"""Round 7: Novel retrieval proposals on FiQA (57K docs).

Tests 6 genuinely novel hypotheses:
  Phase 1 (geometry-only, no LLM):
    1. Residual Query Retrieval
    2. Spectral Query Decomposition
    3. Embedding Gradient Ascent
    4. Void Detection
  Phase 2 (LLM-augmented):
    5. Relevance Field Estimation
    6. Contrastive Embedding Steering

Baselines: CE reranker, deep-pool-50

IMPORTANT: FiQA has ~57K docs. Embedding ingestion takes ~15 min.
We ingest ONCE and reuse the backend across all hypotheses.
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
from arena.hypotheses.residual_query import ResidualQueryHypothesis
from arena.hypotheses.spectral_query_decomp import SpectralQueryDecompHypothesis
from arena.hypotheses.embedding_gradient_ascent import EmbeddingGradientAscentHypothesis
from arena.hypotheses.void_detection import VoidDetectionHypothesis
from arena.hypotheses.relevance_field import RelevanceFieldHypothesis
from arena.hypotheses.contrastive_steering import ContrastiveSteeringHypothesis


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


def run_hypothesis_on_samples(name, hyp, samples, backend, config, verbose=True):
    """Run a single hypothesis against pre-ingested backend. No re-ingestion."""
    from tqdm import tqdm

    per_question = []
    iterator = tqdm(samples, desc=name, disable=not verbose)

    for sample in iterator:
        t0 = time.time()

        # Retrieve (standard top_k from config)
        results, embeddings = backend.retrieve_with_embeddings(
            sample.question, config.top_k
        )

        # Query embedding
        query_emb = None
        try:
            query_emb = backend.embed_query(sample.question)
        except Exception:
            pass

        # Apply hypothesis
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

    max_samples = 200  # Statistical power

    print("=" * 70)
    print("ROUND 7: Novel Retrieval Proposals on FiQA (57K docs)")
    print("=" * 70)

    # ─── Load benchmark (FiQA only) ───
    print("\n### Loading FiQA benchmark...")
    t0 = time.time()
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()
    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Queries: {len(samples)} total")
    if max_samples:
        samples = samples[:max_samples]
        print(f"  Using: {len(samples)} samples")
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ─── Ingest corpus ONCE ───
    print(f"\n### Ingesting {len(corpus)} documents into hybrid backend...")
    print("  This will embed all docs via Ollama — may take 10-20 min...")
    backend = HybridBackend(config)
    t0 = time.time()
    backend.ingest(corpus)
    ingest_time = time.time() - t0
    print(f"  Ingestion complete in {ingest_time:.1f}s")

    # ─── Define hypotheses ───
    hypotheses = []

    # Baselines
    ce_baseline = CrossEncoderRerankerHypothesis()
    hypotheses.append(("cross-encoder (baseline)", ce_baseline))

    dp50 = DeepPool50CEHypothesis()
    dp50.set_backend(backend)
    hypotheses.append(("deep-pool-50-ce (baseline)", dp50))

    # Phase 1: Geometry-only (no LLM)
    residual = ResidualQueryHypothesis()
    residual.set_backend(backend)
    hypotheses.append(("residual-query", residual))

    spectral = SpectralQueryDecompHypothesis()
    spectral.set_backend(backend)
    hypotheses.append(("spectral-query-decomp", spectral))

    gradient = EmbeddingGradientAscentHypothesis()
    gradient.set_backend(backend)
    hypotheses.append(("embedding-gradient-ascent", gradient))

    void = VoidDetectionHypothesis()
    void.set_backend(backend)
    hypotheses.append(("void-detection", void))

    # Phase 2: LLM-augmented
    relevance = RelevanceFieldHypothesis()
    relevance.set_backend(backend)
    hypotheses.append(("relevance-field", relevance))

    contrastive = ContrastiveSteeringHypothesis()
    contrastive.set_backend(backend)
    hypotheses.append(("contrastive-steering", contrastive))

    # ─── Run experiments ───
    results_data = {}

    for name, hyp in hypotheses:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        start = time.time()
        try:
            per_question, scorecard = run_hypothesis_on_samples(
                name, hyp, samples, backend, config, verbose=True
            )
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

        # Save incrementally
        _save_incremental(results_data, output_dir)

    # ─── Statistical comparison ───
    _print_and_save_comparison(results_data, output_dir, max_samples)


def _save_incremental(results_data, output_dir):
    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round7_fiqa_incremental.json", "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  [Saved incremental: {len(results_data)} hypotheses]")


def _print_and_save_comparison(results_data, output_dir, max_samples):
    print(f"\n{'#' * 70}")
    print("ROUND 7: STATISTICAL COMPARISON — FiQA (57K docs)")
    print(f"{'#' * 70}")

    baseline = results_data.get("cross-encoder (baseline)")
    dp50 = results_data.get("deep-pool-50-ce (baseline)")

    if not baseline:
        print("ERROR: CE baseline missing")
        return

    baseline_ndcgs = baseline["ndcgs"]

    print(f"\n{'Hypothesis':<30} {'nDCG':>7} {'95% CI':>22} {'vs CE':>8} {'vs DP50':>8} "
          f"{'p(vsCE)':>8} {'Sig':>5} {'R@K':>6} {'Time':>8}")
    print("-" * 115)

    md_lines = [
        "# Round 7: Novel Retrieval Proposals — FiQA (57K docs)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Samples:** {max_samples}",
        f"**Benchmark:** FiQA (BEIR) — 57K financial documents",
        f"**Backend:** Hybrid BM25+Dense (nomic-embed-text 768d)",
        f"**Stats:** Bootstrap 95% CI, paired permutation test (5000 perms)",
        "",
        "## Results (sorted by nDCG@10)",
        "",
        "| Rank | Hypothesis | nDCG@10 | 95% CI | vs CE | vs DP50 | p-value | Sig? | R@K | Time |",
        "|------|-----------|---------|--------|-------|---------|---------|------|-----|------|",
    ]

    sorted_names = sorted(results_data.keys(),
                          key=lambda n: results_data[n]["mean_ndcg"], reverse=True)

    for rank, name in enumerate(sorted_names, 1):
        data = results_data[name]
        ndcgs = data["ndcgs"]
        mean = ndcgs.mean()
        ci_lo, ci_hi = bootstrap_ci(ndcgs)
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        time_str = f"{data['elapsed']:.0f}s"
        recall = data["scorecard"].recall_at_k

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

        # vs deep-pool-50
        if dp50 and name != "deep-pool-50-ce (baseline)":
            dp_ndcgs = dp50["ndcgs"]
            dp_diff = (mean - dp_ndcgs.mean()) / dp_ndcgs.mean() * 100
            diff_dp50 = f"{dp_diff:+.1f}%"
        elif name == "deep-pool-50-ce (baseline)":
            diff_dp50 = "baseline"
        else:
            diff_dp50 = "N/A"

        print(f"{name:<30} {mean:>7.4f} {ci_str:>22} {diff_ce:>8} {diff_dp50:>8} "
              f"{p_str:>8} {sig_str:>5} {recall:>6.3f} {time_str:>8}")
        md_lines.append(
            f"| {rank} | {name} | {mean:.4f} | {ci_str} | {diff_ce} | {diff_dp50} | "
            f"{p_str} | {sig_str} | {recall:.3f} | {time_str} |"
        )

    md_lines.extend([
        "",
        "## Novel Hypotheses Tested",
        "",
        "1. **Residual Query** — Project out covered subspace, boost docs matching residual",
        "2. **Spectral Query Decomp** — Eigendecompose doc similarity, retrieve per facet",
        "3. **Embedding Gradient Ascent** — Step query toward high-CE regions of embedding space",
        "4. **Void Detection** — Probe sparse regions between query and top results",
        "5. **Relevance Field** — Sparse LLM scoring → k-NN interpolation over embedding space",
        "6. **Contrastive Steering** — LLM positive/negative descriptions → steering vector",
        "",
        "## Key Questions",
        "",
        "1. Do geometric methods (1-4) beat deep-pool-50 on a larger corpus?",
        "2. Do LLM methods (5-6) justify their cost on FiQA?",
        "3. Does the corpus size (57K vs 491) change which approaches win?",
    ])

    with open(output_dir / "round7_fiqa.md", "w") as f:
        f.write("\n".join(md_lines))

    raw = {name: data["ndcgs"].tolist() for name, data in results_data.items()}
    with open(output_dir / "round7_fiqa_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\nResults saved to {output_dir}/round7_fiqa.*")


if __name__ == "__main__":
    main()
