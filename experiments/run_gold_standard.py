#!/usr/bin/env python3
"""Gold-standard baseline benchmarks on HotpotQA.

Tests the REAL competition:
1. BM25 (lexical baseline — should get ~0.603 nDCG@10)
2. Dense (nomic-embed-text, our previous flat baseline)
3. Hybrid BM25+Dense with RRF
4. Dense + Cross-encoder reranking
5. Hybrid + Cross-encoder reranking
6. Dense + CSLS (our previous best)
7. Hybrid + CSLS

This establishes the true baseline that novel hypotheses must beat.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.backends.bm25_backend import BM25Backend
from arena.backends.direct_embeddings import DirectEmbeddingBackend
from arena.backends.hybrid_backend import HybridBackend
from arena.benchmarks.hotpotqa import HotpotQABenchmark
from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
from arena.hypotheses.csls_pure import PureCSLSHypothesis
from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from arena.hypotheses.anti_hubness import AntiHubnessHypothesis
from arena.runners.arena_runner import ArenaRunner


def run_configuration(name, backend, hypotheses, benchmark, runner, max_samples):
    """Run a single backend+hypotheses configuration."""
    print(f"\n{'='*70}")
    print(f"CONFIGURATION: {name}")
    print(f"Backend: {backend.name}")
    print(f"Hypotheses: {[h.name for h in hypotheses]}")
    print(f"{'='*70}")

    start = time.time()
    results = runner.run_arena(
        benchmark=benchmark,
        backend=backend,
        hypotheses=hypotheses,
        max_samples=max_samples,
        skip_llm=True,
        verbose=True,
    )
    elapsed = time.time() - start

    runner.print_comparison(results)
    print(f"\nConfiguration completed in {elapsed:.1f}s")
    return results


def main():
    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = 50
    benchmark = HotpotQABenchmark(max_samples=max_samples)
    runner = ArenaRunner(config)

    all_results = []

    # --- Config 1: BM25 only (flat) ---
    bm25 = BM25Backend(config)
    r = run_configuration(
        "BM25 (lexical baseline)",
        bm25, [FlatBaselineHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 2: Dense only (flat) ---
    dense = DirectEmbeddingBackend(config)
    r = run_configuration(
        "Dense nomic-embed-text (flat)",
        dense, [FlatBaselineHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 3: Dense + CSLS ---
    r = run_configuration(
        "Dense + CSLS (our previous best)",
        dense, [PureCSLSHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 4: Dense + Cross-encoder reranking ---
    r = run_configuration(
        "Dense + Cross-encoder reranking",
        dense, [CrossEncoderRerankerHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 5: Hybrid BM25+Dense (flat) ---
    hybrid = HybridBackend(config)
    r = run_configuration(
        "Hybrid BM25+Dense RRF (flat)",
        hybrid, [FlatBaselineHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 6: Hybrid + CSLS ---
    r = run_configuration(
        "Hybrid + CSLS",
        hybrid, [PureCSLSHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 7: Hybrid + Cross-encoder ---
    r = run_configuration(
        "Hybrid + Cross-encoder (full pipeline)",
        hybrid, [CrossEncoderRerankerHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Config 8: Hybrid + Anti-hubness ---
    r = run_configuration(
        "Hybrid + Anti-hubness",
        hybrid, [AntiHubnessHypothesis()],
        benchmark, runner, max_samples,
    )
    all_results.extend(r)

    # --- Grand comparison ---
    print(f"\n{'#'*70}")
    print("GOLD STANDARD COMPARISON")
    print(f"{'#'*70}")
    runner.print_comparison(all_results)

    # Save
    runner.save_results(all_results, output_dir / "gold_standard_hotpotqa.json")

    # Write markdown
    sorted_results = sorted(all_results, key=lambda r: r.scorecard.ndcg_at_k, reverse=True)
    lines = [
        "# Gold Standard Baselines: HotpotQA",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Benchmark:** HotpotQA (multi-hop, 50 samples)",
        "",
        "## Rankings (sorted by nDCG@10)",
        "",
        "| Rank | Backend + Hypothesis | nDCG@10 | MRR | R@K | Latency |",
        "|------|---------------------|---------|-----|-----|---------|",
    ]
    for rank, r in enumerate(sorted_results, 1):
        sc = r.scorecard
        lines.append(
            f"| {rank} | {r.backend_name} + {r.hypothesis_name} | "
            f"{sc.ndcg_at_k:.4f} | {sc.mrr:.4f} | {sc.recall_at_k:.4f} | {sc.latency_ms:.0f}ms |"
        )

    lines.extend([
        "",
        "## Published Baselines (for reference)",
        "| Method | HotpotQA nDCG@10 |",
        "|--------|-----------------|",
        "| BM25 (published) | 0.603 |",
        "| DPR | 0.391 |",
        "| ColBERTv2 | 0.667 |",
        "| E5-large-v2 | 0.633 |",
        "| SPLADE-v3 | 0.692 |",
        "| BM25+CE (published) | 0.707 |",
    ])

    with open(output_dir / "gold_standard_hotpotqa.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_dir}/gold_standard_hotpotqa.*")


if __name__ == "__main__":
    main()
