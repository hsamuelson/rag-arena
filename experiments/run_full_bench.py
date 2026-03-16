#!/usr/bin/env python3
"""Run the full local benchmark suite.

This runs all 10 scaling-focused hypotheses against:
  1. Synthetic multi-hop (controllable, no download)
  2. HotpotQA (multi-hop, from HuggingFace)
  3. BEIR subset (SciFact + NFCorpus + FiQA, zero-shot)

Results are compared against published industry baselines:
  - BM25 on NQ:     59.1% top-20 accuracy
  - DPR on NQ:      78.4% top-20 accuracy
  - BM25 on BEIR:   ~0.437 avg nDCG@10
  - BM25 on HotpotQA: 0.603 nDCG@10

Usage:
    # Quick test (50 samples per benchmark, retrieval-only):
    python experiments/run_full_bench.py

    # Full run with LLM answer generation:
    python experiments/run_full_bench.py --full

    # Custom sample count:
    python experiments/run_full_bench.py --samples 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse


def main():
    parser = argparse.ArgumentParser(description="Full local benchmark suite")
    parser.add_argument("--samples", "-n", type=int, default=50)
    parser.add_argument("--full", action="store_true", help="Include LLM answer generation")
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/bench")
    args = parser.parse_args()

    # Build CLI args for the bench command
    cli_args = [
        "bench",
        "--max-samples", str(args.samples),
        "--output-dir", args.output_dir,
    ]
    if not args.full:
        cli_args.append("--skip-llm")
    if args.config:
        cli_args.extend(["--config", args.config])

    sys.argv = [sys.argv[0]] + cli_args

    from arena.cli import main as arena_main
    arena_main()


if __name__ == "__main__":
    main()
