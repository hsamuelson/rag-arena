#!/usr/bin/env python3
"""Run LoCoMo benchmark.

Requires:
    - Ollama running with nomic-embed-text and qwen3.5:122b
    - LoCoMo dataset (auto-downloads from HuggingFace on first run)

Usage:
    # Quick test with limited samples:
    python experiments/run_locomo_arena.py --max-samples 10

    # Full run:
    python experiments/run_locomo_arena.py

    # Retrieval-only (skip LLM answer generation):
    python experiments/run_locomo_arena.py --skip-llm
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "run",
            "--benchmark", "locomo",
            "--backend", "direct",
            "--hypotheses", "flat", "pca-diversity", "pca-grouped",
            "--max-samples", "20",
        ]
    main()
