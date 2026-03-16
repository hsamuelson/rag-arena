#!/usr/bin/env python3
"""Example experiment: run all hypotheses on the synthetic benchmark.

Usage:
    # Quick retrieval-only test (no Ollama needed for LLM generation):
    python experiments/run_synthetic_arena.py --skip-llm --max-samples 20

    # Full run with hybrid backend:
    python experiments/run_synthetic_arena.py --backend hybrid

    # Compare specific hypotheses:
    python experiments/run_synthetic_arena.py -H flat pca-diversity --max-samples 50
"""

import sys
from pathlib import Path

# Add parent to path so we can import arena
sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.cli import main

if __name__ == "__main__":
    # Default args if none provided
    if len(sys.argv) == 1:
        sys.argv = [
            sys.argv[0],
            "run",
            "--benchmark", "synthetic",
            "--backend", "direct",
            "--hypotheses", "flat", "pca-diversity", "pca-grouped",
            "--skip-llm",
            "--max-samples", "30",
        ]
    main()
