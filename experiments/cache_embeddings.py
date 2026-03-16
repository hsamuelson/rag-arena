#!/usr/bin/env python3
"""Pre-compute and cache corpus embeddings to disk for composability experiments."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.benchmarks.beir_subset import BEIRSubsetBenchmark


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Snowflake/snowflake-arctic-embed-l"
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")

    print(f"### Loading FiQA benchmark...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    print(f"  Corpus: {len(corpus)} docs")

    print(f"\n### Loading model {model_name}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, trust_remote_code=True)
    dims = model.get_sentence_embedding_dimension()
    print(f"  Dimensions: {dims}, Device: {model.device}")

    texts = [doc["text"] for doc in corpus]
    doc_ids = [doc["id"] for doc in corpus]

    # Check if cache exists
    import hashlib
    key = f"{model_name}:{len(doc_ids)}:{doc_ids[0]}:{doc_ids[-1]}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    short = model_name.split("/")[-1] if "/" in model_name else model_name
    cache_path = cache_dir / f"emb_cache_{short}_{h}.npy"

    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(doc_ids):
            print(f"\n  Cache already exists: {cache_path} ({arr.shape})")
            return

    print(f"\n### Embedding {len(texts)} documents...")
    texts = [t if t.strip() else " " for t in texts]

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    elapsed = time.time() - t0
    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"  Embedding took {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Shape: {embeddings.shape}")

    np.save(cache_path, embeddings)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"  Saved to {cache_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
