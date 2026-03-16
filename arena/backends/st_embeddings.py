"""SentenceTransformer embedding backend — drop-in replacement for DirectEmbeddingBackend.

Uses HuggingFace sentence-transformers models (e.g., Snowflake/snowflake-arctic-embed-l,
Alibaba-NLP/gte-large-en-v1.5) instead of Ollama for dense embeddings. Supports models
that require trust_remote_code=True (like GTE-Qwen2).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

from .base import Backend, RetrievalResult
from ..config import ArenaConfig


class STEmbeddingBackend(Backend):
    """In-memory cosine similarity search using sentence-transformers embeddings.

    Drop-in replacement for DirectEmbeddingBackend that uses a local
    SentenceTransformer model instead of the Ollama REST API.
    """

    def __init__(
        self,
        config: ArenaConfig,
        model_name: str | None = None,
        trust_remote_code: bool = True,
        batch_size: int = 64,
        device: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._model_name = model_name or config.st_embed_model or "Snowflake/snowflake-arctic-embed-l"
        self._batch_size = batch_size

        print(f"[ST] Loading model {self._model_name!r} ...", file=sys.stderr)
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=trust_remote_code,
            device=device,
        )
        self._dimensions = self._model.get_sentence_embedding_dimension()
        print(
            f"[ST] Model loaded — dimensions={self._dimensions}, device={self._model.device}",
            file=sys.stderr,
        )

        # Corpus storage (same attribute names as DirectEmbeddingBackend so
        # MRAM/LI hypotheses can access them directly).
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._corpus_embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        short = self._model_name.split("/")[-1] if "/" in self._model_name else self._model_name
        return f"st-{short}"

    @property
    def dimensions(self) -> int:
        """Auto-detected embedding dimensionality."""
        return self._dimensions

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[dict], cache_dir: str | Path | None = None) -> None:
        """Embed and store all documents in memory.

        If *cache_dir* is given, embeddings are saved/loaded as a .npy file
        keyed by model name + corpus hash so re-ingestion is instant.
        """
        self._corpus_ids = [doc["id"] for doc in documents]
        self._corpus_texts = [doc["text"] for doc in documents]
        self._corpus_metadata = [doc.get("metadata", {}) for doc in documents]

        cached = self._try_load_cache(cache_dir)
        if cached is not None:
            self._corpus_embeddings = cached
        else:
            self._corpus_embeddings = self.embed_batch(self._corpus_texts)
            self._save_cache(cache_dir)

    def _cache_path(self, cache_dir: str | Path | None) -> Path | None:
        if cache_dir is None:
            return None
        cache_dir = Path(cache_dir)
        # Hash model name + number of docs + first+last doc IDs for cache key
        key = f"{self._model_name}:{len(self._corpus_ids)}"
        if self._corpus_ids:
            key += f":{self._corpus_ids[0]}:{self._corpus_ids[-1]}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        short_model = self._model_name.split("/")[-1] if "/" in self._model_name else self._model_name
        return cache_dir / f"emb_cache_{short_model}_{h}.npy"

    def _try_load_cache(self, cache_dir: str | Path | None) -> np.ndarray | None:
        path = self._cache_path(cache_dir)
        if path is None or not path.exists():
            return None
        print(f"[ST] Loading cached embeddings from {path}", file=sys.stderr)
        arr = np.load(path)
        if arr.shape[0] != len(self._corpus_ids):
            print(f"[ST] Cache shape mismatch ({arr.shape[0]} vs {len(self._corpus_ids)}), re-embedding", file=sys.stderr)
            return None
        return arr

    def _save_cache(self, cache_dir: str | Path | None) -> None:
        path = self._cache_path(cache_dir)
        if path is None or self._corpus_embeddings is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._corpus_embeddings)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[ST] Saved embeddings cache ({size_mb:.0f} MB) to {path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        results, _ = self.retrieve_with_embeddings(query, top_k)
        return results

    def retrieve_with_embeddings(
        self, query: str, top_k: int = 10
    ) -> tuple[list[RetrievalResult], np.ndarray | None]:
        """Cosine top-K retrieval with embeddings returned."""
        if self._corpus_embeddings is None or len(self._corpus_ids) == 0:
            return [], None

        query_emb = self.embed_query(query)

        # Cosine similarity (normalised dot product)
        c_norms = np.linalg.norm(self._corpus_embeddings, axis=1, keepdims=True)
        c_norms = np.maximum(c_norms, 1e-12)
        q_norm = np.linalg.norm(query_emb)
        if q_norm < 1e-12:
            return [], None

        corpus_normed = self._corpus_embeddings / c_norms
        query_normed = query_emb / q_norm
        similarities = corpus_normed @ query_normed

        k = min(top_k, len(self._corpus_ids))
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                doc_id=self._corpus_ids[idx],
                text=self._corpus_texts[idx],
                score=float(similarities[idx]),
                metadata=self._corpus_metadata[idx],
            ))

        retrieved_embeddings = self._corpus_embeddings[top_indices]
        return results, retrieved_embeddings

    # ------------------------------------------------------------------
    # Embedding primitives
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text. Returns (D,) array."""
        return self.embed_batch([query])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using SentenceTransformer. Returns (N, D) array."""
        # Replace empty/whitespace-only strings (some models choke on them)
        texts = [t if t.strip() else " " for t in texts]

        show_progress = len(texts) > self._batch_size
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadata = []
        self._corpus_embeddings = None
