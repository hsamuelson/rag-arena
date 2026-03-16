"""Direct embedding backend — in-memory cosine search via Ollama embeddings.

This is the simplest possible RAG backend: embed everything into memory
and do cosine top-K. Useful as a baseline and for quick hypothesis testing
without needing an external backend.
"""

import numpy as np
import requests

from .base import Backend, RetrievalResult
from ..config import ArenaConfig


class DirectEmbeddingBackend(Backend):
    """In-memory cosine similarity search using Ollama embeddings.

    No external database — just numpy arrays. Good for:
    - Baseline comparisons
    - Quick prototyping
    - Testing hypotheses without infrastructure
    """

    def __init__(self, config: ArenaConfig):
        self.config = config
        self._ollama_url = config.ollama.base_url
        self._embed_model = config.ollama.embed_model
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._corpus_embeddings: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "direct-embeddings"

    def ingest(self, documents: list[dict]) -> None:
        """Embed and store all documents in memory."""
        self._corpus_ids = [doc["id"] for doc in documents]
        self._corpus_texts = [doc["text"] for doc in documents]
        self._corpus_metadata = [doc.get("metadata", {}) for doc in documents]
        self._corpus_embeddings = self.embed_batch(self._corpus_texts)

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

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_batch([query])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed via Ollama /api/embed endpoint."""
        all_embeddings = []
        batch_size = 32

        # Ensure no empty strings (Ollama rejects them)
        texts = [t if t.strip() else " " for t in texts]

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = requests.post(
                    f"{self._ollama_url}/api/embed",
                    json={"model": self._embed_model, "input": batch},
                    timeout=self.config.ollama.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                all_embeddings.extend(data["embeddings"])
            except requests.exceptions.HTTPError:
                # Fall back to one-by-one embedding on batch failure
                for text in batch:
                    try:
                        resp2 = requests.post(
                            f"{self._ollama_url}/api/embed",
                            json={"model": self._embed_model, "input": [text]},
                            timeout=self.config.ollama.timeout,
                        )
                        resp2.raise_for_status()
                        all_embeddings.extend(resp2.json()["embeddings"])
                    except Exception:
                        # Use zero vector for completely unembeddable texts
                        if all_embeddings:
                            dim = len(all_embeddings[0])
                        else:
                            dim = self.config.ollama.embed_dimensions
                        all_embeddings.append([0.0] * dim)

        return np.array(all_embeddings, dtype=np.float32)

    def clear(self) -> None:
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadata = []
        self._corpus_embeddings = None
