"""Hybrid BM25 + SentenceTransformer dense retrieval with Reciprocal Rank Fusion.

Drop-in replacement for HybridBackend that uses a sentence-transformers model
for the dense component instead of Ollama embeddings.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from .base import Backend, RetrievalResult
from .st_embeddings import STEmbeddingBackend
from ..config import ArenaConfig


class HybridSTBackend(Backend):
    """Hybrid BM25 + SentenceTransformer dense retrieval fused with RRF.

    Identical to HybridBackend but swaps DirectEmbeddingBackend for
    STEmbeddingBackend so that all dense work is done locally via
    sentence-transformers instead of the Ollama REST API.
    """

    def __init__(
        self,
        config: ArenaConfig,
        rrf_k: int = 60,
        model_name: str | None = None,
        trust_remote_code: bool = True,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.config = config
        self._rrf_k = rrf_k

        # BM25 sub-system
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._bm25: BM25Okapi | None = None

        # Dense sub-system (SentenceTransformer instead of Ollama)
        self._dense = STEmbeddingBackend(
            config,
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            batch_size=batch_size,
            device=device,
        )

    @property
    def name(self) -> str:
        return f"hybrid-bm25-{self._dense.name}"

    # Expose corpus embeddings through the dense backend so hypotheses
    # that access _corpus_embeddings on the hybrid backend still work.
    @property
    def _corpus_embeddings(self) -> np.ndarray | None:
        return self._dense._corpus_embeddings

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[dict], cache_dir: str | None = None) -> None:
        """Build both BM25 and dense embedding indices."""
        self._corpus_ids = [doc["id"] for doc in documents]
        self._corpus_texts = [doc["text"] for doc in documents]
        self._corpus_metadata = [doc.get("metadata", {}) for doc in documents]

        # BM25 index
        if documents:
            tokenized_corpus = [self._tokenize(text) for text in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

        # Dense index
        self._dense.ingest(documents, cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # Sub-retrieval helpers
    # ------------------------------------------------------------------

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (index, score) pairs from BM25."""
        if self._bm25 is None or len(self._corpus_ids) == 0:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        k = min(top_k, len(self._corpus_ids))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _dense_retrieve(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return (index, score) pairs from dense retrieval."""
        results = self._dense.retrieve(query, top_k)
        id_to_idx = {did: i for i, did in enumerate(self._corpus_ids)}
        return [(id_to_idx[r.doc_id], r.score) for r in results if r.doc_id in id_to_idx]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse_with_scores(
        self, ranked_lists: list[list[tuple[int, float]]], top_k: int
    ) -> list[tuple[int, float]]:
        """Fuse multiple ranked lists with Reciprocal Rank Fusion."""
        fused_scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _orig_score) in enumerate(ranked, start=1):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (self._rrf_k + rank)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve with RRF fusion of BM25 and dense results."""
        if len(self._corpus_ids) == 0:
            return []

        candidates_k = min(2 * top_k, len(self._corpus_ids))

        bm25_ranked = self._bm25_retrieve(query, candidates_k)
        dense_ranked = self._dense_retrieve(query, candidates_k)

        fused = self._rrf_fuse_with_scores([bm25_ranked, dense_ranked], top_k)

        results = []
        for doc_idx, rrf_score in fused:
            results.append(RetrievalResult(
                doc_id=self._corpus_ids[doc_idx],
                text=self._corpus_texts[doc_idx],
                score=rrf_score,
                metadata=self._corpus_metadata[doc_idx],
            ))
        return results

    def retrieve_with_embeddings(
        self, query: str, top_k: int = 10
    ) -> tuple[list[RetrievalResult], np.ndarray | None]:
        """Retrieve with RRF fusion and return dense embeddings for the top results."""
        results = self.retrieve(query, top_k)
        if not results or self._dense._corpus_embeddings is None:
            return results, None

        id_to_idx = {did: i for i, did in enumerate(self._corpus_ids)}
        indices = [id_to_idx[r.doc_id] for r in results]
        embeddings = self._dense._corpus_embeddings[indices]
        return results, embeddings

    # ------------------------------------------------------------------
    # Embedding pass-through
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the dense sub-system."""
        return self._dense.embed_query(query)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using the dense sub-system."""
        return self._dense.embed_batch(texts)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadata = []
        self._bm25 = None
        self._dense.clear()
