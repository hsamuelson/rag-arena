"""Hybrid BM25 + Dense retrieval backend with Reciprocal Rank Fusion.

Combines lexical (BM25) and semantic (Ollama embeddings) retrieval using RRF,
giving the best of both sparse and dense approaches.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from .base import Backend, RetrievalResult
from .direct_embeddings import DirectEmbeddingBackend
from ..config import ArenaConfig


class HybridBackend(Backend):
    """Hybrid BM25 + dense cosine retrieval fused with RRF.

    On retrieve: fetches 2*top_k candidates from each sub-system,
    then fuses rankings with Reciprocal Rank Fusion and returns top_k.

    RRF formula: score(d) = sum(1 / (k + rank_i)) with k=60 by default.
    """

    def __init__(self, config: ArenaConfig, rrf_k: int = 60):
        self.config = config
        self._rrf_k = rrf_k

        # BM25 sub-system (managed inline to avoid extra config plumbing)
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._bm25: BM25Okapi | None = None

        # Dense sub-system (delegates to DirectEmbeddingBackend)
        self._dense = DirectEmbeddingBackend(config)

    @property
    def name(self) -> str:
        return "hybrid-bm25-dense"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def ingest(self, documents: list[dict]) -> None:
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
        self._dense.ingest(documents)

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
        # Map doc_ids back to corpus indices
        id_to_idx = {did: i for i, did in enumerate(self._corpus_ids)}
        return [(id_to_idx[r.doc_id], r.score) for r in results if r.doc_id in id_to_idx]

    def _rrf_fuse(
        self, ranked_lists: list[list[tuple[int, float]]], top_k: int
    ) -> list[int]:
        """Fuse multiple ranked lists with Reciprocal Rank Fusion.

        Each ranked list is a sequence of (doc_index, original_score).
        Returns doc indices sorted by fused RRF score.
        """
        fused_scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _orig_score) in enumerate(ranked, start=1):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (self._rrf_k + rank)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _score in sorted_docs[:top_k]]

    def _rrf_fuse_with_scores(
        self, ranked_lists: list[list[tuple[int, float]]], top_k: int
    ) -> list[tuple[int, float]]:
        """Like _rrf_fuse but also returns the fused RRF scores."""
        fused_scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _orig_score) in enumerate(ranked, start=1):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (self._rrf_k + rank)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

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

        # Gather dense embeddings for the fused top-K
        id_to_idx = {did: i for i, did in enumerate(self._corpus_ids)}
        indices = [id_to_idx[r.doc_id] for r in results]
        embeddings = self._dense._corpus_embeddings[indices]
        return results, embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query using the dense sub-system."""
        return self._dense.embed_query(query)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts using the dense sub-system."""
        return self._dense.embed_batch(texts)

    def clear(self) -> None:
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadata = []
        self._bm25 = None
        self._dense.clear()
