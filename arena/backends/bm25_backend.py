"""BM25 retrieval backend — in-memory lexical search via rank_bm25.

A sparse retrieval baseline using Okapi BM25. Useful for comparison against
dense embedding approaches and as a component in hybrid retrieval.
"""

import numpy as np
from rank_bm25 import BM25Okapi

from .base import Backend, RetrievalResult


class BM25Backend(Backend):
    """In-memory BM25 lexical retrieval.

    Tokenizes documents with simple whitespace + lowercase splitting,
    then scores with Okapi BM25. No embeddings are produced.
    """

    def __init__(self, config=None):
        self.config = config
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metadata: list[dict] = []
        self._bm25: BM25Okapi | None = None

    @property
    def name(self) -> str:
        return "bm25"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def ingest(self, documents: list[dict]) -> None:
        """Build BM25 index from documents."""
        self._corpus_ids = [doc["id"] for doc in documents]
        self._corpus_texts = [doc["text"] for doc in documents]
        self._corpus_metadata = [doc.get("metadata", {}) for doc in documents]

        if not documents:
            self._bm25 = None
            return

        tokenized_corpus = [self._tokenize(text) for text in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-K documents using BM25 scoring."""
        if self._bm25 is None or len(self._corpus_ids) == 0:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        k = min(top_k, len(self._corpus_ids))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                doc_id=self._corpus_ids[idx],
                text=self._corpus_texts[idx],
                score=float(scores[idx]),
                metadata=self._corpus_metadata[idx],
            ))
        return results

    def retrieve_with_embeddings(
        self, query: str, top_k: int = 10
    ) -> tuple[list[RetrievalResult], np.ndarray | None]:
        """Retrieve top-K documents. Returns None for embeddings (BM25 has none)."""
        return self.retrieve(query, top_k), None

    def embed_query(self, query: str) -> np.ndarray:
        """Not supported — BM25 has no embeddings. Returns empty array."""
        return np.array([], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Not supported — BM25 has no embeddings. Returns empty array."""
        return np.array([], dtype=np.float32)

    def clear(self) -> None:
        self._corpus_ids = []
        self._corpus_texts = []
        self._corpus_metadata = []
        self._bm25 = None
