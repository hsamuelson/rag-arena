"""Abstract hypothesis interface.

A hypothesis defines two transforms on RAG results:

1. rerank(results, embeddings, query_embedding) -> results
   Modify which documents are selected and in what order.

2. format_context(results) -> str
   Modify how retrieved documents are presented to the LLM.

The default (baseline) does neither — just cosine top-K in a flat list.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..backends.base import RetrievalResult


@dataclass
class HypothesisResult:
    """Output of a hypothesis application."""
    results: list[RetrievalResult]
    context_prompt: str
    metadata: dict = field(default_factory=dict)  # Hypothesis-specific data (e.g., PCA axes)


class Hypothesis(ABC):
    """Abstract hypothesis that transforms retrieval results."""

    @abstractmethod
    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        """Apply the hypothesis to retrieval results.

        Args:
            query: The original query text.
            results: Retrieved documents from the backend.
            embeddings: (K, D) embedding matrix for retrieved docs, or None.
            query_embedding: (D,) query embedding, or None.

        Returns:
            HypothesisResult with reranked results and formatted context.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this hypothesis."""

    @property
    def description(self) -> str:
        """What this hypothesis tests."""
        return ""
