"""Baseline: flat cosine top-K with no reranking or grouping.

This is the control condition — standard RAG as most systems implement it.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class FlatBaselineHypothesis(Hypothesis):
    """No reranking, flat ranked list presentation."""

    @property
    def name(self) -> str:
        return "flat-baseline"

    @property
    def description(self) -> str:
        return "Standard RAG: cosine top-K retrieval, flat ranked list prompt"

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        # No reranking — keep backend order
        context = self._format_flat(results)
        return HypothesisResult(results=results, context_prompt=context)

    def _format_flat(self, results: list[RetrievalResult]) -> str:
        """Format as a numbered list of retrieved passages."""
        lines = ["Retrieved context:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
