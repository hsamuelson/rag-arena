"""Cross-encoder reranking hypothesis.

Uses a sentence-transformers CrossEncoder model to jointly score
query-document pairs through a transformer, then reranks by those scores.
This is the gold standard for reranking — unlike bi-encoders which embed
query and document separately, cross-encoders attend over the concatenated
pair, capturing fine-grained token-level interactions.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

# Approximate character limit for ~512 tokens.
_MAX_CHARS = 2000


class CrossEncoderRerankerHypothesis(Hypothesis):
    """Rerank retrieval results using a cross-encoder transformer model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._model_name = model_name
        self._model = None  # lazy-loaded

    def _get_model(self):
        """Lazy-load the CrossEncoder to avoid slow imports at startup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "cross-encoder"

    @property
    def description(self) -> str:
        return (
            "Rerank retrieved passages with a cross-encoder transformer "
            "that jointly attends over query-document pairs"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if not results:
            return HypothesisResult(
                results=[],
                context_prompt="Retrieved context:\n(no results)",
                metadata={"cross_encoder_scores": [], "original_scores": [], "rerank_order": []},
            )

        # Build query-document pairs, truncating long texts.
        pairs = [
            (query, r.text[:_MAX_CHARS]) for r in results
        ]

        model = self._get_model()
        ce_scores = model.predict(pairs).tolist()

        # Record original ordering info.
        original_scores = [r.score for r in results]

        # Sort by cross-encoder score (descending).
        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)

        reranked_results = [results[i] for i in ranked_indices]
        reranked_ce_scores = [ce_scores[i] for i in ranked_indices]

        # Build context prompt.
        context = self._format(reranked_results, reranked_ce_scores)

        return HypothesisResult(
            results=reranked_results,
            context_prompt=context,
            metadata={
                "cross_encoder_scores": reranked_ce_scores,
                "original_scores": [original_scores[i] for i in ranked_indices],
                "rerank_order": ranked_indices,
            },
        )

    @staticmethod
    def _format(results: list[RetrievalResult], ce_scores: list[float]) -> str:
        lines = ["Retrieved context (cross-encoder reranked):"]
        for i, (r, ce) in enumerate(zip(results, ce_scores), 1):
            lines.append(f"\n[{i}] (ce_score: {ce:.4f}, retrieval_score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
