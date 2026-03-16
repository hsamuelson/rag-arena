"""Cross-encoder + retrieval score fusion hypothesis.

Instead of using cross-encoder scores alone, fuse them with the original
retrieval scores (BM25/RRF). The intuition: retrieval scores carry
complementary signal (lexical match, rank fusion) that CE alone misses.

Score = alpha * normalize(CE_score) + (1-alpha) * normalize(retrieval_score)
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


def _normalize(scores):
    """Min-max normalize to [0, 1]."""
    arr = np.array(scores, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


class CEScoreFusionHypothesis(Hypothesis):
    """Fuse cross-encoder reranking scores with original retrieval scores."""

    def __init__(self, alpha: float = 0.7, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._alpha = alpha
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-score-fusion-{self._alpha}"

    @property
    def description(self) -> str:
        return f"Fuse cross-encoder scores (weight={self._alpha}) with original retrieval scores"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        model = self._get_model()
        ce_scores = model.predict(pairs).tolist()

        retrieval_scores = [r.score for r in results]

        # Normalize both score sets to [0,1]
        ce_norm = _normalize(ce_scores)
        ret_norm = _normalize(retrieval_scores)

        # Fuse
        fused = self._alpha * ce_norm + (1 - self._alpha) * ret_norm

        ranked_indices = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (CE + retrieval score fusion):"]
        for i, idx in enumerate(ranked_indices, 1):
            r = results[idx]
            lines.append(f"\n[{i}] (fused: {fused[idx]:.4f}, ce: {ce_scores[idx]:.4f}, ret: {retrieval_scores[idx]:.3f})")
            lines.append(r.text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in ranked_indices],
                "fused_scores": [float(fused[i]) for i in ranked_indices],
                "alpha": self._alpha,
            },
        )
