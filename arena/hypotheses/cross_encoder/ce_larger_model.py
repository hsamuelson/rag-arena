"""Cross-encoder with a larger model.

Our default (ms-marco-MiniLM-L-6-v2, 22M params) is a small model.
Larger cross-encoders are known to produce better rankings.

This tests:
- ms-marco-MiniLM-L-12-v2 (33M params, deeper)
- ms-marco-electra-base (110M params, different architecture)

The hypothesis is that model capacity is the bottleneck, not scoring strategy.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CELargerModelHypothesis(Hypothesis):
    """Cross-encoder with a larger/better model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self._model_name = model_name
        self._model = None
        # Derive short name from model path
        self._short_name = model_name.split("/")[-1]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-{self._short_name}"

    @property
    def description(self) -> str:
        return f"Cross-encoder reranking with {self._model_name}"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (CE {self._short_name}):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in ranked_indices],
                "model": self._model_name,
            },
        )
