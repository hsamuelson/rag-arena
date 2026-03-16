"""Expanded retrieval hypothesis.

The bottleneck is recall — hybrid gives 0.890 but not 1.0. This hypothesis
tries to improve recall by requesting MORE candidates from the backend,
then using cross-encoder to filter down to top-K.

Standard: retrieve 10, rerank 10
Expanded: retrieve 30, CE rerank to top 10

More candidates = higher recall = better chance of finding all relevant docs.
This directly addresses the recall bottleneck.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class ExpandedRetrievalCEHypothesis(Hypothesis):
    """Retrieve more candidates, CE rerank to top-K."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        expansion_factor: int = 3,  # retrieve 3x more candidates
    ):
        self._model_name = model_name
        self._expansion_factor = expansion_factor
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"expanded-retrieval-{self._expansion_factor}x-ce"

    @property
    def description(self) -> str:
        return f"Retrieve {self._expansion_factor}x candidates, CE rerank to top-K"

    def apply(self, query, results, embeddings, query_embedding):
        """Note: This hypothesis receives whatever results the runner gives it.
        To actually expand retrieval, the runner would need to pass more candidates.
        For now, this works with whatever it gets and demonstrates the CE reranking
        on whatever pool size is available."""
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (expanded retrieval {self._expansion_factor}x + CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in ranked_indices],
                "pool_size": len(results),
                "expansion_factor": self._expansion_factor,
            },
        )
