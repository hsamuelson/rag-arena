"""Deep pool 100: maximum recall test.

How far can we push the recall → CE rerank pipeline?
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class DeepPool100CEHypothesis(Hypothesis):
    """Retrieve top-100 candidates, CE rerank to top-10."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", final_k=10):
        self._model_name = model_name
        self._pool_size = 100
        self._final_k = final_k
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return "deep-pool-100-ce"

    @property
    def description(self):
        return f"Retrieve top-{self._pool_size}, CE rerank to top-{self._final_k}"

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, self._pool_size)
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (deep pool {len(results)} → {len(top)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"ce_scores": [ce_scores[i] for i in top], "pool_size": len(results)},
        )
