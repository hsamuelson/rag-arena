"""Deep pool cross-encoder: retrieve more, rerank more.

The core bottleneck is recall@10 = 0.890. If we retrieve top-30 instead of
top-10, recall will be higher. Then CE can pick the best 10 from a deeper pool.

This hypothesis overrides the normal apply flow:
- It stores a reference to the backend so it can do its own retrieval
- On apply(), it re-retrieves with a larger pool_size, then CE reranks

Note: This requires the hypothesis to have access to the backend, which
we inject via set_backend() before running.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class DeepPoolCEHypothesis(Hypothesis):
    """Retrieve a deeper candidate pool, CE rerank to top-K."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 30,
        final_k: int = 10,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        """Inject the backend for deep retrieval."""
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"deep-pool-{self._pool_size}-ce"

    @property
    def description(self) -> str:
        return f"Retrieve top-{self._pool_size} candidates, CE rerank to top-{self._final_k}"

    def apply(self, query, results, embeddings, query_embedding):
        # If we have a backend, do deep retrieval
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to normal results

        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Rank by CE and take top final_k
        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top_indices = ranked_indices[:self._final_k]

        reranked = [results[i] for i in top_indices]

        lines = [f"Retrieved context (deep pool {len(results)} → {len(top_indices)}, CE reranked):"]
        for i, idx in enumerate(top_indices, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top_indices],
                "pool_size": len(results),
                "final_k": len(top_indices),
            },
        )
