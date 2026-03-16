"""Adaptive Pool Depth CE.

Use the score distribution from first-stage retrieval to decide how deep
to pool per query. High-confidence queries (clear score gap between top
results and rest) need a shallow pool. Ambiguous queries (flat score
distribution) need a deeper pool.

Research basis: SEE (SIGIR 2025) - adaptive computation for reranking.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class AdaptivePoolDepthCEHypothesis(Hypothesis):
    """Adaptively choose pool depth (10-50) based on retrieval score distribution."""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_pool=10,
        max_pool=50,
        final_k=10,
    ):
        self._model_name = model_name
        self._min_pool = min_pool
        self._max_pool = max_pool
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
        return "adaptive-pool-depth-ce"

    @property
    def description(self):
        return f"Adapt pool depth ({self._min_pool}-{self._max_pool}) based on score distribution"

    def _compute_pool_size(self, results):
        """Determine pool size based on retrieval score spread."""
        if len(results) < 3:
            return self._min_pool

        scores = np.array([r.score for r in results])
        if scores.std() < 1e-8:
            return self._max_pool  # flat scores = ambiguous, need deep pool

        # Coefficient of variation: higher = more spread = more confident
        cv = scores.std() / (abs(scores.mean()) + 1e-8)

        # Score gap between top-1 and rest
        gap = scores[0] - scores[1:].mean() if len(scores) > 1 else 0

        # Low CV or small gap → need deeper pool
        # Normalize to [0, 1] range and invert for pool size
        confidence = min(cv * gap * 10, 1.0)  # higher = more confident

        pool = int(self._max_pool - confidence * (self._max_pool - self._min_pool))
        return max(self._min_pool, min(self._max_pool, pool))

    def apply(self, query, results, embeddings, query_embedding):
        # Determine adaptive pool size from initial results
        pool_size = self._compute_pool_size(results)

        # Retrieve deeper if needed and backend available
        if self._backend is not None and pool_size > len(results):
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, pool_size)
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

        lines = [f"Retrieved context (adaptive pool={pool_size}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": pool_size, "actual_pool": len(results)},
        )
