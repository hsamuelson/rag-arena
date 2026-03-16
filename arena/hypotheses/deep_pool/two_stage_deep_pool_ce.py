"""Two-stage deep pool CE: coarse → fine reranking.

Stage 1: Retrieve 50 candidates, CE rerank to top-20
Stage 2: CE rerank top-20 to top-10 (with full text, not truncated)

The idea is that stage 1 filters aggressively, then stage 2 can afford
to read more text per document for finer discrimination.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS_STAGE1 = 1500  # Shorter for speed in stage 1
_MAX_CHARS_STAGE2 = 3000  # Longer for accuracy in stage 2


class TwoStageDeepPoolCEHypothesis(Hypothesis):
    """Two-stage CE: retrieve 50 → CE to 20 → CE to 10 with more text."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
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
        return "two-stage-deep-pool-ce"

    @property
    def description(self):
        return "Retrieve 50 → CE coarse to 20 → CE fine to 10"

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, 50)
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()

        # Stage 1: Coarse rerank with shorter text
        pairs1 = [(query, r.text[:_MAX_CHARS_STAGE1]) for r in results]
        scores1 = model.predict(pairs1).tolist()
        ranked1 = sorted(range(len(scores1)), key=lambda i: scores1[i], reverse=True)
        top20 = ranked1[:20]
        stage1_results = [results[i] for i in top20]

        # Stage 2: Fine rerank with longer text
        pairs2 = [(query, r.text[:_MAX_CHARS_STAGE2]) for r in stage1_results]
        scores2 = model.predict(pairs2).tolist()
        ranked2 = sorted(range(len(scores2)), key=lambda i: scores2[i], reverse=True)
        top10 = ranked2[:10]
        reranked = [stage1_results[i] for i in top10]

        lines = [f"Retrieved context (2-stage: {len(results)}→20→{len(top10)}, CE reranked):"]
        for i, idx in enumerate(top10, 1):
            lines.append(f"\n[{i}] (ce_score: {scores2[idx]:.4f})")
            lines.append(stage1_results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "stage1_pool": len(results),
                "stage2_pool": len(stage1_results),
                "final": len(reranked),
            },
        )
