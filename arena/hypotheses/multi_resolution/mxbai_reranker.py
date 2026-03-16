"""Mxbai-rerank-base-v2 reranker on deep pool 50.

mixedbread-ai/mxbai-rerank-base-v2 is a ~500M-param cross-encoder
from the mixedbread.ai team. Tests an alternative large reranker
on the same candidate pool.
"""

import numpy as np
import torch

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class MxbaiRerankerHypothesis(Hypothesis):
    """Retrieve top-50 candidates, mxbai-rerank-base-v2 rerank to top-10."""

    def __init__(self, model_name="mixedbread-ai/mxbai-rerank-base-v2", final_k=10):
        self._model_name = model_name
        self._pool_size = 50
        self._final_k = final_k
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                trust_remote_code=True,
                automodel_args={"torch_dtype": "auto"},
            )
        return self._model

    @property
    def name(self):
        return "mxbai-rerank-base-v2"

    @property
    def description(self):
        return f"Retrieve top-{self._pool_size}, mxbai-rerank-base-v2 rerank to top-{self._final_k}"

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
        raw_scores = model.predict(pairs)
        # mxbai-rerank-base-v2 outputs raw logits; apply sigmoid for proper relevance scores
        ce_scores = torch.sigmoid(torch.tensor(raw_scores)).numpy().tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (deep pool {len(results)} → {len(top)}, mxbai-rerank reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"ce_scores": [ce_scores[i] for i in top], "pool_size": len(results)},
        )
