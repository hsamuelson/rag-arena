"""Cross-encoder ensemble hypothesis.

Run multiple cross-encoder models and fuse their scores. Different
models capture different relevance patterns — ensembling reduces
individual model bias.

Models:
1. cross-encoder/ms-marco-MiniLM-L-6-v2 (default, 22M params)
2. cross-encoder/ms-marco-TinyBERT-L-2-v2 (4.4M params, different architecture)

Fusion via RRF or normalized score averaging.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CEEnsembleHypothesis(Hypothesis):
    """Ensemble of multiple cross-encoder models with score fusion."""

    def __init__(
        self,
        model_names=None,
        fusion: str = "rrf",  # "rrf" or "avg"
        rrf_k: int = 60,
    ):
        self._model_names = model_names or [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        ]
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._models = {}

    def _get_model(self, name):
        if name not in self._models:
            from sentence_transformers import CrossEncoder
            self._models[name] = CrossEncoder(name)
        return self._models[name]

    @property
    def name(self) -> str:
        return f"ce-ensemble-{self._fusion}-{len(self._model_names)}m"

    @property
    def description(self) -> str:
        return f"Ensemble of {len(self._model_names)} cross-encoder models with {self._fusion} fusion"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        n = len(results)

        # Score with each model
        all_scores = []
        for model_name in self._model_names:
            model = self._get_model(model_name)
            scores = model.predict(pairs).tolist()
            all_scores.append(scores)

        if self._fusion == "rrf":
            # RRF fusion across model rankings
            fused = np.zeros(n, dtype=np.float64)
            for scores in all_scores:
                ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
                for rank, idx in enumerate(ranking):
                    fused[idx] += 1.0 / (self._rrf_k + rank + 1)
        else:
            # Normalized score averaging
            fused = np.zeros(n, dtype=np.float64)
            for scores in all_scores:
                arr = np.array(scores, dtype=np.float64)
                mn, mx = arr.min(), arr.max()
                if mx - mn > 1e-12:
                    arr = (arr - mn) / (mx - mn)
                else:
                    arr = np.ones_like(arr)
                fused += arr
            fused /= len(all_scores)

        ranked_indices = sorted(range(n), key=lambda i: fused[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (CE ensemble, {self._fusion}, {len(self._model_names)} models):"]
        for i, idx in enumerate(ranked_indices, 1):
            model_scores = ", ".join(f"{s[idx]:.3f}" for s in all_scores)
            lines.append(f"\n[{i}] (fused: {fused[idx]:.4f}, models: [{model_scores}])")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "fused_scores": [float(fused[i]) for i in ranked_indices],
                "per_model_scores": [[s[i] for i in ranked_indices] for s in all_scores],
                "fusion": self._fusion,
                "models": self._model_names,
            },
        )
