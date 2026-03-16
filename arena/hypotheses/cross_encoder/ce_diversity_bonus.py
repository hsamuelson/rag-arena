"""Cross-encoder with diversity bonus (MMR-style).

Standard CE ranks by pure relevance, but for multi-hop QA, we need
diverse passages covering different aspects. This applies MMR (Maximal
Marginal Relevance) on top of CE scores using embeddings for diversity.

Score = lambda * CE_score - (1-lambda) * max_sim_to_already_selected

This greedily selects documents that are both relevant (CE) and
diverse (not redundant with already-selected docs).
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CEDiversityBonusHypothesis(Hypothesis):
    """Cross-encoder reranking with MMR diversity bonus."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        lambda_param: float = 0.85,
    ):
        self._model_name = model_name
        self._lambda = lambda_param
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-diversity-{self._lambda}"

    @property
    def description(self) -> str:
        return f"Cross-encoder + MMR diversity (lambda={self._lambda})"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Normalize CE scores to [0, 1]
        ce_arr = np.array(ce_scores, dtype=np.float64)
        ce_min, ce_max = ce_arr.min(), ce_arr.max()
        if ce_max - ce_min > 1e-12:
            ce_norm = (ce_arr - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_arr)

        # If no embeddings, fall back to pure CE
        if embeddings is None or len(embeddings) != len(results):
            ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        else:
            # Compute pairwise cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normed = embeddings / norms
            sim_matrix = normed @ normed.T

            # Greedy MMR selection
            n = len(results)
            selected = []
            remaining = set(range(n))

            for _ in range(n):
                best_idx = None
                best_score = -float('inf')

                for idx in remaining:
                    relevance = ce_norm[idx]

                    if selected:
                        max_sim = max(float(sim_matrix[idx, s]) for s in selected)
                    else:
                        max_sim = 0.0

                    mmr_score = self._lambda * relevance - (1 - self._lambda) * max_sim

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                selected.append(best_idx)
                remaining.remove(best_idx)

            ranked = selected

        reranked = [results[i] for i in ranked]

        lines = ["Retrieved context (CE + diversity):"]
        for i, idx in enumerate(ranked, 1):
            lines.append(f"\n[{i}] (ce: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in ranked],
                "selection_order": ranked,
                "lambda": self._lambda,
            },
        )
