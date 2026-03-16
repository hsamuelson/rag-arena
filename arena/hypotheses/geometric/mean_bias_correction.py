"""Mean bias correction (anisotropy removal).

Hypothesis: Text embeddings exhibit a consistent mean bias — all vectors
point partially in a shared direction, causing the embedding space to be
anisotropic. This wastes dimensions on encoding "I am a text embedding"
rather than semantic content. Removing this bias before retrieval makes
cosine similarity more discriminative.

This is a training-free, zero-cost fix that has been shown to consistently
improve retrieval and classification (arXiv:2511.11041, Nov 2025).

Algorithm:
1. Compute the mean embedding of the retrieved set (or ideally the corpus)
2. Subtract it from all embeddings (centering)
3. Re-normalise to unit sphere
4. Re-rank by corrected cosine similarity

Geometric intuition: The original embeddings cluster in a narrow cone.
After centering, they spread across the full hypersphere, making angular
differences more meaningful.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class MeanBiasCorrectionHypothesis(Hypothesis):
    """Remove mean bias from embeddings before re-ranking."""

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "mean-bias-correction"

    @property
    def description(self) -> str:
        return (
            "Mean bias correction — remove shared directional bias from "
            "embeddings to improve cosine similarity discriminativeness"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 3:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        # Compute mean of document embeddings
        mean_emb = embeddings.mean(axis=0)
        mean_norm = np.linalg.norm(mean_emb)

        # Centre embeddings
        centred_docs = embeddings - mean_emb
        centred_query = query_embedding - mean_emb

        # Re-normalise
        doc_norms = np.linalg.norm(centred_docs, axis=1, keepdims=True)
        doc_norms = np.maximum(doc_norms, 1e-12)
        centred_docs = centred_docs / doc_norms

        q_norm = np.linalg.norm(centred_query)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "degenerate query after centering"},
            )
        centred_query = centred_query / q_norm

        # Recompute cosine similarities
        corrected_sims = centred_docs @ centred_query

        # Measure how much the ranking changed
        original_order = list(range(len(results)))
        new_order = np.argsort(corrected_sims)[::-1].tolist()

        reranked = [results[i] for i in new_order]

        # Kendall tau distance (number of swaps)
        rank_changes = sum(
            1 for i in range(len(new_order))
            if new_order[i] != original_order[i]
        )

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "mean_bias_norm": float(mean_norm),
                "mean_bias_direction_cos_to_query": float(
                    mean_emb @ query_embedding / (mean_norm * np.linalg.norm(query_embedding) + 1e-12)
                ),
                "rank_changes": rank_changes,
                "corrected_similarities": corrected_sims[new_order].tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (mean-bias corrected):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
