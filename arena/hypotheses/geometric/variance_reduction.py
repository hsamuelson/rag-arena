"""Variance reduction reranking via control variates.

Hypothesis (Novel #14): The retrieved set has high variance in relevance —
some documents are genuinely relevant while others are borderline distractors
that happened to score well due to embedding noise. This variance hurts LLM
generation because the model cannot distinguish signal from noise in context.

Geometric intuition: Borrowed from Monte Carlo control variates. If we know
a correlated quantity whose expectation we can compute, we can subtract it
to reduce variance. Here the "control variate" for each document is the mean
score in its embedding neighbourhood. A document scoring 0.85 whose neighbours
average 0.84 is unremarkable (adjusted score ~0.01). A document scoring 0.80
whose neighbours average 0.65 is genuinely distinctive (adjusted score ~0.15).

Algorithm:
1. Cluster documents by pairwise embedding similarity (greedy grouping).
2. For each group, compute the within-group relevance variance.
3. For each document, subtract the group mean score ("control variate")
   to get the adjusted score — how far above its local average.
4. Prefer documents from low-variance groups (more reliable signal)
   by applying a penalty proportional to group variance.
5. Final score = adjusted_score - lambda * group_variance.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class VarianceReductionHypothesis(Hypothesis):
    """Rerank by subtracting a control-variate baseline to reduce score variance."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        variance_penalty: float = 1.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.variance_penalty = variance_penalty

    @property
    def name(self) -> str:
        return "variance-reduction"

    @property
    def description(self) -> str:
        return (
            "Control-variate variance reduction — subtract neighbourhood mean "
            "score and penalise high-variance groups to surface genuinely "
            "above-average documents"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 4:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        n = len(results)

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        q = query_embedding / q_norm

        # Query-document similarities
        raw_sims = E @ q  # (n,)

        # Pairwise document similarities for grouping
        doc_sims = E @ E.T  # (n, n)

        # Greedy grouping: assign each doc to the first group whose centroid
        # it is sufficiently similar to, or start a new group
        groups: list[list[int]] = []
        group_assignment = np.full(n, -1, dtype=int)

        for i in range(n):
            assigned = False
            for g_idx, group in enumerate(groups):
                # Check similarity to group centroid
                group_centroid = E[group].mean(axis=0)
                group_centroid_norm = np.linalg.norm(group_centroid)
                if group_centroid_norm > 1e-12:
                    sim_to_group = float(E[i] @ group_centroid / group_centroid_norm)
                    if sim_to_group >= self.similarity_threshold:
                        group.append(i)
                        group_assignment[i] = g_idx
                        assigned = True
                        break
            if not assigned:
                group_assignment[i] = len(groups)
                groups.append([i])

        # Compute per-group statistics
        num_groups = len(groups)
        group_mean_sims = np.zeros(num_groups)
        group_variances = np.zeros(num_groups)

        for g_idx, group in enumerate(groups):
            group_scores = raw_sims[group]
            group_mean_sims[g_idx] = group_scores.mean()
            group_variances[g_idx] = group_scores.var() if len(group) > 1 else 0.0

        # Adjusted score: subtract control variate (group mean)
        adjusted_scores = np.zeros(n)
        for i in range(n):
            g = group_assignment[i]
            control_variate = group_mean_sims[g]
            variance_pen = group_variances[g]
            adjusted_scores[i] = (
                (raw_sims[i] - control_variate)
                - self.variance_penalty * variance_pen
            )

        # Rank by adjusted score
        order = np.argsort(adjusted_scores)[::-1]
        reranked = [results[i] for i in order]

        # Metadata
        original_variance = float(raw_sims.var())
        adjusted_variance = float(adjusted_scores.var())

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "num_groups": num_groups,
                "group_sizes": [len(g) for g in groups],
                "group_variances": group_variances.tolist(),
                "original_score_variance": original_variance,
                "adjusted_score_variance": adjusted_variance,
                "variance_reduction_ratio": (
                    1.0 - adjusted_variance / max(original_variance, 1e-12)
                ),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (variance-reduced):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
