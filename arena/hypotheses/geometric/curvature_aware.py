"""Curvature-aware reranking.

Hypothesis (Novel #17): The embedding manifold is not uniformly curved.
In high-curvature regions, two points that appear nearby in Euclidean distance
may actually lie in different "semantic valleys" separated by a ridge. In
low-curvature (flat) regions, Euclidean proximity reliably indicates semantic
similarity. Ignoring curvature leads to false positives in curved regions.

Geometric intuition: Imagine the embedding space as a hilly landscape. In a
flat plain, nearby points are truly related. Near a ridge or saddle point,
two nearby points may be on opposite sides of a semantic boundary. By
estimating local curvature and down-weighting documents in high-curvature
zones, we prefer documents whose similarity to the query is geometrically
trustworthy.

Algorithm:
1. For each document, estimate local curvature using a discrete Hessian
   approximation: curvature ~ variance of pairwise distances to k nearest
   neighbours. High variance = distances change rapidly = high curvature.
2. Also estimate curvature at the query location.
3. Weight each document's score by 1 / (1 + alpha * curvature), so
   documents in flat, trustworthy regions are preferred.
4. Combine curvature-adjusted similarity with original score.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class CurvatureAwareHypothesis(Hypothesis):
    """Rerank by weighting documents inversely to local manifold curvature."""

    def __init__(
        self,
        k_neighbors: int = 4,
        curvature_weight: float = 2.0,
        blend: float = 0.5,
    ):
        self.k_neighbors = k_neighbors
        self.curvature_weight = curvature_weight
        self.blend = blend  # weight on curvature-adjusted score vs raw score

    @property
    def name(self) -> str:
        return "curvature-aware"

    @property
    def description(self) -> str:
        return (
            "Curvature-aware reranking — down-weight documents in "
            "high-curvature embedding regions where proximity is unreliable"
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

        # Normalise
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

        k = min(self.k_neighbors, n - 1)

        # Build combined point set: docs + query
        all_points = np.vstack([E, q.reshape(1, -1)])  # (n+1, D)

        # Pairwise distances in embedding space
        # Use squared Euclidean for efficiency
        dists = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            diff = all_points - all_points[i]
            dists[i] = np.sqrt(np.sum(diff ** 2, axis=1))

        # Estimate curvature at each point via Hessian approximation:
        # curvature_i = std of distances to k-nearest neighbours.
        # Rationale: in flat regions, k-NN distances are uniform (low std);
        # in curved regions, they vary wildly (high std).
        curvatures = np.zeros(n + 1)
        for i in range(n + 1):
            # Get sorted distances, excluding self
            d_sorted = np.sort(dists[i])
            # d_sorted[0] is 0 (self), take next k
            knn_dists = d_sorted[1 : k + 1]
            if len(knn_dists) > 1:
                curvatures[i] = float(np.std(knn_dists))
            else:
                curvatures[i] = 0.0

        # Second-order curvature estimate: variance of second differences
        # of distances along nearest-neighbour chains
        for i in range(n + 1):
            d_sorted = np.sort(dists[i])
            knn_dists = d_sorted[1 : k + 1]
            if len(knn_dists) >= 3:
                second_diffs = np.diff(knn_dists, n=2)
                curvatures[i] = float(
                    curvatures[i] + np.abs(second_diffs).mean()
                )

        doc_curvatures = curvatures[:n]
        query_curvature = curvatures[n]

        # Curvature-adjusted weights: prefer low-curvature regions
        alpha = self.curvature_weight
        curvature_weights = 1.0 / (1.0 + alpha * doc_curvatures)

        # Raw query-document similarities
        raw_sims = E @ q

        # Curvature-adjusted scores
        adjusted_sims = raw_sims * curvature_weights

        # Normalise both to [0, 1] for blending
        def _norm01(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-12:
                return np.ones_like(arr) * 0.5
            return (arr - lo) / (hi - lo)

        norm_raw = _norm01(raw_sims)
        norm_adj = _norm01(adjusted_sims)

        combined = self.blend * norm_adj + (1 - self.blend) * norm_raw

        order = np.argsort(combined)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "doc_curvatures": doc_curvatures[order].tolist(),
                "query_curvature": float(query_curvature),
                "curvature_weights": curvature_weights[order].tolist(),
                "raw_sims": raw_sims[order].tolist(),
                "adjusted_sims": adjusted_sims[order].tolist(),
                "combined_scores": combined[order].tolist(),
                "mean_curvature": float(doc_curvatures.mean()),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (curvature-aware ranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
