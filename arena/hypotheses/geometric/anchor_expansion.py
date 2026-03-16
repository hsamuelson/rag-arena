"""Anchor expansion reranking.

Hypothesis (Novel #18): A single query embedding is a point in high-dimensional
space. But complex queries often seek the intersection of multiple concepts.
A point query cannot capture this — it retrieves documents near that single
point. Instead, we should expand the query into a geometric region.

Geometric intuition: Identify the top-2 most relevant documents as "anchors"
that exemplify different aspects of what the query seeks. The affine hull of
{query, anchor1, anchor2} forms a triangle in embedding space. Documents
closest to this triangle (not just to the query point) are more likely to
cover the multi-faceted answer. A document on the edge between two anchors
bridges both aspects of the query.

Algorithm:
1. Select the top-2 scoring documents as anchors.
2. Compute the affine hull of {query, anchor1, anchor2} — a 2D plane in
   embedding space.
3. For each document, compute its distance to this affine hull by
   projecting onto the plane and measuring the residual.
4. Re-score as a blend of original similarity and inverse hull distance.
5. This naturally promotes documents that lie in the "semantic plane"
   defined by the query and its best matches.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class AnchorExpansionHypothesis(Hypothesis):
    """Rerank by distance to the affine hull of query + anchor documents."""

    def __init__(self, hull_weight: float = 0.5):
        self.hull_weight = hull_weight

    @property
    def name(self) -> str:
        return "anchor-expansion"

    @property
    def description(self) -> str:
        return (
            "Anchor expansion — expand query to affine hull of "
            "{query, anchor1, anchor2} and rank by hull distance"
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

        # Select anchors: top-2 by original score
        scores = np.array([r.score for r in results])
        anchor_indices = np.argsort(scores)[::-1][:2]
        anchor1 = E[anchor_indices[0]]
        anchor2 = E[anchor_indices[1]]

        # Build orthonormal basis for the affine hull (plane through q, a1, a2)
        # Origin = query point
        # v1 = a1 - q, v2 = a2 - q
        v1 = anchor1 - q
        v2 = anchor2 - q

        # Gram-Schmidt to get orthonormal basis
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 1e-12:
            # Degenerate: anchor1 == query, fall back
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "degenerate_anchor1"},
            )
        e1 = v1 / v1_norm

        v2_orth = v2 - np.dot(v2, e1) * e1
        v2_orth_norm = np.linalg.norm(v2_orth)

        if v2_orth_norm < 1e-12:
            # Degenerate: all three points are collinear.
            # Fall back to line distance (query to anchor1)
            hull_distances = np.zeros(n)
            for i in range(n):
                diff = E[i] - q
                proj = np.dot(diff, e1) * e1
                residual = diff - proj
                hull_distances[i] = float(np.linalg.norm(residual))
        else:
            e2 = v2_orth / v2_orth_norm
            # Project each document onto the plane
            # Basis matrix: columns are e1, e2
            basis = np.column_stack([e1, e2])  # (D, 2)

            hull_distances = np.zeros(n)
            for i in range(n):
                diff = E[i] - q  # vector from query to doc
                # Project onto the plane spanned by e1, e2
                coords = basis.T @ diff  # (2,) coordinates in plane
                proj = basis @ coords  # projection back to D-space
                residual = diff - proj
                hull_distances[i] = float(np.linalg.norm(residual))

        # Normalise hull distances to [0, 1]
        hd_max = hull_distances.max()
        if hd_max > 1e-12:
            norm_hull_dist = hull_distances / hd_max
        else:
            norm_hull_dist = np.zeros(n)

        # Hull proximity score: 1 - normalised distance (higher = closer to hull)
        hull_proximity = 1.0 - norm_hull_dist

        # Normalise original scores
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-12:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(n) * 0.5

        # Combined: blend hull proximity with original score
        w = self.hull_weight
        combined = w * hull_proximity + (1 - w) * norm_scores

        order = np.argsort(combined)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "anchor_indices": anchor_indices.tolist(),
                "anchor_doc_ids": [results[i].doc_id for i in anchor_indices],
                "hull_distances": hull_distances[order].tolist(),
                "hull_proximity": hull_proximity[order].tolist(),
                "combined_scores": combined[order].tolist(),
                "mean_hull_distance": float(hull_distances.mean()),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (anchor-expanded):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
