"""Embedding triangulation reranking.

Hypothesis (Novel #20): Instead of ranking documents solely by their distance
to the query, consider the geometric quality of the triangle formed by
(query, document, centroid_of_top3). This triangulation reveals whether a
document lies along the "main retrieval axis" or is an outlier.

Geometric intuition: The centroid of the top-3 results represents the corpus
region most relevant to the query. A good document should lie on or near the
line segment from query to this centroid — it is "mainstream relevant". The
triangle (query, doc, centroid) should have small area (doc is near the
query-centroid line) and a short query-doc edge (doc is close to the query).

A document with large triangle area is off-axis: it may be close to the query
in one dimension but deviates in another, making it an outlier. The triangle
inequality also helps: if the query-doc distance plus the doc-centroid distance
greatly exceeds the query-centroid distance, the document takes a detour
through irrelevant semantic space.

Algorithm:
1. Compute the centroid of the top-3 scoring documents.
2. For each document, compute the triangle (query, doc, centroid).
3. Measure triangle area (via cross product in embedding space) and
   triangle inequality violation = (d_qd + d_dc) / d_qc - 1.
4. Score = original_score * (1 / (1 + area)) * (1 / (1 + violation)).
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class EmbeddingTriangulationHypothesis(Hypothesis):
    """Rerank by quality of the (query, doc, centroid) triangle."""

    def __init__(
        self,
        top_k_centroid: int = 3,
        area_penalty: float = 2.0,
        violation_penalty: float = 1.5,
    ):
        self.top_k_centroid = top_k_centroid
        self.area_penalty = area_penalty
        self.violation_penalty = violation_penalty

    @property
    def name(self) -> str:
        return "embedding-triangulation"

    @property
    def description(self) -> str:
        return (
            "Embedding triangulation — rank by quality of the "
            "(query, doc, centroid) triangle to detect off-axis outliers"
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

        # Centroid of top-k scoring docs
        scores = np.array([r.score for r in results])
        top_k = min(self.top_k_centroid, n)
        top_indices = np.argsort(scores)[::-1][:top_k]
        centroid = E[top_indices].mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 1e-12:
            centroid = centroid / c_norm

        # Distance from query to centroid (the baseline axis)
        d_qc = float(np.linalg.norm(q - centroid))
        if d_qc < 1e-12:
            # Query and centroid coincide; just use raw scores
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "qc_coincide"},
            )

        # For each document, compute triangle metrics
        areas = np.zeros(n)
        violations = np.zeros(n)
        d_qd_arr = np.zeros(n)
        d_dc_arr = np.zeros(n)

        for i in range(n):
            d_qd = float(np.linalg.norm(q - E[i]))
            d_dc = float(np.linalg.norm(E[i] - centroid))
            d_qd_arr[i] = d_qd
            d_dc_arr[i] = d_dc

            # Triangle area via Heron's formula
            # Semi-perimeter
            s = (d_qd + d_dc + d_qc) / 2.0
            # Clamp to avoid numerical issues with sqrt of negative
            area_sq = s * (s - d_qd) * (s - d_dc) * (s - d_qc)
            areas[i] = float(np.sqrt(max(area_sq, 0.0)))

            # Triangle inequality violation:
            # How much longer is the detour through doc vs direct q->c?
            # Ratio >= 1 always (triangle inequality). Higher = more detour.
            if d_qc > 1e-12:
                violations[i] = (d_qd + d_dc) / d_qc - 1.0
            else:
                violations[i] = 0.0

        # Quality factors: lower area and lower violation = better
        area_factor = 1.0 / (1.0 + self.area_penalty * areas)
        violation_factor = 1.0 / (1.0 + self.violation_penalty * violations)

        # Proximity factor: closer to query = better
        d_qd_max = d_qd_arr.max()
        if d_qd_max > 1e-12:
            proximity = 1.0 - d_qd_arr / d_qd_max
        else:
            proximity = np.ones(n)

        # Combined triangulation score
        tri_score = area_factor * violation_factor * proximity

        # Normalise original scores to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-12:
            norm_scores = (scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(n) * 0.5

        # Blend: 50/50 triangulation quality and original relevance
        combined = 0.5 * tri_score + 0.5 * norm_scores

        order = np.argsort(combined)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "centroid_top_k": top_k,
                "query_centroid_distance": d_qc,
                "triangle_areas": areas[order].tolist(),
                "triangle_violations": violations[order].tolist(),
                "area_factors": area_factor[order].tolist(),
                "violation_factors": violation_factor[order].tolist(),
                "proximity_scores": proximity[order].tolist(),
                "combined_scores": combined[order].tolist(),
                "mean_area": float(areas.mean()),
                "mean_violation": float(violations.mean()),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (triangulation-ranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
