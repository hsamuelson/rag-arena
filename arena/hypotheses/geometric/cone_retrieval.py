"""Cone retrieval with adaptive aperture.

Hypothesis: Standard cosine similarity retrieves documents within a
"hyperspherical cap" around the query direction. But the optimal
retrieval cone aperture varies by query — narrow for specific factual
queries, wide for exploratory/multi-faceted queries. By estimating
the query's "specificity" from the local density of the embedding
space, we can adapt the retrieval aperture.

Geometric intuition: Cosine similarity is the dot product of unit
vectors — it measures the angle between them. A "cone" in embedding
space is the set of points within angle θ of the query. We adaptively
set θ based on:
  - Local density: dense regions need narrower cones
  - Score distribution: uniform scores suggest wide cone needed
  - Score gap: sharp dropoff suggests narrow cone optimal

This effectively turns top-K into a top-θ retrieval strategy.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class ConeRetrievalHypothesis(Hypothesis):
    """Adaptive cone aperture retrieval based on local embedding density."""

    def __init__(self, min_results: int = 3, score_gap_threshold: float = 0.15):
        self.min_results = min_results
        self.score_gap_threshold = score_gap_threshold

    @property
    def name(self) -> str:
        return "cone-retrieval"

    @property
    def description(self) -> str:
        return (
            "Adaptive cone retrieval — dynamically set the retrieval aperture "
            "based on score distribution and local embedding density"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if not results or len(results) < 3:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        scores = np.array([r.score for r in results])

        # Estimate optimal cone aperture from score distribution
        cutoff_idx, cone_metrics = self._estimate_cone_cutoff(scores)

        # Apply cutoff: keep documents inside the cone
        cone_results = results[:cutoff_idx]

        # If embeddings available, also measure angular spread
        angular_stats = {}
        if embeddings is not None and query_embedding is not None:
            angular_stats = self._angular_analysis(
                query_embedding, embeddings[:cutoff_idx]
            )

        return HypothesisResult(
            results=cone_results,
            context_prompt=self._format(cone_results),
            metadata={
                "cone_cutoff": cutoff_idx,
                "total_candidates": len(results),
                **cone_metrics,
                **angular_stats,
            },
        )

    def _estimate_cone_cutoff(self, scores: np.ndarray) -> tuple[int, dict]:
        """Estimate where to cut off based on score distribution geometry.

        Uses three signals:
        1. Score gaps: large gap = natural boundary
        2. Score curvature: inflection point in score curve
        3. Entropy: high entropy = uniform = keep more
        """
        n = len(scores)
        if n <= self.min_results:
            return n, {"method": "min_results"}

        # 1. Find largest score gap
        gaps = np.diff(scores)  # These are negative (scores descending)
        abs_gaps = np.abs(gaps)

        # Normalise gaps by score range
        score_range = scores[0] - scores[-1]
        if score_range > 1e-6:
            norm_gaps = abs_gaps / score_range
        else:
            return n, {"method": "uniform_scores"}

        # 2. Find significant gap (above threshold)
        gap_cutoff = n
        for i, g in enumerate(norm_gaps):
            if g > self.score_gap_threshold and i + 1 >= self.min_results:
                gap_cutoff = i + 1
                break

        # 3. Score distribution entropy (uniform = high entropy = explore more)
        # Normalise scores to probabilities
        shifted = scores - scores.min() + 1e-8
        probs = shifted / shifted.sum()
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = np.log(n)
        normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # High entropy (> 0.8) suggests uniform scores — keep more
        if normalised_entropy > 0.8:
            entropy_cutoff = n
        else:
            # Scale cutoff by entropy: low entropy = fewer docs
            entropy_cutoff = max(
                self.min_results,
                int(n * normalised_entropy)
            )

        # 4. Score curve curvature (second derivative)
        if n >= 4:
            second_deriv = np.diff(gaps)
            # Find inflection point (max absolute second derivative)
            inflection_idx = int(np.argmax(np.abs(second_deriv))) + 2
            inflection_cutoff = max(self.min_results, inflection_idx)
        else:
            inflection_cutoff = n

        # Combine: take the minimum of gap-based and curvature-based
        cutoff = min(gap_cutoff, inflection_cutoff, entropy_cutoff)
        cutoff = max(cutoff, self.min_results)

        return cutoff, {
            "method": "adaptive",
            "gap_cutoff": gap_cutoff,
            "inflection_cutoff": inflection_cutoff,
            "entropy_cutoff": entropy_cutoff,
            "normalised_entropy": float(normalised_entropy),
            "max_normalised_gap": float(norm_gaps.max()) if len(norm_gaps) > 0 else 0,
        }

    def _angular_analysis(
        self, query_emb: np.ndarray, doc_embs: np.ndarray
    ) -> dict:
        """Measure the angular spread of retrieved docs around query."""
        q_norm = np.linalg.norm(query_emb)
        if q_norm < 1e-12:
            return {}

        q = query_emb / q_norm
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        D = doc_embs / norms

        # Cosine angles to query
        cos_angles = D @ q
        angles_rad = np.arccos(np.clip(cos_angles, -1, 1))
        angles_deg = np.degrees(angles_rad)

        return {
            "cone_half_angle_deg": float(angles_deg.max()),
            "mean_angle_deg": float(angles_deg.mean()),
            "angle_std_deg": float(angles_deg.std()),
        }

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = [f"Retrieved context ({len(results)} docs in adaptive cone):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
