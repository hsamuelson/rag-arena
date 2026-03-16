"""Adaptive context window via score distribution analysis.

Hypothesis: Fixed top-K retrieval includes a constant number of documents
regardless of how confident the retrieval is. When the query has a clear
answer, a few documents score much higher than the rest — including the
low-scoring tail adds noise. When the query is ambiguous, scores are
more uniformly distributed and more context is needed.

By finding the natural "elbow" in the score distribution, we can
adaptively size the context window to include only high-confidence
results, reducing noise without losing coverage.

Algorithm:
1. Sort results by score (descending)
2. Compute the second derivative (curvature) of the score sequence
3. Find the point of maximum curvature (the "elbow")
4. Include only results above the elbow point
5. Ensure a minimum of min_results and maximum of max_results

Geometric intuition: The sorted score curve is a monotonically
decreasing function. The elbow point is where the curve transitions
from "confidently relevant" to "probably noise". The second derivative
measures how sharply the curve bends — the maximum curvature point
is the natural boundary between signal and noise.

References:
  - Satopaa et al. (2011): Finding a "Kneedle" in a Haystack
  - Lassance et al. (2023): Adaptive retrieval length for RAG
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class AdaptiveContextWindowHypothesis(Hypothesis):
    """Dynamically size the context window using score distribution elbow detection."""

    def __init__(self, min_results: int = 2, max_results: int | None = None):
        """
        Args:
            min_results: Minimum number of results to always include.
            max_results: Maximum cap (None = no cap beyond input size).
        """
        self.min_results = min_results
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "adaptive-context-window"

    @property
    def description(self) -> str:
        return (
            "Adaptive context window — find the natural elbow in the "
            "score distribution to include only high-confidence results"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if len(results) < 3:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        n = len(results)

        # If we have embeddings, recompute similarities for consistency
        if embeddings is not None and query_embedding is not None:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            E = embeddings / norms
            q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
            sims = E @ q
            sort_idx = np.argsort(sims)[::-1]
            sorted_scores = sims[sort_idx]
            sorted_results = [results[i] for i in sort_idx]
        else:
            # Use existing scores
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            sorted_scores = np.array([r.score for r in sorted_results])

        # Find the elbow using the Kneedle algorithm (simplified)
        elbow_idx = self._find_elbow(sorted_scores)

        # Apply constraints
        elbow_idx = max(elbow_idx, self.min_results)
        if self.max_results is not None:
            elbow_idx = min(elbow_idx, self.max_results)
        elbow_idx = min(elbow_idx, n)

        # Truncate to elbow
        truncated = sorted_results[:elbow_idx]

        # Compute score statistics
        included_scores = sorted_scores[:elbow_idx]
        excluded_scores = sorted_scores[elbow_idx:]

        score_gap = 0.0
        if len(excluded_scores) > 0:
            score_gap = float(included_scores[-1] - excluded_scores[0])

        return HypothesisResult(
            results=truncated,
            context_prompt=self._format_adaptive(truncated, elbow_idx, n),
            metadata={
                "original_count": n,
                "adaptive_count": elbow_idx,
                "documents_removed": n - elbow_idx,
                "removal_fraction": float(n - elbow_idx) / n,
                "elbow_score": float(sorted_scores[elbow_idx - 1]),
                "score_gap_at_elbow": score_gap,
                "score_range": float(sorted_scores[0] - sorted_scores[-1]),
                "score_std": float(np.std(sorted_scores)),
                "included_score_mean": float(np.mean(included_scores)),
            },
        )

    def _find_elbow(self, scores: np.ndarray) -> int:
        """Find the elbow point using the Kneedle algorithm.

        Normalise the score curve to [0,1] x [0,1], compute the distance
        from each point to the line connecting the first and last points,
        and return the point with maximum distance.
        """
        n = len(scores)
        if n <= 2:
            return n

        # Normalise x and y to [0, 1]
        x = np.linspace(0, 1, n)
        y_min, y_max = scores[-1], scores[0]
        if y_max - y_min < 1e-12:
            # All scores are the same; include everything
            return n
        y = (scores - y_min) / (y_max - y_min)

        # Distance from each point to the line from (0, y[0]) to (1, y[-1])
        # Line: from (x0, y0) to (x1, y1)
        x0, y0 = 0.0, y[0]
        x1, y1 = 1.0, y[-1]
        line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if line_len < 1e-12:
            return n

        # Perpendicular distance: |cross product| / line_len
        distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / line_len

        # The elbow is the point with maximum distance
        # We want to include up to and including the elbow point
        elbow = int(np.argmax(distances)) + 1  # +1 for inclusive

        # Also check second derivative as a fallback
        if n >= 4:
            second_deriv = np.diff(scores, n=2)
            # The sharpest drop is where second derivative is most positive
            # (score is decreasing, so acceleration = flattening after steep drop)
            sd_elbow = int(np.argmax(np.abs(second_deriv))) + 1
            # Take the more conservative of the two
            elbow = min(elbow, sd_elbow + 1)

        return max(1, elbow)

    def _format_adaptive(
        self, results: list[RetrievalResult], kept: int, total: int
    ) -> str:
        lines = [
            f"Retrieved context ({kept} of {total} results above confidence threshold):"
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (adaptive window):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
