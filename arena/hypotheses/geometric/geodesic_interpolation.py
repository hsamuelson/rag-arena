"""Geodesic interpolation retrieval.

Hypothesis: For multi-hop questions, the answer requires connecting
documents that lie along a path in embedding space. Instead of
retrieving the K nearest neighbours (which cluster around the query),
we should retrieve documents along the geodesic path from the query
to promising "destination" embeddings.

Geometric intuition: In a flat space, the geodesic is a straight line.
In the curved embedding manifold, the geodesic follows the data
distribution. By sampling points along interpolated paths and
retrieving near those waypoints, we find the "stepping stone" documents
that connect the query to the answer.

Algorithm:
1. Retrieve initial top-K
2. Identify the farthest relevant result as the "destination"
3. Interpolate between query and destination in embedding space
4. At each waypoint, find the nearest unretrieved document
5. This builds a chain from query → answer through intermediate docs

References:
  - arXiv 2506.01599: Relative Geodesic Representations
  - Slerp interpolation on the unit hypersphere
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class GeodesicInterpolationHypothesis(Hypothesis):
    """Retrieve along geodesic paths to find multi-hop stepping stones."""

    def __init__(self, n_waypoints: int = 5, path_fraction: float = 0.5):
        self.n_waypoints = n_waypoints
        self.path_fraction = path_fraction  # How far toward destination to explore

    @property
    def name(self) -> str:
        return f"geodesic-{self.n_waypoints}wp"

    @property
    def description(self) -> str:
        return (
            f"Geodesic interpolation — retrieve along {self.n_waypoints} waypoints "
            "between query and destination to find multi-hop stepping stones"
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

        # Normalise to unit sphere for slerp
        q = self._normalise(query_embedding)
        E = np.array([self._normalise(e) for e in embeddings])

        # Identify destinations: the lowest-scoring results that are still
        # somewhat relevant (they represent the "other end" of the answer)
        # Use the median-scored result as destination
        dest_idx = n // 2
        destination = E[dest_idx]

        # Generate waypoints via slerp between query and destination
        waypoints = self._slerp_path(q, destination, self.n_waypoints)

        # For each waypoint, find the nearest document not yet selected
        selected: list[int] = []
        used = set()

        # Always include the top result
        selected.append(0)
        used.add(0)

        for wp in waypoints:
            # Find nearest unretrieved doc to this waypoint
            best_idx = -1
            best_sim = -float("inf")

            for i in range(n):
                if i in used:
                    continue
                sim = float(E[i] @ wp)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
                used.add(best_idx)

        # Fill remaining slots with unused results by original score
        for i in range(n):
            if i not in used:
                selected.append(i)
                used.add(i)

        reranked = [results[i] for i in selected]

        # Compute path metrics
        path_length = self._path_length(q, E, selected[:self.n_waypoints + 1])

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "destination_idx": dest_idx,
                "n_waypoints_used": min(self.n_waypoints, n - 1),
                "geodesic_path_length": path_length,
                "selected_order": selected,
                "query_dest_angle_deg": float(np.degrees(
                    np.arccos(np.clip(float(q @ destination), -1, 1))
                )),
            },
        )

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / max(norm, 1e-12)

    def _slerp(self, v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit vectors."""
        dot = np.clip(float(v0 @ v1), -1.0, 1.0)
        omega = np.arccos(dot)

        if abs(omega) < 1e-6:
            # Vectors nearly parallel — linear interpolation
            return self._normalise((1 - t) * v0 + t * v1)

        sin_omega = np.sin(omega)
        return (np.sin((1 - t) * omega) / sin_omega) * v0 + (np.sin(t * omega) / sin_omega) * v1

    def _slerp_path(
        self, start: np.ndarray, end: np.ndarray, n_points: int
    ) -> list[np.ndarray]:
        """Generate waypoints along the slerp path."""
        waypoints = []
        for i in range(1, n_points + 1):
            t = (i / (n_points + 1)) * self.path_fraction
            wp = self._slerp(start, end, t)
            waypoints.append(wp)
        return waypoints

    def _path_length(
        self, query: np.ndarray, embeddings: np.ndarray, path_indices: list[int]
    ) -> float:
        """Total angular distance along the selected path."""
        if len(path_indices) < 2:
            return 0.0

        total = 0.0
        prev = query
        for idx in path_indices:
            v = embeddings[idx]
            dot = np.clip(float(prev @ v), -1.0, 1.0)
            total += np.arccos(dot)
            prev = v

        return float(np.degrees(total))

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (geodesic path retrieval):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
