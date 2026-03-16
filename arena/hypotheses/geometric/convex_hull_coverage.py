"""Convex hull coverage reranking.

Hypothesis: The best retrieval set is the one whose convex hull in
embedding space contains (or comes closest to containing) the query
point. This ensures the retrieved documents "surround" the query
from all relevant directions, rather than clustering on one side.

Geometric intuition: If the query lies inside the convex hull of
retrieved embeddings, the answer can be interpolated from context.
If outside, there's a blind spot the LLM can't fill.

Algorithm:
1. Retrieve 2K candidates
2. Use PCA to project to low-D (3-5 dims)
3. Greedily select K documents that maximise coverage of the
   query's neighbourhood, measured by:
   - Minimum enclosing ball containing query
   - Angular coverage around query direction
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class ConvexHullCoverageHypothesis(Hypothesis):
    """Select documents that maximally surround the query in embedding space."""

    def __init__(self, n_components: int = 5, angular_bins: int = 8):
        self.n_components = n_components
        self.angular_bins = angular_bins  # Discretise angles for coverage

    @property
    def name(self) -> str:
        return f"hull-coverage-{self.n_components}d"

    @property
    def description(self) -> str:
        return (
            "Convex hull coverage — select documents that surround the query "
            "in embedding space, maximising angular coverage"
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

        # Project to lower dimension via PCA
        all_embs = np.vstack([query_embedding[np.newaxis, :], embeddings])
        proj_all, explained = self._pca_project(all_embs, self.n_components)
        query_proj = proj_all[0]
        doc_projs = proj_all[1:]

        # Compute displacement vectors from query
        displacements = doc_projs - query_proj  # (K, M)

        # Measure angular coverage using direction binning
        selected = self._greedy_angular_coverage(
            displacements,
            [r.score for r in results],
            len(results),
        )

        reranked = [results[i] for i in selected]

        # Compute coverage metric
        coverage = self._angular_coverage_score(displacements[selected])

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "angular_coverage": coverage,
                "explained_variance": explained,
                "selected_indices": selected,
                "query_inside_hull": self._point_in_hull_approx(query_proj, doc_projs[selected]),
            },
        )

    def _pca_project(self, X: np.ndarray, n_comp: int) -> tuple[np.ndarray, list[float]]:
        """PCA projection to n_comp dimensions."""
        n, d = X.shape
        n_comp = min(n_comp, n - 1, d)
        mean = X.mean(axis=0)
        centred = X - mean
        _, S, Vt = np.linalg.svd(centred, full_matrices=False)
        total = (S ** 2).sum()
        explained = [(S[i] ** 2 / total) for i in range(n_comp)] if total > 0 else []
        return centred @ Vt[:n_comp].T, explained

    def _greedy_angular_coverage(
        self,
        displacements: np.ndarray,
        relevances: list[float],
        k: int,
    ) -> list[int]:
        """Greedily select documents to maximise angular coverage around query.

        Uses direction vectors from query to each doc. Tracks which angular
        "sectors" are covered, preferring docs in uncovered directions.
        """
        n, m = displacements.shape
        if m == 0:
            return list(range(min(k, n)))

        # Normalise displacements to unit directions
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        directions = displacements / norms

        # For each pair of dimensions, discretise angles
        # This captures multi-dimensional angular spread
        covered_angles: set[tuple[int, ...]] = set()
        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(min(k, n)):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                # Count new angle bins this document would cover
                angle_sig = self._angle_signature(directions[idx])
                new_coverage = 1 if angle_sig not in covered_angles else 0

                # Combined score: relevance + coverage bonus
                score = 0.4 * relevances[idx] + 0.6 * new_coverage

                # Tiebreak by distance from already-selected (favour far points)
                if selected:
                    min_sim = min(
                        float(directions[idx] @ directions[s])
                        for s in selected
                    )
                    score += 0.1 * (1.0 - min_sim)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            covered_angles.add(self._angle_signature(directions[best_idx]))

        return selected

    def _angle_signature(self, direction: np.ndarray) -> tuple[int, ...]:
        """Discretise a direction vector into angular bin indices.

        For each pair of adjacent dimensions, compute the angle and bin it.
        This gives a multi-dimensional "sector" signature.
        """
        bins = self.angular_bins
        sig = []
        for i in range(len(direction) - 1):
            angle = np.arctan2(direction[i + 1], direction[i])
            # Map [-pi, pi] to [0, bins)
            bin_idx = int((angle + np.pi) / (2 * np.pi) * bins) % bins
            sig.append(bin_idx)
        return tuple(sig)

    def _angular_coverage_score(self, displacements: np.ndarray) -> float:
        """Fraction of angular bins covered by the selected set."""
        if displacements.shape[0] == 0 or displacements.shape[1] < 2:
            return 0.0

        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        directions = displacements / norms

        bins = self.angular_bins
        n_pairs = displacements.shape[1] - 1
        total_bins = bins * n_pairs
        covered: set[tuple[int, int]] = set()

        for d in directions:
            for i in range(n_pairs):
                angle = np.arctan2(d[i + 1], d[i])
                bin_idx = int((angle + np.pi) / (2 * np.pi) * bins) % bins
                covered.add((i, bin_idx))

        return len(covered) / total_bins if total_bins > 0 else 0.0

    def _point_in_hull_approx(self, point: np.ndarray, hull_points: np.ndarray) -> bool:
        """Approximate check: is the point inside the convex hull?

        Uses the sign of the minimum barycentric-like coordinate.
        For low dimensions, checks if point can be expressed as a
        convex combination of hull points (all weights >= 0, sum = 1).
        """
        if hull_points.shape[0] <= hull_points.shape[1]:
            return False  # Not enough points to form a hull

        # Least-squares: find weights w such that hull_points.T @ w ≈ point, sum(w) = 1
        n = hull_points.shape[0]
        # Augmented system: [hull_points.T; ones] @ w = [point; 1]
        A = np.vstack([hull_points.T, np.ones(n)])
        b = np.append(point, 1.0)

        try:
            w, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return bool(np.all(w > -0.05))  # Small tolerance
        except np.linalg.LinAlgError:
            return False

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (hull-coverage reranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
