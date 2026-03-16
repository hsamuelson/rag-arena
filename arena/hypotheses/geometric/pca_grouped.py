"""PCA-grouped presentation (from SEMDA/Caldera research).

Hypothesis: LLM answer quality improves when retrieved documents are
presented grouped by their principal component structure, rather than
as a flat ranked list. The grouping reveals the semantic axes of
variation in the retrieved set.

This is the "presentation" half of SEMDA — it doesn't change *which*
documents are retrieved, only *how* they're shown to the LLM.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class PCAGroupedHypothesis(Hypothesis):
    """PCA-grouped presentation — cluster results by principal axes."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    @property
    def name(self) -> str:
        return f"pca-grouped-{self.n_components}c"

    @property
    def description(self) -> str:
        return f"PCA-grouped presentation with {self.n_components} semantic axes"

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or len(results) < 3:
            context = self._format_flat(results)
            return HypothesisResult(
                results=results, context_prompt=context,
                metadata={"fallback": True},
            )

        # PCA decomposition
        n, d = embeddings.shape
        n_comp = min(self.n_components, n - 1, d)
        if n_comp < 1:
            context = self._format_flat(results)
            return HypothesisResult(results=results, context_prompt=context)

        mean = embeddings.mean(axis=0)
        centred = embeddings - mean
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)

        total_var = (S ** 2).sum() / (n - 1) if n > 1 else 1.0
        projections = centred @ Vt[:n_comp].T

        # Assign each result to its dominant axis (highest absolute projection)
        assignments = np.argmax(np.abs(projections), axis=1)

        # Build groups
        axes_info = []
        groups: dict[int, list[tuple[int, float]]] = {}
        for i in range(n_comp):
            explained = (S[i] ** 2 / (n - 1)) / total_var if total_var > 0 else 0.0
            proj_1d = projections[:, i]
            axes_info.append({
                "index": i,
                "explained_variance": float(explained),
                "positive_pole_idx": int(np.argmax(proj_1d)),
                "negative_pole_idx": int(np.argmin(proj_1d)),
            })

        for doc_idx, axis_idx in enumerate(assignments):
            groups.setdefault(int(axis_idx), []).append(
                (doc_idx, float(projections[doc_idx, axis_idx]))
            )

        # Sort within groups by projection magnitude
        for axis_idx in groups:
            groups[axis_idx].sort(key=lambda x: abs(x[1]), reverse=True)

        # Format as grouped context
        context = self._format_grouped(results, groups, axes_info)

        return HypothesisResult(
            results=results,
            context_prompt=context,
            metadata={
                "pca_axes": axes_info,
                "group_assignments": assignments.tolist(),
            },
        )

    def _format_grouped(
        self,
        results: list[RetrievalResult],
        groups: dict[int, list[tuple[int, float]]],
        axes_info: list[dict],
    ) -> str:
        """Format results grouped by semantic axis."""
        lines = ["Retrieved context (grouped by semantic dimension):"]

        for axis_idx in sorted(groups.keys()):
            info = axes_info[axis_idx]
            var_pct = info["explained_variance"] * 100
            lines.append(f"\n── Dimension {axis_idx + 1} ({var_pct:.1f}% of variance) ──")

            for doc_idx, projection in groups[axis_idx]:
                r = results[doc_idx]
                pole = "+" if projection > 0 else "-"
                lines.append(f"  [{pole}{abs(projection):.2f}] (score: {r.score:.3f})")
                lines.append(f"  {r.text}")

        return "\n".join(lines)

    def _format_flat(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
