"""Spectral reranking via Fiedler vector embedding (Novel #8).

Unlike spectral-gap clustering (which detects cluster boundaries via
eigengaps), this hypothesis uses the graph Laplacian's Fiedler vector
(2nd smallest eigenvector) as a 1-D spectral embedding of the documents.

Geometric intuition
-------------------
The Fiedler vector assigns each document a real-valued coordinate that
captures its position along the "widest cut" of the similarity graph.
Documents at opposite extremes of the Fiedler vector are maximally
separated in graph-diffusion distance.  By selecting documents that are
evenly spaced along the Fiedler vector, we guarantee topical coverage
across the full breadth of the retrieved set — without needing to
explicitly cluster or pick k.

Algorithm
---------
1. Build cosine similarity graph from embeddings.
2. Compute normalised graph Laplacian.
3. Extract Fiedler vector (eigenvector of 2nd smallest eigenvalue).
4. Partition the Fiedler coordinate range into equal-width bins.
5. From each bin, select the document with highest query relevance.
6. Fill remaining slots by relevance from under-represented bins.

This differs from spectral_gap.py which uses eigengaps for clustering.
Here we use the spectral embedding directly for balanced spatial sampling.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class SpectralRerankingHypothesis(Hypothesis):
    """Rerank by evenly sampling the Fiedler vector for topical balance."""

    def __init__(self, n_bins: int = 5):
        """
        Args:
            n_bins: Number of equal-width bins along the Fiedler vector.
                Controls granularity of coverage.
        """
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return f"spectral-rerank-{self.n_bins}b"

    @property
    def description(self) -> str:
        return (
            "Fiedler vector reranking — evenly sample the spectral embedding "
            "of the similarity graph for balanced topical coverage"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or len(results) < 4:
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

        # Similarity matrix (clipped to non-negative)
        S = E @ E.T
        S = np.maximum(S, 0.0)

        # Normalised Laplacian: L_sym = I - D^{-1/2} S D^{-1/2}
        degrees = S.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.maximum(np.sqrt(degrees), 1e-12))
        L_sym = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt

        # Eigendecomposition (sorted ascending)
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        # Fiedler vector = 2nd smallest eigenvector
        fiedler = eigenvectors[:, 1]

        # Bin documents along the Fiedler coordinate
        f_min, f_max = float(fiedler.min()), float(fiedler.max())
        f_range = f_max - f_min
        if f_range < 1e-12:
            # Degenerate case: all Fiedler values identical
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "degenerate_fiedler"},
            )

        n_bins = min(self.n_bins, n)
        bin_width = f_range / n_bins

        # Assign each document to a bin
        bin_assignments = np.clip(
            ((fiedler - f_min) / bin_width).astype(int), 0, n_bins - 1
        )

        # Build bin -> doc mapping
        bins: dict[int, list[int]] = {b: [] for b in range(n_bins)}
        for i in range(n):
            bins[bin_assignments[i]].append(i)

        # From each bin, select by relevance score (descending)
        for b in bins:
            bins[b].sort(key=lambda i: results[i].score, reverse=True)

        # Round-robin selection across bins
        selected: list[int] = []
        selected_set: set[int] = set()
        pointers = {b: 0 for b in range(n_bins)}

        while len(selected) < n:
            added = False
            for b in range(n_bins):
                while pointers[b] < len(bins[b]):
                    idx = bins[b][pointers[b]]
                    pointers[b] += 1
                    if idx not in selected_set:
                        selected.append(idx)
                        selected_set.add(idx)
                        added = True
                        break
            if not added:
                break

        # Append any remaining (should not happen, but safety)
        for i in range(n):
            if i not in selected_set:
                selected.append(i)

        reranked = [results[i] for i in selected]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "fiedler_values": fiedler.tolist(),
                "fiedler_eigenvalue": float(eigenvalues[1]),
                "bin_sizes": {b: len(bins[b]) for b in range(n_bins)},
                "selected_indices": selected,
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (Fiedler-balanced selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
