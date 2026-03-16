"""Density Peak Selection reranking (Novel #6).

Based on Rodriguez & Laio (2014) "Clustering by fast search and find of
density peaks" (Science 344:1492).  The core idea: cluster centres are
documents that are (a) surrounded by many neighbours (high local density rho)
AND (b) far from any other document with even higher density (high delta).

Geometric intuition
-------------------
In embedding space, retrieved documents form irregular clumps.  Plain
cosine-top-K over-represents the densest clump.  Density-peak analysis
identifies the *centre* of every distinct clump — these are the documents
with the highest rho * delta product.  Selecting peaks first, then padding
with their nearest neighbours, guarantees that every information thread is
represented while still favouring locally popular (high-density) content.

Algorithm
---------
1. Compute pairwise distances in embedding space.
2. Set an adaptive cut-off radius d_c (median of pairwise distances).
3. rho_i = number of documents within d_c of document i.
4. delta_i = min distance to any document j with rho_j > rho_i.
   (For the globally densest document, delta = max pairwise distance.)
5. Rank by gamma_i = rho_i * delta_i  (descending).
6. Select top-k peaks, then fill remaining slots with nearest
   neighbours of each peak in round-robin order.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class DensityPeakSelectionHypothesis(Hypothesis):
    """Select cluster-centre documents via density-peak analysis."""

    def __init__(self, n_peaks: int = 3, dc_percentile: float = 50.0):
        """
        Args:
            n_peaks: Maximum number of density peaks to select.
            dc_percentile: Percentile of pairwise distances used as the
                adaptive cut-off radius d_c.
        """
        self.n_peaks = n_peaks
        self.dc_percentile = dc_percentile

    @property
    def name(self) -> str:
        return f"density-peaks-{self.n_peaks}p"

    @property
    def description(self) -> str:
        return (
            "Density peak clustering — select documents that are both locally "
            "dense and far from other dense regions (Rodriguez-Laio 2014)"
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

        # Normalise embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        # Pairwise cosine distances (1 - similarity)
        sim = E @ E.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)

        # Adaptive cut-off radius
        upper_tri = dist[np.triu_indices(n, k=1)]
        if len(upper_tri) == 0:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        d_c = float(np.percentile(upper_tri, self.dc_percentile))
        d_c = max(d_c, 1e-8)

        # Local density rho: count of neighbours within d_c
        rho = np.sum(dist < d_c, axis=1).astype(float)
        # Subtract self-count
        rho -= 1.0
        rho = np.maximum(rho, 0.0)

        # Delta: distance to nearest neighbour with higher density
        delta = np.full(n, 0.0)
        nearest_higher = np.full(n, -1, dtype=int)
        sorted_idx = np.argsort(-rho)  # descending density

        for rank, i in enumerate(sorted_idx):
            if rank == 0:
                # Highest density point: delta = max distance
                delta[i] = float(np.max(dist[i]))
                nearest_higher[i] = -1
                continue
            higher = sorted_idx[:rank]
            dists_to_higher = dist[i, higher]
            min_idx = int(np.argmin(dists_to_higher))
            delta[i] = float(dists_to_higher[min_idx])
            nearest_higher[i] = int(higher[min_idx])

        # Gamma score = rho * delta
        gamma = rho * delta

        # Select peaks
        n_peaks = min(self.n_peaks, n)
        peak_indices = np.argsort(-gamma)[:n_peaks].tolist()

        # Build final ordering: peaks first, then fill with nearest
        # neighbours of each peak in round-robin
        selected = list(peak_indices)
        remaining = set(range(n)) - set(selected)

        # For each peak, rank remaining docs by distance to that peak
        peak_neighbours: dict[int, list[int]] = {}
        for p in peak_indices:
            candidates = sorted(remaining, key=lambda j: dist[p, j])
            peak_neighbours[p] = candidates

        # Round-robin fill
        pointers = {p: 0 for p in peak_indices}
        while remaining and len(selected) < n:
            added = False
            for p in peak_indices:
                while pointers[p] < len(peak_neighbours[p]):
                    cand = peak_neighbours[p][pointers[p]]
                    pointers[p] += 1
                    if cand in remaining:
                        selected.append(cand)
                        remaining.discard(cand)
                        added = True
                        break
            if not added:
                break

        # Append any leftover
        for idx in range(n):
            if idx not in set(selected):
                selected.append(idx)

        reranked = [results[i] for i in selected]
        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "peak_indices": peak_indices,
                "d_c": d_c,
                "rho": rho.tolist(),
                "delta": delta.tolist(),
                "gamma": gamma.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (density-peak selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
