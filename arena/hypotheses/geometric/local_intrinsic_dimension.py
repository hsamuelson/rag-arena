"""Local Intrinsic Dimensionality (LID) weighting.

Hypothesis: Documents in low-LID regions of the embedding space are
more "confident" retrievals — the manifold is locally flat and the
nearest-neighbor structure is trustworthy. Documents in high-LID
regions (locally complex manifold) have less reliable similarity
scores and should be down-weighted or flagged.

Geometric intuition: LID estimates the dimensionality of the data
manifold near a point. A point with LID=2 in a 768-d space means
the data locally lives on a 2-d surface. Low LID = simple local
structure = reliable cosine similarity. High LID = tangled manifold
= cosine similarity is less meaningful.

References:
  - Amsaleg et al. (2015): TwoNN estimator for intrinsic dimensionality
  - Tsukagoshi & Sasano (ACL Findings 2025): LID analysis of embeddings
  - Houle (2017): Local Intrinsic Dimensionality
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class LocalIntrinsicDimensionHypothesis(Hypothesis):
    """Weight retrieval results by local intrinsic dimensionality."""

    def __init__(self, lid_penalty: float = 0.3, k_neighbors: int = 5):
        self.lid_penalty = lid_penalty
        self.k_neighbors = k_neighbors

    @property
    def name(self) -> str:
        return f"lid-weighted-{self.lid_penalty}p"

    @property
    def description(self) -> str:
        return (
            "Local Intrinsic Dimensionality weighting — down-weight results "
            "in high-LID (manifold-complex) regions where cosine similarity is unreliable"
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
        k = min(self.k_neighbors, n - 1)
        if k < 2:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        # Compute pairwise distances
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        dists = 1.0 - E @ E.T  # cosine distance matrix
        np.fill_diagonal(dists, np.inf)

        # Estimate LID for each document using TwoNN estimator
        lids = self._estimate_lid_twonn(dists, k)

        # Normalise LIDs to [0, 1]
        lid_min, lid_max = lids.min(), lids.max()
        if lid_max - lid_min > 1e-6:
            lid_norm = (lids - lid_min) / (lid_max - lid_min)
        else:
            lid_norm = np.zeros_like(lids)

        # Adjusted scores: penalise high-LID regions
        original_scores = np.array([r.score for r in results])
        adjusted_scores = original_scores * (1.0 - self.lid_penalty * lid_norm)

        # Rerank by adjusted score
        order = np.argsort(adjusted_scores)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "lid_values": lids.tolist(),
                "lid_mean": float(lids.mean()),
                "lid_std": float(lids.std()),
                "lid_min": float(lid_min),
                "lid_max": float(lid_max),
                "rerank_order": order.tolist(),
            },
        )

    def _estimate_lid_twonn(self, dist_matrix: np.ndarray, k: int) -> np.ndarray:
        """Estimate LID using the TwoNN (Two Nearest Neighbors) method.

        For each point, LID ≈ log(2) / log(r2/r1) where r1, r2 are
        distances to the 1st and 2nd nearest neighbors.

        For more robust estimates, we use the maximum likelihood estimator
        over k neighbors: LID = -k / sum(log(d_j / d_k)) for j=1..k-1.
        """
        n = dist_matrix.shape[0]
        lids = np.zeros(n)

        for i in range(n):
            dists_i = np.sort(dist_matrix[i])[:k]
            dists_i = np.maximum(dists_i, 1e-12)  # avoid log(0)
            d_k = dists_i[-1]

            if d_k < 1e-10:
                lids[i] = 0.0
                continue

            # MLE estimator: LID = -m / sum(log(d_j / d_k))
            m = k - 1
            log_ratios = np.log(dists_i[:m] / d_k)
            sum_log = log_ratios.sum()

            if abs(sum_log) < 1e-12:
                lids[i] = 0.0
            else:
                lids[i] = -m / sum_log

        return lids

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (LID-weighted):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
