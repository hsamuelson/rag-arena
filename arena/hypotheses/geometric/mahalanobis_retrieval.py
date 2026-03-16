"""Mahalanobis distance retrieval.

Hypothesis: Cosine similarity treats all embedding dimensions as equally
important and independent. In practice, dimensions are correlated — some
pairs of dimensions move together, and some dimensions have much higher
variance than others. The Mahalanobis distance accounts for these
correlations by using the inverse covariance matrix of the retrieved
embeddings as a metric tensor.

This provides a statistically grounded distance metric: if two dimensions
are highly correlated, movement along their shared direction is "cheap"
(less informative), while movement orthogonal to correlations is "expensive"
(more discriminative).

Algorithm:
1. Centre the retrieved embeddings (subtract mean)
2. Compute the covariance matrix
3. Regularise and invert: Sigma_inv = (Cov + eps * I)^{-1}
4. Compute Mahalanobis distance: d(q, d) = sqrt((q-d)^T Sigma_inv (q-d))
5. Convert to similarity: sim = 1 / (1 + d)
6. Re-rank by Mahalanobis similarity

Geometric intuition: Cosine similarity measures angles in a round
(Euclidean) space. Mahalanobis distance measures distances in an
ellipsoidal space defined by the data's covariance structure. Directions
of high variance (common variation among retrieved docs) contribute less
to the distance, focusing the metric on discriminative dimensions.

References:
  - Mahalanobis (1936): On the generalised distance in statistics
  - De Maesschalck et al. (2000): The Mahalanobis distance
  - Musgrave et al. (2020): Metric learning with Mahalanobis for retrieval
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class MahalanobisRetrievalHypothesis(Hypothesis):
    """Replace cosine similarity with Mahalanobis distance for reranking."""

    def __init__(self, regularisation: float = 1e-3, use_shrinkage: bool = True):
        """
        Args:
            regularisation: Tikhonov regularisation for covariance inversion.
            use_shrinkage: If True, use Ledoit-Wolf-style shrinkage toward
                          a diagonal covariance (better for n < d).
        """
        self.regularisation = regularisation
        self.use_shrinkage = use_shrinkage

    @property
    def name(self) -> str:
        return "mahalanobis-retrieval"

    @property
    def description(self) -> str:
        return (
            "Mahalanobis distance retrieval — use inverse covariance "
            "as metric tensor for correlation-aware similarity"
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

        n, d = embeddings.shape

        # Centre embeddings
        mean_emb = embeddings.mean(axis=0)
        E_centred = embeddings - mean_emb
        q_centred = query_embedding - mean_emb

        # Compute covariance matrix
        cov = (E_centred.T @ E_centred) / max(n - 1, 1)

        # Apply shrinkage if requested (helps when n << d)
        if self.use_shrinkage and n < d:
            shrinkage_target = np.diag(np.diag(cov))  # diagonal of covariance
            # Shrinkage intensity: higher when n/d ratio is lower
            shrinkage_alpha = min(1.0, max(0.1, 1.0 - n / d))
            cov = (1 - shrinkage_alpha) * cov + shrinkage_alpha * shrinkage_target

        # Regularise and invert
        cov_reg = cov + self.regularisation * np.eye(d)

        # Use eigendecomposition for numerically stable inversion
        eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
        eigenvalues = np.maximum(eigenvalues, self.regularisation)
        cov_inv = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

        # Compute Mahalanobis distances
        mahal_distances = np.zeros(n)
        for i in range(n):
            diff = q_centred - E_centred[i]
            mahal_sq = float(diff @ cov_inv @ diff)
            mahal_distances[i] = np.sqrt(max(mahal_sq, 0.0))

        # Convert to similarity (smaller distance = higher similarity)
        mahal_sims = 1.0 / (1.0 + mahal_distances)

        # Original cosine similarities for comparison
        E_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        q_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
        cosine_sims = E_norm @ q_norm

        # Re-rank by Mahalanobis similarity
        new_order = np.argsort(mahal_sims)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        # Compute condition number of covariance (measure of anisotropy)
        cov_eigenvalues = np.linalg.eigvalsh(cov)
        cov_eigenvalues_pos = cov_eigenvalues[cov_eigenvalues > 1e-12]
        if len(cov_eigenvalues_pos) > 0:
            condition_number = float(cov_eigenvalues_pos.max() / cov_eigenvalues_pos.min())
        else:
            condition_number = float("inf")

        # Correlation between Mahalanobis and cosine rankings
        if np.std(cosine_sims) > 1e-12 and np.std(mahal_sims) > 1e-12:
            rank_correlation = float(np.corrcoef(cosine_sims, mahal_sims)[0, 1])
        else:
            rank_correlation = 1.0

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "mahalanobis_similarities": mahal_sims[new_order].tolist(),
                "mahalanobis_distances": mahal_distances[new_order].tolist(),
                "cosine_similarities": cosine_sims[new_order].tolist(),
                "cosine_mahal_correlation": rank_correlation,
                "covariance_condition_number": condition_number,
                "rank_changes": rank_changes,
                "shrinkage_used": self.use_shrinkage and n < d,
                "regularisation": self.regularisation,
                "n_samples": n,
                "n_dimensions": d,
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (Mahalanobis-reranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
