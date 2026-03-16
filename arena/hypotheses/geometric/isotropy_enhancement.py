"""Isotropy enhancement via ZCA whitening.

Hypothesis: Embedding spaces are highly anisotropic — vectors cluster
in a narrow cone, meaning most dimensions are underutilised. Mean-bias
correction (centering) removes the first-order bias but leaves
higher-order correlations between dimensions intact. ZCA (Zero-phase
Component Analysis) whitening fully decorrelates the embedding space,
making all directions equally informative for similarity computation.

This is strictly more powerful than mean-bias correction: centering is
to ZCA as removing the mean is to full standardisation.

Algorithm:
1. Centre embeddings (subtract mean)
2. Compute covariance matrix of centred embeddings
3. Eigendecompose: Cov = V diag(lambda) V^T
4. Apply ZCA whitening: W = V diag(1/sqrt(lambda + eps)) V^T
5. Transform all embeddings and query: x' = W @ x
6. Re-normalise and recompute cosine similarities

Geometric intuition: ZCA whitening rotates the embedding space so that
the covariance becomes the identity matrix. Directions with high variance
(which dominate cosine similarity) are compressed, while low-variance
directions (which may carry discriminative semantic signal) are amplified.
Unlike PCA whitening, ZCA preserves the original orientation of the space.

References:
  - Mu & Viswanath (2018): All-but-the-Top — removing dominant directions
  - Su et al. (2021): Whitening sentence representations for similarity
  - arXiv:2511.11041 (Nov 2025): mean-bias correction (this extends it)
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class IsotropyEnhancementHypothesis(Hypothesis):
    """ZCA whitening to make the embedding space isotropic before re-ranking."""

    def __init__(self, regularisation: float = 1e-4, n_components: int | None = None):
        """
        Args:
            regularisation: Epsilon added to eigenvalues for numerical stability.
            n_components: If set, only whiten along top-n eigenvalue directions
                         (partial whitening). None = full whitening.
        """
        self.regularisation = regularisation
        self.n_components = n_components

    @property
    def name(self) -> str:
        return "isotropy-enhancement"

    @property
    def description(self) -> str:
        return (
            "ZCA whitening for isotropy enhancement — fully decorrelate "
            "embedding dimensions to make cosine similarity more discriminative"
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

        # Step 1: Centre embeddings
        mean_emb = embeddings.mean(axis=0)
        E_centred = embeddings - mean_emb
        q_centred = query_embedding - mean_emb

        # Step 2: Compute covariance matrix
        # Use min(n, d) to handle both cases efficiently
        cov = (E_centred.T @ E_centred) / max(n - 1, 1)

        # Step 3: Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Measure anisotropy before whitening
        eigenvalues_pos = np.maximum(eigenvalues, 0)
        total_var = eigenvalues_pos.sum()
        if total_var > 0:
            anisotropy_ratio = float(eigenvalues_pos[0] / total_var)
            top5_ratio = float(eigenvalues_pos[:5].sum() / total_var) if len(eigenvalues_pos) >= 5 else 1.0
        else:
            anisotropy_ratio = 0.0
            top5_ratio = 0.0

        # Step 4: ZCA whitening transform
        # W = V @ diag(1/sqrt(lambda + eps)) @ V^T
        n_comp = self.n_components or len(eigenvalues)
        n_comp = min(n_comp, len(eigenvalues))

        V = eigenvectors[:, :n_comp]
        lam = eigenvalues[:n_comp]
        scale = 1.0 / np.sqrt(np.maximum(lam, 0) + self.regularisation)

        # Apply whitening: x' = V @ diag(scale) @ V^T @ x
        # For efficiency: W_half = V @ diag(scale) @ V^T
        # But we can compute it as: x' = V @ (diag(scale) @ (V^T @ x))
        E_whitened = E_centred @ V * scale[np.newaxis, :]
        E_whitened = E_whitened @ V.T
        q_whitened = (q_centred @ V * scale) @ V.T

        # Step 5: Re-normalise
        e_norms = np.linalg.norm(E_whitened, axis=1, keepdims=True)
        e_norms = np.maximum(e_norms, 1e-12)
        E_whitened = E_whitened / e_norms

        q_norm = np.linalg.norm(q_whitened)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "degenerate query after whitening"},
            )
        q_whitened = q_whitened / q_norm

        # Step 6: Recompute similarities
        new_sims = E_whitened @ q_whitened

        new_order = np.argsort(new_sims)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        # Original similarities for comparison
        E_orig_norm = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        q_orig_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
        orig_sims = E_orig_norm @ q_orig_norm

        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "anisotropy_ratio_top1": anisotropy_ratio,
                "anisotropy_ratio_top5": top5_ratio,
                "condition_number": float(eigenvalues[0] / max(eigenvalues[-1], 1e-12)),
                "n_components_used": n_comp,
                "rank_changes": rank_changes,
                "whitened_similarities": new_sims[new_order].tolist(),
                "original_similarities": orig_sims[new_order].tolist(),
                "regularisation": self.regularisation,
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (isotropy-enhanced):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
