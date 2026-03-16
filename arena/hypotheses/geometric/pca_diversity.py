"""PCA-MMR diversity reranking (from SEMDA/Caldera research).

Hypothesis: Retrieval quality improves when we select documents that
cover diverse principal components of the embedding space, rather
than just the nearest neighbours.

Algorithm:
1. Retrieve 2K candidates via cosine top-K
2. Run PCA on candidate embeddings
3. Use MMR (Maximal Marginal Relevance) with PCA projections
   to select K documents maximising both relevance and diversity
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class PCADiversityHypothesis(Hypothesis):
    """PCA-based diverse retrieval — rerank for coverage across principal axes."""

    def __init__(self, n_components: int = 3, diversity_weight: float = 0.3):
        self.n_components = n_components
        self.diversity_weight = diversity_weight  # lambda in MMR: 0=pure relevance, 1=pure diversity

    @property
    def name(self) -> str:
        return f"pca-diversity-{self.n_components}c-{self.diversity_weight}w"

    @property
    def description(self) -> str:
        return (
            f"PCA-MMR reranking: {self.n_components} components, "
            f"diversity weight {self.diversity_weight}"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or len(results) < 3:
            # Fall back to flat if no embeddings available
            context = self._format_flat(results)
            return HypothesisResult(
                results=results, context_prompt=context,
                metadata={"fallback": True},
            )

        # PCA decomposition
        pca_data = self._pca_analyse(embeddings)
        projections = pca_data["projections"]

        # MMR reranking using PCA projections
        reranked_indices = self._mmr_rerank(
            similarities=[r.score for r in results],
            projections=projections,
            k=len(results),
        )

        reranked_results = [results[i] for i in reranked_indices]
        context = self._format_flat(reranked_results)

        return HypothesisResult(
            results=reranked_results,
            context_prompt=context,
            metadata={
                "pca_axes": pca_data["axes"],
                "rerank_order": reranked_indices,
            },
        )

    def _pca_analyse(self, embeddings: np.ndarray) -> dict:
        """PCA decomposition — mirrors Basin's pca.rs logic."""
        n, d = embeddings.shape
        n_comp = min(self.n_components, n - 1, d)
        if n_comp < 1:
            return {"axes": [], "projections": np.zeros((n, 0))}

        mean = embeddings.mean(axis=0)
        centred = embeddings - mean

        # Economy SVD (faster than eigendecomposition for K << D)
        U, S, Vt = np.linalg.svd(centred, full_matrices=False)

        total_var = (S ** 2).sum() / (n - 1) if n > 1 else 1.0
        axes = []
        projections = centred @ Vt[:n_comp].T

        for i in range(n_comp):
            explained = (S[i] ** 2 / (n - 1)) / total_var if total_var > 0 else 0.0
            proj_1d = projections[:, i]
            axes.append({
                "index": i,
                "explained_variance": float(explained),
                "positive_pole_idx": int(np.argmax(proj_1d)),
                "negative_pole_idx": int(np.argmin(proj_1d)),
            })

        return {"axes": axes, "projections": projections}

    def _mmr_rerank(
        self,
        similarities: list[float],
        projections: np.ndarray,
        k: int,
    ) -> list[int]:
        """Maximal Marginal Relevance using PCA projections for diversity."""
        n = len(similarities)
        if n == 0:
            return []

        selected: list[int] = []
        remaining = set(range(n))
        lam = 1.0 - self.diversity_weight

        # Normalise projections for diversity distance
        if projections.shape[1] > 0:
            proj_norms = np.linalg.norm(projections, axis=1, keepdims=True)
            proj_norms = np.maximum(proj_norms, 1e-12)
            proj_normed = projections / proj_norms
        else:
            proj_normed = projections

        for _ in range(min(k, n)):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = similarities[idx]

                if not selected or projections.shape[1] == 0:
                    diversity = 0.0
                else:
                    # Max similarity to already-selected docs in PCA space
                    selected_projs = proj_normed[selected]
                    cos_sims = selected_projs @ proj_normed[idx]
                    diversity = float(np.max(cos_sims))

                mmr_score = lam * relevance - (1 - lam) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected

    def _format_flat(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (PCA-diversity reranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
