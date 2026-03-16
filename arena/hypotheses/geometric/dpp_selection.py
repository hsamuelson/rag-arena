"""Determinantal Point Process (DPP) selection.

Hypothesis: Selecting a diverse subset via DPP (which maximises the volume
of the parallelotope spanned by selected embeddings) produces better
coverage of the answer space than greedy MMR.

Geometric intuition: det(L_Y) = squared volume of the parallelotope formed
by the selected vectors. High volume = high diversity. Unlike MMR (which
makes greedy local decisions), DPP evaluates subset quality globally.

References:
  - Kulesza & Tassos (2012): Determinantal Point Processes for ML
  - MS-DPPs (IJCAI 2025): multi-source DPPs for contextual diversity
  - Reliability-Aware DPPs (arXiv, Feb 2025): DPP for RAG context budgets
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class DPPSelectionHypothesis(Hypothesis):
    """Select diverse subsets using a Determinantal Point Process kernel."""

    def __init__(self, subset_size: int | None = None, relevance_weight: float = 0.7):
        self._subset_size = subset_size  # None = same as input
        self.relevance_weight = relevance_weight

    @property
    def name(self) -> str:
        return f"dpp-selection-{self.relevance_weight}w"

    @property
    def description(self) -> str:
        return (
            "DPP-based diverse subset selection — maximises parallelotope volume "
            "in embedding space, weighted by relevance"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or len(results) < 3:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        k = self._subset_size or len(results)
        k = min(k, len(results))

        # Build the DPP L-ensemble kernel: L = diag(q) @ S @ diag(q)
        # where q_i = relevance score, S_ij = cosine similarity
        selected = self._greedy_dpp(embeddings, [r.score for r in results], k)

        reranked = [results[i] for i in selected]
        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": selected,
                "kernel_logdet": self._log_det_subset(embeddings, selected),
            },
        )

    def _greedy_dpp(
        self,
        embeddings: np.ndarray,
        relevances: list[float],
        k: int,
    ) -> list[int]:
        """Greedy MAP inference for DPP (fast approximation).

        Uses the greedy algorithm from Chen et al. (2018) which iteratively
        selects the item maximising the marginal gain in log-determinant.
        """
        n, d = embeddings.shape

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        V = embeddings / norms

        # Quality scores (from retrieval relevance)
        q = np.array(relevances, dtype=np.float64)
        q = np.maximum(q, 1e-12)

        # Build quality-weighted feature vectors: phi_i = q_i^alpha * v_i
        alpha = self.relevance_weight
        phi = V * (q ** alpha)[:, np.newaxis]

        # Greedy selection using Cholesky-based incremental log-det
        selected: list[int] = []
        remaining = set(range(n))

        # Cache for incremental Cholesky
        # L_inv_phi[i] stores L^{-1} @ phi[i] for selected items
        L_rows: list[np.ndarray] = []  # rows of Cholesky factor

        for _ in range(k):
            best_idx = -1
            best_gain = -float("inf")

            for idx in remaining:
                # Marginal gain = phi_idx^T phi_idx - sum_j (L_inv_phi_j^T phi_idx)^2
                phi_i = phi[idx]

                if not selected:
                    gain = float(phi_i @ phi_i)
                else:
                    # Compute projections onto already-selected subspace
                    proj_sq_sum = 0.0
                    for row in L_rows:
                        proj = float(row @ phi_i)
                        proj_sq_sum += proj * proj
                    gain = float(phi_i @ phi_i) - proj_sq_sum

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx < 0 or best_gain < 1e-12:
                break

            # Update Cholesky factor
            phi_sel = phi[best_idx]
            if not L_rows:
                L_rows.append(phi_sel / np.sqrt(phi_sel @ phi_sel))
            else:
                # Compute new row of L
                projections = np.array([float(row @ phi_sel) for row in L_rows])
                residual = phi_sel - sum(p * row for p, row in zip(projections, L_rows))
                res_norm = np.sqrt(float(residual @ residual))
                if res_norm > 1e-12:
                    L_rows.append(residual / res_norm)

            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected

    def _log_det_subset(self, embeddings: np.ndarray, indices: list[int]) -> float:
        """Compute log-determinant of the kernel submatrix (quality metric)."""
        if not indices:
            return 0.0
        sub = embeddings[indices]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        sub_normed = sub / norms
        K = sub_normed @ sub_normed.T
        sign, logdet = np.linalg.slogdet(K)
        return float(logdet) if sign > 0 else -float("inf")

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (DPP-diverse selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
