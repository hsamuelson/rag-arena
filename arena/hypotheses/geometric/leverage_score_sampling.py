"""Leverage score sampling reranking (Novel #9).

From randomised numerical linear algebra (Mahoney & Drineas, 2009;
Woodruff, 2014).  Statistical leverage scores measure how "structurally
important" each row (document) is to the column space of the embedding
matrix.

Geometric intuition
-------------------
The leverage score of document i is the i-th diagonal entry of the hat
matrix H = X (X^T X)^{-1} X^T.  Intuitively, a document has high leverage
when it lies in a direction of embedding space that is poorly represented
by other documents.  Removing a high-leverage document would collapse an
entire dimension of the column space — it carries irreplaceable structural
information.

Conversely, low-leverage documents are well-explained by linear
combinations of other documents (they are redundant).

Algorithm
---------
1. Compute thin SVD of the (centred) embedding matrix: X = U S V^T.
2. Leverage score of row i = ||U[i, :]||^2.
3. Combine leverage with query relevance: final_score = relevance^alpha *
   leverage^(1-alpha).
4. Rank by final_score descending.

This gives a principled balance between relevance and structural importance.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class LeverageScoreSamplingHypothesis(Hypothesis):
    """Rerank documents by statistical leverage scores from the hat matrix."""

    def __init__(self, alpha: float = 0.6, rank_fraction: float = 0.8):
        """
        Args:
            alpha: Blending weight between relevance (alpha) and leverage
                (1-alpha).  alpha=1 is pure relevance; alpha=0 is pure
                leverage.
            rank_fraction: Fraction of singular values to keep when
                computing leverage scores.  Lower values emphasise the
                most important structural directions.
        """
        self.alpha = alpha
        self.rank_fraction = rank_fraction

    @property
    def name(self) -> str:
        return f"leverage-scores-{self.alpha}a"

    @property
    def description(self) -> str:
        return (
            "Leverage score reranking — weight documents by their structural "
            "importance to the embedding matrix column space (hat matrix diagonal)"
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

        n, d = embeddings.shape

        # Centre the embeddings (remove mean bias)
        X = embeddings - embeddings.mean(axis=0, keepdims=True)

        # Thin SVD
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "svd_failed"},
            )

        # Truncate to top-k singular values
        rank = max(1, int(len(S) * self.rank_fraction))
        U_k = U[:, :rank]

        # Leverage scores = row norms of U_k squared
        leverage = np.sum(U_k ** 2, axis=1)

        # Normalise leverage to [0, 1]
        lev_max = leverage.max()
        if lev_max > 1e-12:
            leverage_norm = leverage / lev_max
        else:
            leverage_norm = np.ones(n) / n

        # Relevance scores normalised to [0, 1]
        scores = np.array([r.score for r in results], dtype=np.float64)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-12:
            rel_norm = (scores - s_min) / (s_max - s_min)
        else:
            rel_norm = np.ones(n)

        # Combined score
        combined = (rel_norm ** self.alpha) * (leverage_norm ** (1.0 - self.alpha))

        # Rank by combined score
        order = np.argsort(-combined).tolist()
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": order,
                "leverage_scores": leverage.tolist(),
                "combined_scores": combined.tolist(),
                "effective_rank": rank,
                "singular_values_top5": S[:min(5, len(S))].tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (leverage-score reranking):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
