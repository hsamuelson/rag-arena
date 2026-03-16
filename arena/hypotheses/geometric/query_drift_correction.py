"""Query drift correction via pseudo-relevance feedback.

Hypothesis: The query embedding often "drifts" from the true information
need — it encodes the query's surface form rather than the underlying
semantic intent. By treating the top-K retrieved documents as pseudo-relevant,
we can estimate a better "ideal query point" and interpolate between the
original query and this estimate.

This is a geometric interpretation of Rocchio's relevance feedback:
the corrected query moves towards the centroid of relevant documents
and (optionally) away from non-relevant ones.

Algorithm:
1. Identify top-T results as pseudo-relevant (by original score)
2. Compute weighted centroid of pseudo-relevant embeddings
3. Interpolate: q_corrected = alpha * q_original + (1-alpha) * centroid
4. Optionally: push away from bottom-T results (negative feedback)
5. Re-normalise and recompute all similarities

Geometric intuition: The original query point sits somewhere near the
relevant region of the embedding space, but not necessarily at its centre.
Pseudo-relevance feedback nudges the query towards the dense core of
relevant documents, improving recall for borderline documents that were
semantically relevant but lexically different from the query.

References:
  - Rocchio (1971): Relevance feedback in information retrieval
  - Li et al. (2023): Pseudo-relevance feedback for dense retrieval
  - Yu et al. (2021): Improving query representations for dense retrieval
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class QueryDriftCorrectionHypothesis(Hypothesis):
    """Correct query embedding drift using pseudo-relevance feedback."""

    def __init__(
        self,
        alpha: float = 0.6,
        n_positive: int = 3,
        n_negative: int = 0,
        beta_negative: float = 0.1,
    ):
        """
        Args:
            alpha: Interpolation weight for original query (1.0 = no correction).
            n_positive: Number of top results to use as positive feedback.
            n_negative: Number of bottom results for negative feedback (0 = disabled).
            beta_negative: Weight for negative feedback subtraction.
        """
        self.alpha = alpha
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.beta_negative = beta_negative

    @property
    def name(self) -> str:
        return f"query-drift-a{self.alpha}"

    @property
    def description(self) -> str:
        return (
            "Query drift correction — pseudo-relevance feedback to move "
            "the query embedding towards the centroid of relevant documents"
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

        n = len(results)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # Original similarities
        orig_sims = E @ q

        # Identify pseudo-relevant documents (top-T by original similarity)
        n_pos = min(self.n_positive, n)
        top_indices = np.argsort(orig_sims)[::-1][:n_pos]

        # Weighted centroid of pseudo-relevant docs (weight by similarity)
        pos_weights = orig_sims[top_indices]
        pos_weights = np.maximum(pos_weights, 0)  # clip negatives
        weight_sum = pos_weights.sum()
        if weight_sum < 1e-12:
            pos_centroid = E[top_indices].mean(axis=0)
        else:
            pos_centroid = (E[top_indices] * pos_weights[:, np.newaxis]).sum(axis=0) / weight_sum

        # Corrected query: interpolate
        q_corrected = self.alpha * q + (1.0 - self.alpha) * pos_centroid

        # Optional negative feedback
        if self.n_negative > 0 and n > n_pos + self.n_negative:
            n_neg = min(self.n_negative, n - n_pos)
            bottom_indices = np.argsort(orig_sims)[:n_neg]
            neg_centroid = E[bottom_indices].mean(axis=0)
            q_corrected = q_corrected - self.beta_negative * neg_centroid

        # Re-normalise
        q_norm = np.linalg.norm(q_corrected)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True, "reason": "degenerate corrected query"},
            )
        q_corrected = q_corrected / q_norm

        # Recompute similarities with corrected query
        new_sims = E @ q_corrected

        # Measure drift
        drift_angle = float(np.arccos(np.clip(q @ q_corrected, -1.0, 1.0)))
        drift_degrees = float(np.degrees(drift_angle))

        new_order = np.argsort(new_sims)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "drift_angle_rad": drift_angle,
                "drift_angle_deg": drift_degrees,
                "alpha": self.alpha,
                "n_positive_used": n_pos,
                "n_negative_used": min(self.n_negative, max(0, n - n_pos)),
                "rank_changes": rank_changes,
                "corrected_similarities": new_sims[new_order].tolist(),
                "original_similarities": orig_sims[new_order].tolist(),
                "positive_centroid_sim_to_query": float(pos_centroid @ q / max(np.linalg.norm(pos_centroid), 1e-12)),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (query-drift corrected):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
