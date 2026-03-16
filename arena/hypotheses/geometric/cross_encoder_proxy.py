"""Cross-encoder proxy using bi-encoder interaction features.

Hypothesis: Cosine similarity between bi-encoder embeddings only captures
first-order alignment (do these vectors point the same way?). A true
cross-encoder would compute token-level interactions, capturing richer
patterns. We can approximate some of this by computing element-wise
interaction features between query and document embeddings — Hadamard
product, absolute difference, and sum — then combining them into a
richer similarity score.

This captures effects that cosine similarity misses: dimensions where
query and document both have large values (strong agreement), dimensions
where they disagree (potential mismatch signals), and dimensions where
their combined magnitude is high (shared importance).

Algorithm:
1. For each (query, document) pair, compute three feature vectors:
   - Hadamard product: q * d (element-wise multiplication)
   - Absolute difference: |q - d|
   - Element-wise sum: q + d
2. Compute a combined score from these features:
   score = w1 * mean(q*d) + w2 * (1 - mean(|q-d|)) + w3 * var(q+d)
3. The Hadamard product captures dimension-wise agreement
4. The absolute difference captures mismatch (inverted)
5. The sum variance captures shared dimensional importance
6. Re-rank by the combined interaction score

Geometric intuition: Cosine similarity projects the relationship onto a
single number (the angle). Our interaction features preserve per-dimension
information, allowing us to weight dimensions by their informativeness.
Dimensions where both query and document are "active" (large Hadamard
product) are more informative than dimensions where only one is active.

References:
  - Reimers & Gurevych (2019): Sentence-BERT (bi-encoder baseline)
  - Nogueira & Cho (2019): Cross-encoder reranking
  - Humeau et al. (2020): Poly-encoders (intermediate approach)
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class CrossEncoderProxyHypothesis(Hypothesis):
    """Approximate cross-encoder reranking using bi-encoder interaction features."""

    def __init__(
        self,
        w_hadamard: float = 0.5,
        w_diff: float = 0.3,
        w_sum_var: float = 0.2,
    ):
        """
        Args:
            w_hadamard: Weight for Hadamard product feature (agreement).
            w_diff: Weight for absolute difference feature (mismatch penalty).
            w_sum_var: Weight for sum variance feature (shared importance).
        """
        self.w_hadamard = w_hadamard
        self.w_diff = w_diff
        self.w_sum_var = w_sum_var

    @property
    def name(self) -> str:
        return "cross-encoder-proxy"

    @property
    def description(self) -> str:
        return (
            "Cross-encoder proxy — approximate cross-encoder reranking "
            "using element-wise bi-encoder interaction features"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 3:
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
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # Original cosine similarities for comparison
        cosine_sims = E @ q

        # Compute interaction features for each document
        interaction_scores = np.zeros(n)
        feature_details = []

        for i in range(n):
            d = E[i]

            # Feature 1: Hadamard product (element-wise agreement)
            hadamard = q * d
            hadamard_score = float(np.mean(hadamard))

            # Feature 2: Absolute difference (mismatch — lower is better)
            abs_diff = np.abs(q - d)
            diff_score = float(1.0 - np.mean(abs_diff))

            # Feature 3: Sum variance (shared dimensional importance)
            elem_sum = q + d
            sum_var_score = float(np.var(elem_sum))

            # Combined score
            score = (
                self.w_hadamard * hadamard_score
                + self.w_diff * diff_score
                + self.w_sum_var * sum_var_score
            )
            interaction_scores[i] = score

            feature_details.append({
                "hadamard": hadamard_score,
                "diff": diff_score,
                "sum_var": sum_var_score,
            })

        # Normalise interaction scores to [0, 1] for interpretability
        score_min = interaction_scores.min()
        score_max = interaction_scores.max()
        if score_max - score_min > 1e-12:
            interaction_scores_norm = (interaction_scores - score_min) / (score_max - score_min)
        else:
            interaction_scores_norm = np.ones(n) * 0.5

        # Re-rank by interaction score
        new_order = np.argsort(interaction_scores)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        # Correlation between interaction scores and cosine similarity
        if np.std(cosine_sims) > 1e-12 and np.std(interaction_scores) > 1e-12:
            rank_correlation = float(np.corrcoef(cosine_sims, interaction_scores)[0, 1])
        else:
            rank_correlation = 1.0

        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "interaction_scores": interaction_scores[new_order].tolist(),
                "interaction_scores_normalised": interaction_scores_norm[new_order].tolist(),
                "cosine_similarities": cosine_sims[new_order].tolist(),
                "cosine_interaction_correlation": rank_correlation,
                "rank_changes": rank_changes,
                "feature_weights": {
                    "hadamard": self.w_hadamard,
                    "diff": self.w_diff,
                    "sum_var": self.w_sum_var,
                },
                "top_feature_breakdown": [feature_details[i] for i in new_order[:3]],
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (cross-encoder proxy reranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
