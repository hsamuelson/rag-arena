"""Cross-encoder with negative feedback.

After initial CE scoring, use the LOWEST-scoring documents as negative
examples. Re-score by penalizing documents similar to the negatives.

Intuition: if CE confidently says some docs are irrelevant, any doc
that looks similar to those irrelevant docs should also be penalized.

This is a form of "contrastive reranking" — pushing docs away from
known negatives.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CENegativeFeedbackHypothesis(Hypothesis):
    """CE with negative feedback: penalize docs similar to CE-identified negatives."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        n_negatives: int = 3,
        penalty_weight: float = 0.2,
    ):
        self._model_name = model_name
        self._n_negatives = n_negatives
        self._penalty_weight = penalty_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-negative-feedback-{self._n_negatives}n"

    @property
    def description(self) -> str:
        return f"CE + negative feedback: penalize docs similar to bottom-{self._n_negatives} CE results"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # Standard CE scoring
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # If no embeddings, just use CE scores
        if embeddings is None or len(embeddings) != n:
            ranked_indices = sorted(range(n), key=lambda i: ce_scores[i], reverse=True)
        else:
            # Identify negative documents (lowest CE scores)
            sorted_by_ce = sorted(range(n), key=lambda i: ce_scores[i])
            negative_indices = sorted_by_ce[:self._n_negatives]

            # Compute similarity to negatives
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normed = embeddings / norms

            # Average negative embedding
            neg_embs = normed[negative_indices]
            neg_centroid = neg_embs.mean(axis=0)
            neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-12)

            # Similarity to negative centroid
            neg_similarity = normed @ neg_centroid  # (n,)

            # Normalize CE scores to [0, 1]
            ce_min, ce_max = ce_scores.min(), ce_scores.max()
            if ce_max - ce_min > 1e-12:
                ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
            else:
                ce_norm = np.ones_like(ce_scores)

            # Penalty: high similarity to negatives = penalty
            # Normalize neg_similarity to [0, 1]
            ns_min, ns_max = neg_similarity.min(), neg_similarity.max()
            if ns_max - ns_min > 1e-12:
                neg_norm = (neg_similarity - ns_min) / (ns_max - ns_min)
            else:
                neg_norm = np.zeros_like(neg_similarity)

            final_scores = ce_norm - self._penalty_weight * neg_norm
            ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)

        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (CE + negative feedback):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
            },
        )
