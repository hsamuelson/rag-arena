"""Cross-encoder with reciprocal nearest neighbor pre-filtering.

Before running the expensive cross-encoder, filter candidates using
the reciprocal nearest neighbor (RNN) criterion: a document d is a
reciprocal neighbor of query q if d is among q's nearest neighbors
AND q is among d's nearest neighbors (in embedding space).

RNN documents are more likely to be truly relevant (not hub artifacts).
We give RNN documents a bonus in the final CE scoring.

This combines the hub-correction insight (like CSLS) with CE reranking.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CEReciprocalNeighborHypothesis(Hypothesis):
    """Cross-encoder with reciprocal nearest neighbor bonus."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rnn_k: int = 10,
        rnn_bonus: float = 0.15,
    ):
        self._model_name = model_name
        self._rnn_k = rnn_k
        self._rnn_bonus = rnn_bonus
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-rnn-{self._rnn_k}k"

    @property
    def description(self) -> str:
        return f"Cross-encoder + reciprocal nearest neighbor bonus (k={self._rnn_k})"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # Normalize CE scores to [0, 1]
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max - ce_min > 1e-12:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_scores)

        # Compute RNN if embeddings available
        is_rnn = np.zeros(len(results), dtype=bool)
        if embeddings is not None and query_embedding is not None and len(embeddings) == len(results):
            n = len(results)
            k = min(self._rnn_k, n - 1)

            # Cosine similarities between all docs
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normed = embeddings / norms

            # Query-doc similarities (already ranked by these, but recalculate)
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
            q_sims = normed @ q_norm  # shape (n,)

            # Doc-doc similarities
            doc_sims = normed @ normed.T

            # For each doc, check if query is in its top-k neighbors
            # (approximate: check if query similarity to doc is above the doc's k-th NN similarity)
            for i in range(n):
                # Doc i's similarities to all other docs
                d_sims = doc_sims[i].copy()
                d_sims[i] = -1  # exclude self
                kth_sim = np.partition(d_sims, -k)[-k] if k > 0 else -1

                # If query's similarity to doc i is above the k-th neighbor threshold,
                # then query would be in doc i's neighborhood
                if q_sims[i] >= kth_sim:
                    is_rnn[i] = True

        # Apply RNN bonus
        final_scores = ce_norm + self._rnn_bonus * is_rnn.astype(np.float64)

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        n_rnn = int(is_rnn.sum())
        lines = [f"Retrieved context (CE + RNN, {n_rnn}/{len(results)} reciprocal neighbors):"]
        for i, idx in enumerate(ranked_indices, 1):
            rnn_marker = " [RNN]" if is_rnn[idx] else ""
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, ce: {ce_scores[idx]:.4f}{rnn_marker})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "is_rnn": [bool(is_rnn[i]) for i in ranked_indices],
                "n_reciprocal_neighbors": n_rnn,
            },
        )
