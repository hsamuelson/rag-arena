"""Calibrated cross-encoder hypothesis.

Cross-encoder scores are not calibrated — a score of 5.0 might mean
"highly relevant" for one query but "mediocre" for another. This
hypothesis calibrates CE scores using the score distribution of the
current result set.

Approach:
1. Score all candidates with CE
2. Compute z-scores: z = (score - mean) / std
3. Apply softmax to get calibrated probabilities
4. Rank by calibrated score

Additionally, penalise documents that are "easy" (high retrieval score
but mediocre CE score) — these are likely false positives from lexical overlap.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CECalibratedHypothesis(Hypothesis):
    """Cross-encoder with score calibration and false-positive penalty."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        temperature: float = 1.0,
        fp_penalty: float = 0.1,
    ):
        self._model_name = model_name
        self._temperature = temperature
        self._fp_penalty = fp_penalty
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-calibrated-t{self._temperature}"

    @property
    def description(self) -> str:
        return "Cross-encoder with z-score calibration and false-positive penalty"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # Z-score calibration
        mean = ce_scores.mean()
        std = ce_scores.std()
        if std < 1e-12:
            z_scores = np.zeros_like(ce_scores)
        else:
            z_scores = (ce_scores - mean) / std

        # Softmax with temperature
        scaled = z_scores / self._temperature
        scaled -= scaled.max()  # numerical stability
        calibrated = np.exp(scaled) / np.exp(scaled).sum()

        # False-positive penalty: docs with high retrieval score but low CE score
        ret_scores = np.array([r.score for r in results], dtype=np.float64)
        ret_norm = ret_scores / (ret_scores.max() + 1e-12)
        ce_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-12)

        # Disagreement: high retrieval, low CE = likely false positive
        disagreement = np.maximum(0, ret_norm - ce_norm)
        penalty = self._fp_penalty * disagreement

        final_scores = calibrated - penalty

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (calibrated CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (cal: {final_scores[idx]:.4f}, ce: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
                "calibrated_scores": [float(final_scores[i]) for i in ranked_indices],
                "z_scores": [float(z_scores[i]) for i in ranked_indices],
                "fp_penalties": [float(penalty[i]) for i in ranked_indices],
            },
        )
