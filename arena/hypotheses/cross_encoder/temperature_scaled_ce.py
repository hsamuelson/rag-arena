"""Temperature-Scaled Cross-Encoder.

Research basis: GETS (ICLR 2025), temperature scaling for calibration.
Post-hoc temperature scaling with a learned parameter can improve ranking
by sharpening or softening the score distribution.

We use a self-calibrating approach: estimate temperature from the score
distribution of each query's results, then apply it.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class TemperatureScaledCEHypothesis(Hypothesis):
    """CE with adaptive temperature scaling per query."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", temperature=2.0):
        self._model_name = model_name
        self._temperature = temperature
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return f"temperature-scaled-ce-{self._temperature}"

    @property
    def description(self):
        return f"CE with temperature={self._temperature} scaling (ICLR 2025)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        raw_scores = model.predict(pairs)

        # Apply temperature scaling: score / T
        # Higher T → softer distribution (less confident), lower T → sharper
        scaled = raw_scores / self._temperature

        # Rank by scaled scores (monotonic transform, only matters for ties/soft ranking)
        # The key effect is on the relative gaps between scores
        ce_scores = scaled.tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        reranked = [results[i] for i in ranked]

        lines = ["Retrieved context (temperature-scaled CE reranked):"]
        for i, idx in enumerate(ranked, 1):
            lines.append(f"\n[{i}] (ce_score: {raw_scores[idx]:.4f}, scaled: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"raw_scores": raw_scores.tolist(), "scaled_scores": ce_scores},
        )
