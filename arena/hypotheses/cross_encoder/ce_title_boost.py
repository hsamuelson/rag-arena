"""Cross-encoder with title/entity boost.

For multi-hop QA (like HotpotQA), the document title often indicates
relevance better than the body text. This hypothesis:
1. Scores the full document with CE
2. Scores just the title (first line/sentence) with CE
3. Boosts documents whose titles score highly

This is particularly relevant for Wikipedia-style passages where
the title is the entity name.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


def _extract_title(text, max_chars=200):
    """Extract probable title from text (first line or sentence)."""
    # Try first line
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < max_chars:
        return first_line
    # Try first sentence
    for sep in ['. ', '! ', '? ']:
        if sep in text[:max_chars]:
            return text[:text.index(sep) + 1]
    return text[:max_chars]


class CETitleBoostHypothesis(Hypothesis):
    """Cross-encoder with title-level relevance boost."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        title_weight: float = 0.2,
    ):
        self._model_name = model_name
        self._title_weight = title_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-title-boost-{self._title_weight}"

    @property
    def description(self) -> str:
        return f"Cross-encoder + title relevance boost (weight={self._title_weight})"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()

        # Full document pairs
        full_pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        # Title-only pairs
        title_pairs = [(query, _extract_title(r.text)) for r in results]

        # Score both in one batch
        all_pairs = full_pairs + title_pairs
        all_scores = model.predict(all_pairs).tolist()

        n = len(results)
        full_scores = np.array(all_scores[:n], dtype=np.float64)
        title_scores = np.array(all_scores[n:], dtype=np.float64)

        # Normalize each to [0, 1]
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-12:
                return np.ones_like(arr)
            return (arr - mn) / (mx - mn)

        full_norm = _norm(full_scores)
        title_norm = _norm(title_scores)

        # Weighted combination
        final_scores = (1 - self._title_weight) * full_norm + self._title_weight * title_norm

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (CE + title boost):"]
        for i, idx in enumerate(ranked_indices, 1):
            title = _extract_title(results[idx].text, 60)
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, full: {full_scores[idx]:.4f}, title: {title_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "full_scores": [float(full_scores[i]) for i in ranked_indices],
                "title_scores": [float(title_scores[i]) for i in ranked_indices],
            },
        )
