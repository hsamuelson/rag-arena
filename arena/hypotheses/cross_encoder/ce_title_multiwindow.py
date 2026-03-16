"""Combined title boost + multi-window cross-encoder.

Fuses the two best techniques from Round 3:
1. Title boost: separate CE scoring of title vs body
2. Multi-window: score multiple text windows, take max

Combined approach:
- Score title with CE
- Score first-half of body with CE
- Score second-half of body with CE
- Final = max(first_half, second_half) * (1-title_weight) + title_score * title_weight
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


def _extract_title(text, max_chars=200):
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < max_chars:
        return first_line
    for sep in ['. ', '! ', '? ']:
        if sep in text[:max_chars]:
            return text[:text.index(sep) + 1]
    return text[:max_chars]


class CETitleMultiWindowHypothesis(Hypothesis):
    """Combined title boost + multi-window CE scoring."""

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
        return f"ce-title-multiwindow-{self._title_weight}"

    @property
    def description(self) -> str:
        return f"Combined title boost ({self._title_weight}) + multi-window max CE scoring"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # Build all pairs: title, first-half, second-half for each doc
        all_pairs = []
        for r in results:
            text = r.text[:_MAX_CHARS * 2]
            title = _extract_title(r.text)
            mid = len(text) // 2

            all_pairs.append((query, title))                    # title
            all_pairs.append((query, text[:_MAX_CHARS]))        # first half
            all_pairs.append((query, text[mid:][:_MAX_CHARS]))  # second half

        # Score all in one batch
        all_scores = model.predict(all_pairs).tolist()

        # Parse scores
        title_scores = []
        body_scores = []
        for i in range(n):
            base = i * 3
            title_scores.append(all_scores[base])
            body_scores.append(max(all_scores[base + 1], all_scores[base + 2]))

        # Normalize
        def _norm(arr):
            arr = np.array(arr, dtype=np.float64)
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-12:
                return np.ones_like(arr)
            return (arr - mn) / (mx - mn)

        title_norm = _norm(title_scores)
        body_norm = _norm(body_scores)

        final_scores = (1 - self._title_weight) * body_norm + self._title_weight * title_norm

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (title + multi-window CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, body: {body_scores[idx]:.4f}, title: {title_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "body_scores": [body_scores[i] for i in ranked_indices],
                "title_scores": [title_scores[i] for i in ranked_indices],
            },
        )
