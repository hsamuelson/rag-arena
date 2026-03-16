"""Multi-window cross-encoder hypothesis.

Cross-encoders process a fixed window of text. Different truncation
points may capture different relevance signals. This hypothesis
scores each document at multiple truncation windows and fuses
the scores.

Windows:
1. First 512 tokens (~2000 chars) — captures opening/title relevance
2. First 256 tokens (~1000 chars) — tighter focus on lead paragraph
3. Last 256 tokens (~1000 chars) — captures conclusion/summary

Final score = max(window_scores) or weighted combination.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class CEMultiWindowHypothesis(Hypothesis):
    """Cross-encoder scoring at multiple text windows, fused."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        aggregation: str = "max",
    ):
        self._model_name = model_name
        self._aggregation = aggregation  # "max" or "weighted"
        self._model = None
        # (start_frac, end_frac, weight)
        self._windows = [
            (0.0, 1.0, 0.5),    # full text (truncated at 2000)
            (0.0, 0.5, 0.3),    # first half
            (0.5, 1.0, 0.2),    # second half
        ]

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-multi-window-{self._aggregation}"

    @property
    def description(self) -> str:
        return f"Cross-encoder at multiple text windows ({self._aggregation} fusion)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        max_chars = 2000

        # Build pairs for all windows
        all_pairs = []
        for r in results:
            text = r.text[:max_chars * 2]  # take more text for windowing
            text_len = len(text)
            for start_frac, end_frac, _ in self._windows:
                start = int(text_len * start_frac)
                end = int(text_len * end_frac)
                window_text = text[start:end][:max_chars]
                if not window_text.strip():
                    window_text = text[:max_chars]
                all_pairs.append((query, window_text))

        # Score all windows in one batch
        all_scores = model.predict(all_pairs).tolist()

        # Aggregate per document
        n_windows = len(self._windows)
        doc_scores = []
        window_details = []

        for doc_idx in range(len(results)):
            start = doc_idx * n_windows
            w_scores = all_scores[start:start + n_windows]
            window_details.append(w_scores)

            if self._aggregation == "max":
                doc_scores.append(max(w_scores))
            else:  # weighted
                weighted = sum(s * w for s, (_, _, w) in zip(w_scores, self._windows))
                doc_scores.append(weighted)

        ranked_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (multi-window CE, {self._aggregation}):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {doc_scores[idx]:.4f}, windows: {window_details[idx]})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "doc_scores": [doc_scores[i] for i in ranked_indices],
                "window_scores": [window_details[i] for i in ranked_indices],
                "aggregation": self._aggregation,
            },
        )
