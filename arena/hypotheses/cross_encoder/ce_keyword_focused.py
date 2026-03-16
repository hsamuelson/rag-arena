"""Keyword-focused cross-encoder hypothesis.

Instead of scoring the full document, extract the most relevant passage
around query keywords, then score just that passage. This focuses CE
attention on the information-dense region of the document.

Process:
1. Find sentences containing query keywords
2. Extract a window around those sentences
3. Score the keyword-focused window with CE

This is related to "passage extraction" in reading comprehension systems.
"""

import re
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_WINDOW_CHARS = 1500


def _extract_keyword_passage(text, query, window_chars=_WINDOW_CHARS):
    """Extract the passage most relevant to query keywords."""
    query_terms = set(w.lower().strip(".,?!") for w in query.split() if len(w) > 2)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:window_chars]

    # Score each sentence by keyword overlap
    scored_sentences = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        matches = sum(1 for t in query_terms if t in sent_lower)
        scored_sentences.append((i, matches, sent))

    # Find the sentence with most keyword matches
    best_idx = max(range(len(scored_sentences)), key=lambda i: scored_sentences[i][1])

    # Extract a window around that sentence
    # Include sentences before and after
    start_idx = max(0, best_idx - 2)
    end_idx = min(len(sentences), best_idx + 3)

    passage = " ".join(sentences[start_idx:end_idx])

    # If passage is too short, expand
    while len(passage) < window_chars // 2 and (start_idx > 0 or end_idx < len(sentences)):
        if start_idx > 0:
            start_idx -= 1
        if end_idx < len(sentences):
            end_idx += 1
        passage = " ".join(sentences[start_idx:end_idx])

    return passage[:window_chars]


class CEKeywordFocusedHypothesis(Hypothesis):
    """CE scoring on keyword-focused passage extraction."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        combine_with_full: bool = True,
        focus_weight: float = 0.5,
    ):
        self._model_name = model_name
        self._combine = combine_with_full
        self._focus_weight = focus_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        if self._combine:
            return f"ce-keyword-focused-{self._focus_weight}"
        return "ce-keyword-focused-only"

    @property
    def description(self) -> str:
        return "CE scoring on keyword-focused passage extraction"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        if self._combine:
            # Score both full doc and focused passage
            all_pairs = []
            for r in results:
                all_pairs.append((query, r.text[:_MAX_CHARS]))  # full
                passage = _extract_keyword_passage(r.text, query)
                all_pairs.append((query, passage))  # focused

            all_scores = model.predict(all_pairs).tolist()

            full_scores = [all_scores[i * 2] for i in range(n)]
            focus_scores = [all_scores[i * 2 + 1] for i in range(n)]

            # Normalize and combine
            full_arr = np.array(full_scores, dtype=np.float64)
            focus_arr = np.array(focus_scores, dtype=np.float64)

            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-12:
                    return np.ones_like(arr)
                return (arr - mn) / (mx - mn)

            final_scores = (1 - self._focus_weight) * _norm(full_arr) + self._focus_weight * _norm(focus_arr)
        else:
            # Score only focused passages
            pairs = [(query, _extract_keyword_passage(r.text, query)) for r in results]
            scores = model.predict(pairs).tolist()
            final_scores = np.array(scores, dtype=np.float64)

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (keyword-focused CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
            },
        )
