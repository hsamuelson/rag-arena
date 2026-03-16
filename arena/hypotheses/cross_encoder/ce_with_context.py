"""Cross-encoder with contextual document enhancement.

Before scoring, prepend document metadata (title, source) to the document text.
For HotpotQA, the document title IS the Wikipedia article title — prepending
"Article: {title}\n{text}" gives the CE model explicit entity context.

This is inspired by Anthropic's "Contextual Retrieval" (Sep 2025) which
showed that adding context to chunks before embedding reduces failure rate by 49%.
We apply the same principle at reranking time.
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


class CEWithContextHypothesis(Hypothesis):
    """CE with explicit document context (title/source) prepended."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        context_template: str = "Article: {title}\n\n{text}",
    ):
        self._model_name = model_name
        self._context_template = context_template
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "ce-with-context"

    @property
    def description(self) -> str:
        return "Cross-encoder with document context (title) prepended to text"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()

        # Build pairs with context-enhanced documents
        pairs = []
        for r in results:
            title = _extract_title(r.text)
            enhanced = self._context_template.format(title=title, text=r.text)
            pairs.append((query, enhanced[:_MAX_CHARS]))

        ce_scores = model.predict(pairs).tolist()

        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (CE with document context):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"ce_scores": [ce_scores[i] for i in ranked_indices]},
        )
