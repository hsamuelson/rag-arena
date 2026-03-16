"""Iterative cross-encoder with pseudo-relevance feedback.

Two-pass approach:
1. First pass: Cross-encoder reranks candidates
2. Extract key terms from top-3 CE results
3. Boost documents that contain those key terms
4. Second pass: Re-score with CE using expanded query

This implements pseudo-relevance feedback at the reranking stage.
"""

import numpy as np
from collections import Counter

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_STOPWORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all both each few more "
    "most other some such no nor not only own same so than too very what which "
    "who whom this that these those i me my myself we our ours ourselves you your "
    "yours yourself yourselves he him his himself she her hers herself it its "
    "itself they them their theirs themselves and but or if while".split()
)


def _extract_key_terms(texts, query_terms, top_n=5):
    """Extract most frequent non-stopword, non-query terms from texts."""
    word_counts = Counter()
    query_lower = set(t.lower() for t in query_terms)
    for text in texts:
        words = text.lower().split()
        for w in words:
            w_clean = w.strip(".,!?;:\"'()[]{}").lower()
            if len(w_clean) > 2 and w_clean not in _STOPWORDS and w_clean not in query_lower:
                word_counts[w_clean] += 1
    return [w for w, _ in word_counts.most_common(top_n)]


class CEIterativeHypothesis(Hypothesis):
    """Two-pass cross-encoder with pseudo-relevance feedback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        prf_top_k: int = 3,
        expansion_terms: int = 5,
        expansion_weight: float = 0.3,
    ):
        self._model_name = model_name
        self._prf_top_k = prf_top_k
        self._expansion_terms = expansion_terms
        self._expansion_weight = expansion_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "ce-iterative-prf"

    @property
    def description(self) -> str:
        return "Two-pass cross-encoder: first rerank, extract key terms from top results, re-score with expanded query"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()

        # Pass 1: Standard CE scoring
        pairs_1 = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores_1 = model.predict(pairs_1).tolist()

        # Get top-k from first pass
        top_indices = sorted(range(len(ce_scores_1)), key=lambda i: ce_scores_1[i], reverse=True)[:self._prf_top_k]
        top_texts = [results[i].text for i in top_indices]

        # Extract expansion terms
        query_terms = query.split()
        expansion = _extract_key_terms(top_texts, query_terms, self._expansion_terms)
        expanded_query = query + " " + " ".join(expansion)

        # Pass 2: CE with expanded query
        pairs_2 = [(expanded_query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores_2 = model.predict(pairs_2).tolist()

        # Fuse pass 1 and pass 2 scores
        final_scores = [
            (1 - self._expansion_weight) * s1 + self._expansion_weight * s2
            for s1, s2 in zip(ce_scores_1, ce_scores_2)
        ]

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (iterative CE with PRF):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "pass1_scores": [ce_scores_1[i] for i in ranked_indices],
                "pass2_scores": [ce_scores_2[i] for i in ranked_indices],
                "expansion_terms": expansion,
                "expanded_query": expanded_query,
            },
        )
