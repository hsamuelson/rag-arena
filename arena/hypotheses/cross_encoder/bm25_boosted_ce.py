"""BM25-boosted cross-encoder hypothesis.

The key insight from our gold standard: BM25 alone gets 0.679 nDCG while
dense gets 0.585. BM25 captures term-level relevance that dense misses.

This hypothesis computes a lightweight BM25-style term overlap score
between query and document, then uses it to boost CE scores for documents
with strong lexical matches.

Unlike hybrid RRF (which fuses at retrieval), this fuses at reranking time.
"""

import re
import math
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
    "who whom this that these those i me my myself we our".split()
)


def _tokenize(text):
    """Simple tokenizer."""
    return [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split()
            if len(w.strip(".,!?;:\"'()[]{}")) > 1]


def _compute_bm25_score(query_tokens, doc_tokens, avg_dl, k1=1.5, b=0.75):
    """Compute BM25 score for a single document."""
    doc_len = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        if qt in _STOPWORDS:
            continue
        tf = doc_tf.get(qt, 0)
        if tf == 0:
            continue
        # Simplified IDF (assume rare term)
        idf = 1.0  # We don't have corpus stats, so use uniform
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
        score += idf * numerator / denominator
    return score


class BM25BoostedCEHypothesis(Hypothesis):
    """Cross-encoder with BM25 term overlap boost."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_weight: float = 0.15,
    ):
        self._model_name = model_name
        self._bm25_weight = bm25_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"bm25-boosted-ce-{self._bm25_weight}"

    @property
    def description(self) -> str:
        return f"Cross-encoder + BM25 term overlap boost (weight={self._bm25_weight})"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # Compute BM25-style scores
        query_tokens = _tokenize(query)
        all_doc_tokens = [_tokenize(r.text) for r in results]
        avg_dl = sum(len(dt) for dt in all_doc_tokens) / max(len(all_doc_tokens), 1)

        bm25_scores = np.array([
            _compute_bm25_score(query_tokens, dt, avg_dl)
            for dt in all_doc_tokens
        ], dtype=np.float64)

        # Normalize both to [0, 1]
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-12:
                return np.ones_like(arr)
            return (arr - mn) / (mx - mn)

        ce_norm = _norm(ce_scores)
        bm25_norm = _norm(bm25_scores)

        final_scores = (1 - self._bm25_weight) * ce_norm + self._bm25_weight * bm25_norm

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (BM25-boosted CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, ce: {ce_scores[idx]:.4f}, bm25: {bm25_scores[idx]:.3f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
                "bm25_scores": [float(bm25_scores[i]) for i in ranked_indices],
            },
        )
