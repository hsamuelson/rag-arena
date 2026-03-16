"""Sentence-level cross-encoder scoring.

Instead of scoring the entire document, split it into sentences and
score each sentence independently. The document score is the max
sentence score. This is a finer-grained version of multi-window.

Rationale: Cross-encoders are most accurate on short, focused text.
A document may contain one highly relevant sentence buried in irrelevant
context. Sentence-level scoring isolates the signal.

This is a form of "multi-granularity reranking" at the sentence level.
"""

import re
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_MIN_SENTENCE_LEN = 20  # skip very short sentences


def _split_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= _MIN_SENTENCE_LEN]


class CESentenceLevelHypothesis(Hypothesis):
    """Score documents at sentence level, use max sentence score."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_sentences_per_doc: int = 10,
        aggregation: str = "max",
    ):
        self._model_name = model_name
        self._max_sentences = max_sentences_per_doc
        self._aggregation = aggregation
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-sentence-level-{self._aggregation}"

    @property
    def description(self) -> str:
        return f"Sentence-level CE scoring ({self._aggregation} aggregation)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # Build sentence-level pairs
        all_pairs = []
        doc_sentence_map = []  # (doc_idx, n_sentences)

        for doc_idx, r in enumerate(results):
            sentences = _split_sentences(r.text)[:self._max_sentences]
            if not sentences:
                # Fallback: use whole text
                sentences = [r.text[:_MAX_CHARS]]
            for sent in sentences:
                all_pairs.append((query, sent))
            doc_sentence_map.append((doc_idx, len(sentences)))

        # Score all sentences
        all_scores = model.predict(all_pairs).tolist()

        # Aggregate per document
        doc_scores = []
        doc_n_sentences = []
        pair_idx = 0
        for doc_idx, n_sents in doc_sentence_map:
            sent_scores = all_scores[pair_idx:pair_idx + n_sents]
            pair_idx += n_sents

            if self._aggregation == "max":
                doc_scores.append(max(sent_scores))
            elif self._aggregation == "top2mean":
                sorted_s = sorted(sent_scores, reverse=True)
                doc_scores.append(sum(sorted_s[:2]) / min(2, len(sorted_s)))
            else:
                doc_scores.append(sum(sent_scores) / len(sent_scores))
            doc_n_sentences.append(n_sents)

        ranked_indices = sorted(range(n), key=lambda i: doc_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (sentence-level CE, {self._aggregation}):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {doc_scores[idx]:.4f}, sentences: {doc_n_sentences[idx]})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "doc_scores": [doc_scores[i] for i in ranked_indices],
                "n_sentences": [doc_n_sentences[i] for i in ranked_indices],
                "total_sentences_scored": len(all_pairs),
            },
        )
