"""Cross-encoder with answer extraction signal.

Instead of just asking "is this document relevant?", also check if the
document contains a plausible answer span. Documents with extractable
answers are more likely to be genuinely relevant.

Process:
1. Standard CE scoring for relevance
2. For top candidates, check if query entities appear in the document
3. Check for answer-type patterns (dates, names, numbers) near query terms
4. Boost documents that seem to contain actual answers

This is inspired by reading comprehension models (SQuAD-style) but
implemented as a lightweight heuristic without a separate model.
"""

import re
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000

# Answer type patterns
_PATTERNS = {
    "date": r'\b\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?',
    "number": r'\b\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|percent|%|km|miles|meters|feet))',
    "person": r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
    "place": r'(?:in|at|from|near)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
}


def _answer_density_score(text, query):
    """Score how likely this text contains an answer to the query."""
    text_lower = text.lower()
    query_lower = query.lower()

    # 1. Query term density near potential answers
    query_terms = set(w.strip(".,?!") for w in query_lower.split() if len(w) > 2)

    # 2. Count answer-type patterns
    pattern_matches = 0
    for pattern_name, pattern in _PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        pattern_matches += len(matches)

    # 3. Check if query entities are present AND there are nearby answer patterns
    entity_present = sum(1 for t in query_terms if t in text_lower)

    # 4. Sentence-level: find sentences with both query terms AND answer patterns
    sentences = re.split(r'(?<=[.!?])\s+', text)
    answer_sentences = 0
    for sent in sentences:
        sent_lower = sent.lower()
        has_query = any(t in sent_lower for t in query_terms)
        has_answer = any(re.search(p, sent, re.IGNORECASE) for p in _PATTERNS.values())
        if has_query and has_answer:
            answer_sentences += 1

    # Combine signals
    score = (
        0.3 * min(entity_present / max(len(query_terms), 1), 1.0)
        + 0.3 * min(pattern_matches / 5.0, 1.0)
        + 0.4 * min(answer_sentences / 2.0, 1.0)
    )
    return score


class CEAnswerExtractionHypothesis(Hypothesis):
    """CE + answer extraction signal: boost docs likely containing answers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        answer_weight: float = 0.15,
    ):
        self._model_name = model_name
        self._answer_weight = answer_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-answer-extraction-{self._answer_weight}"

    @property
    def description(self) -> str:
        return "CE + answer extraction heuristic: boost documents containing plausible answers"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # Compute answer density for each document
        answer_scores = np.array([_answer_density_score(r.text, query) for r in results])

        # Normalize CE scores
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max - ce_min > 1e-12:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_scores)

        # Combined score
        final_scores = (1 - self._answer_weight) * ce_norm + self._answer_weight * answer_scores

        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (CE + answer extraction):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, ce: {ce_scores[idx]:.4f}, ans: {answer_scores[idx]:.3f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
                "answer_scores": [float(answer_scores[i]) for i in ranked_indices],
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
            },
        )
