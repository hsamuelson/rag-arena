"""Query decomposition + cross-encoder reranking.

For multi-hop QA, the original query conflates multiple information needs.
This hypothesis decomposes the query into sub-questions by heuristic parsing,
then scores each document against ALL sub-questions and takes the max.

A document might score poorly against "What year was the director of X born?"
but score highly against the sub-question "Who directed X?" — which IS the
information we need from that document.

This is inspired by DecompRC (Min et al. 2019) and Self-Ask (Press et al. 2022).
"""

import re
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000

# Multi-hop question patterns
_BRIDGE_PATTERNS = [
    # "What is the X of the Y that Z?"
    r"(?:What|Who|Where|When|Which|How).*?(?:of|by|in|at|from)\s+(?:the\s+)?(.+?)\s+(?:that|which|who|where|whose)",
    # "The X who/that Y, what is Z?"
    r"[Tt]he\s+(.+?)\s+(?:who|that|which)\s+(.+?),\s+(?:what|who|where|when|how)",
]

# Comparison patterns
_COMPARISON_PATTERNS = [
    r"(?:Which|Who)\s+(?:is|was|were|are)\s+(?:more|less|older|younger|taller|bigger|faster|earlier|later)",
    r"(?:between|comparing)\s+(.+?)\s+and\s+(.+)",
]


def _decompose_query(query):
    """Decompose a multi-hop query into sub-questions."""
    sub_questions = [query]  # always include original

    # Try to extract entities and create sub-questions
    # Pattern: "What nationality was the director of [Film]?"
    # → sub1: "Who directed [Film]?" sub2: "What nationality was [person]?"

    # Extract quoted or capitalized entity phrases
    entities = re.findall(r'"([^"]+)"', query)
    # Also try to find proper nouns (consecutive capitalized words)
    words = query.split()
    current_entity = []
    for w in words[1:]:  # skip first word (often capitalized question word)
        clean = w.strip(".,?!;:")
        if clean and clean[0].isupper() and clean.lower() not in {
            "what", "who", "where", "when", "which", "how", "is", "was",
            "the", "a", "an", "and", "or", "but", "in", "of", "for",
            "to", "at", "by", "from", "with", "on", "that", "this"
        }:
            current_entity.append(clean)
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))

    # Create sub-questions about each entity
    for entity in entities[:3]:  # limit to 3 entities
        sub_questions.append(f"What is {entity}?")
        sub_questions.append(f"Tell me about {entity}")

    # For comparison questions, create per-entity questions
    for pattern in _COMPARISON_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            for g in m.groups():
                if g:
                    sub_questions.append(f"What is {g.strip()}?")

    return list(dict.fromkeys(sub_questions))[:7]  # deduplicate, max 7


class QueryDecompCEHypothesis(Hypothesis):
    """Decompose multi-hop query into sub-questions, score docs against each."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        aggregation: str = "max",  # "max" or "mean"
    ):
        self._model_name = model_name
        self._aggregation = aggregation
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"query-decomp-ce-{self._aggregation}"

    @property
    def description(self) -> str:
        return f"Decompose query into sub-questions, CE score against each ({self._aggregation} aggregation)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        sub_questions = _decompose_query(query)
        n = len(results)

        # Score each doc against each sub-question
        all_pairs = []
        for sq in sub_questions:
            for r in results:
                all_pairs.append((sq, r.text[:_MAX_CHARS]))

        all_scores = model.predict(all_pairs).tolist()

        # Reshape: (n_subquestions, n_docs)
        scores_matrix = []
        for i in range(len(sub_questions)):
            start = i * n
            scores_matrix.append(all_scores[start:start + n])

        scores_matrix = np.array(scores_matrix)

        # Aggregate across sub-questions
        if self._aggregation == "max":
            final_scores = scores_matrix.max(axis=0)
        else:
            final_scores = scores_matrix.mean(axis=0)

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (query decomp CE, {len(sub_questions)} sub-Qs):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "sub_questions": sub_questions,
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "aggregation": self._aggregation,
                "n_subquestions": len(sub_questions),
            },
        )
