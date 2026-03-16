"""Query-type adaptive cross-encoder hypothesis.

Different query types need different scoring strategies:
- Bridge questions ("Who directed the film that won X?"): need entity linking
- Comparison questions ("Which is older, A or B?"): need both entities
- Factoid questions ("When was X born?"): single fact lookup

This hypothesis classifies the query type, then adapts scoring:
- Bridge: boost docs with CE + entity overlap
- Comparison: ensure docs about BOTH compared entities are ranked high
- Factoid: standard CE (works well already)
"""

import re
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


def _classify_query(query):
    """Classify query type based on patterns."""
    q_lower = query.lower()

    # Comparison patterns
    if any(w in q_lower for w in ["which is", "who is more", "who is older",
                                   "which was", "who was more", "between"]):
        return "comparison"

    # Bridge patterns (entity linking)
    if re.search(r"(?:of|by|in|at|from)\s+(?:the\s+)?(?:[A-Z]|\w+\s+(?:that|which|who))", query):
        return "bridge"

    # Multi-hop indicators
    if re.search(r"(?:the\s+\w+\s+(?:who|that|which)\s+\w+)", query, re.IGNORECASE):
        return "bridge"

    return "factoid"


def _extract_comparison_entities(query):
    """Extract entities being compared."""
    # Patterns like "A and B" or "A or B"
    match = re.search(r'(?:between|comparing)\s+(.+?)\s+and\s+(.+?)(?:\?|,|$)', query, re.IGNORECASE)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Try "Which/Who ... A or B"
    match = re.search(r'(?:Which|Who)\s+.+?,\s*(.+?)\s+or\s+(.+?)(?:\?|$)', query, re.IGNORECASE)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return []


class CEQueryTypeAdaptiveHypothesis(Hypothesis):
    """Adapt CE scoring based on query type classification."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "ce-query-type-adaptive"

    @property
    def description(self) -> str:
        return "Adaptive CE scoring based on query type (bridge/comparison/factoid)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)
        query_type = _classify_query(query)

        # Standard CE scoring
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        if query_type == "comparison":
            # For comparison: ensure diversity — docs about DIFFERENT entities
            entities = _extract_comparison_entities(query)
            if len(entities) >= 2:
                entity_coverage = np.zeros(n, dtype=np.float64)
                for i, r in enumerate(results):
                    text_lower = r.text.lower()
                    covers = sum(1 for e in entities if e.lower() in text_lower)
                    entity_coverage[i] = covers / len(entities)

                # Bonus for covering entities not yet covered
                ce_min, ce_max = ce_scores.min(), ce_scores.max()
                if ce_max - ce_min > 1e-12:
                    ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
                else:
                    ce_norm = np.ones_like(ce_scores)

                final_scores = 0.8 * ce_norm + 0.2 * entity_coverage
            else:
                final_scores = ce_scores

        elif query_type == "bridge":
            # For bridge: boost docs that link to other highly-scored docs
            if embeddings is not None and len(embeddings) == n:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                normed = embeddings / norms
                sim_matrix = normed @ normed.T

                # Top-1 CE doc
                top_idx = np.argmax(ce_scores)
                # Similarity to top doc
                sim_to_top = sim_matrix[top_idx]

                # Boost docs that are DISSIMILAR to top (bridge = 2 different articles)
                dissimilarity = 1.0 - sim_to_top
                dis_min, dis_max = dissimilarity.min(), dissimilarity.max()
                if dis_max - dis_min > 1e-12:
                    dis_norm = (dissimilarity - dis_min) / (dis_max - dis_min)
                else:
                    dis_norm = np.zeros_like(dissimilarity)

                ce_min, ce_max = ce_scores.min(), ce_scores.max()
                if ce_max - ce_min > 1e-12:
                    ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
                else:
                    ce_norm = np.ones_like(ce_scores)

                # Don't penalize the top doc itself
                dis_norm[top_idx] = 1.0

                final_scores = 0.85 * ce_norm + 0.15 * dis_norm
            else:
                final_scores = ce_scores
        else:
            # Factoid: pure CE
            final_scores = ce_scores

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (adaptive CE, type={query_type}):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {float(final_scores[idx]):.4f}, ce: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "query_type": query_type,
                "ce_scores": [float(ce_scores[i]) for i in ranked_indices],
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
            },
        )
