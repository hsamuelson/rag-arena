"""Multi-hop iterative retrieval with cross-encoder.

For multi-hop QA (like HotpotQA), a single retrieval pass may only find
one of the two needed passages. This hypothesis:

1. First pass: Hybrid retrieval → CE reranking
2. Extract bridge entities from top-1 CE result
3. Second pass: Retrieve using bridge entity as query
4. Merge results from both passes, deduplicate
5. Final CE reranking on merged set

This is inspired by IRRR (Qi et al. 2020) and Baleen (Khattab et al. 2021)
but implemented as a post-retrieval reranking hypothesis.

Since we can't do actual second-pass retrieval (hypothesis only sees
initial results), we approximate by:
1. CE scoring first pass
2. Using top result's title/key terms to re-score remaining docs
3. Boosting docs that match bridge terms

This captures the "follow the bridge" pattern of multi-hop reasoning.
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
    "itself they them their theirs themselves and but or if while also".split()
)


def _extract_title(text, max_chars=200):
    """Extract probable title from text."""
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) < max_chars:
        return first_line
    for sep in ['. ', '! ', '? ']:
        if sep in text[:max_chars]:
            return text[:text.index(sep) + 1]
    return text[:max_chars]


def _extract_entities(text, query_terms, top_n=5):
    """Extract likely entity names (capitalized multi-word phrases)."""
    words = text.split()
    entities = []
    current_entity = []

    for w in words:
        # Check if word starts with capital (potential entity)
        clean = w.strip(".,!?;:\"'()[]{}").lower()
        if w and w[0].isupper() and clean not in _STOPWORDS and clean not in query_terms:
            current_entity.append(w.strip(".,!?;:\"'()[]{}"))
        else:
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
    if current_entity:
        entities.append(" ".join(current_entity))

    # Count and return most common
    counter = Counter(entities)
    return [e for e, _ in counter.most_common(top_n)]


class CEMultihopIterativeHypothesis(Hypothesis):
    """Multi-hop iterative CE: use bridge entities to boost second-hop docs."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bridge_weight: float = 0.15,
        n_bridge_entities: int = 3,
    ):
        self._model_name = model_name
        self._bridge_weight = bridge_weight
        self._n_bridge_entities = n_bridge_entities
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "ce-multihop-iterative"

    @property
    def description(self) -> str:
        return "Multi-hop iterative CE: detect bridge entities in top results, boost docs containing those entities"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # Pass 1: Standard CE scoring
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Get top result from first pass
        top_idx = max(range(n), key=lambda i: ce_scores[i])
        top_text = results[top_idx].text
        top_title = _extract_title(top_text)

        # Extract bridge entities from top result
        query_terms = set(w.lower().strip(".,!?") for w in query.split())
        bridge_entities = _extract_entities(top_text, query_terms, self._n_bridge_entities)

        # Also use the title of top result as a bridge entity
        if top_title and top_title.lower() not in query_terms:
            bridge_entities.insert(0, top_title)

        # Pass 2: Score remaining docs against bridge query
        # Create bridge query from entities
        bridge_query = query + " " + " ".join(bridge_entities[:3])
        pairs_2 = [(bridge_query, r.text[:_MAX_CHARS]) for r in results]
        bridge_scores = model.predict(pairs_2).tolist()

        # Also compute bridge entity overlap bonus
        entity_bonus = np.zeros(n, dtype=np.float64)
        for i, r in enumerate(results):
            text_lower = r.text.lower()
            for entity in bridge_entities:
                if entity.lower() in text_lower:
                    entity_bonus[i] += 1.0
        # Normalize bonus
        if entity_bonus.max() > 0:
            entity_bonus = entity_bonus / entity_bonus.max()

        # Fuse: primary CE + bridge CE + entity bonus
        ce_arr = np.array(ce_scores, dtype=np.float64)
        bridge_arr = np.array(bridge_scores, dtype=np.float64)

        # Normalize both to [0, 1]
        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-12)

        ce_norm = _norm(ce_arr)
        bridge_norm = _norm(bridge_arr)

        final_scores = (
            (1 - self._bridge_weight) * ce_norm
            + self._bridge_weight * 0.7 * bridge_norm
            + self._bridge_weight * 0.3 * entity_bonus
        )

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (multi-hop iterative CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, ce1: {ce_scores[idx]:.4f}, bridge: {bridge_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in ranked_indices],
                "bridge_scores": [bridge_scores[i] for i in ranked_indices],
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "bridge_entities": bridge_entities,
                "bridge_query": bridge_query,
            },
        )
