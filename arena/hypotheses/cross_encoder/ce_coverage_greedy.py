"""Coverage-greedy CE hypothesis.

For multi-hop QA, we need coverage of DIFFERENT facts/entities, not
just the most relevant single document. This combines CE relevance
with a coverage-greedy selection: each subsequent document must add
new information (measured by unique query-term coverage).

Algorithm:
1. Score all docs with CE
2. Greedily select: each pick maximizes CE_score + coverage_bonus
   where coverage_bonus rewards docs covering query terms not yet
   covered by already-selected docs

This is MMR-like but uses query-term coverage instead of embedding distance.
"""

import numpy as np

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


class CECoverageGreedyHypothesis(Hypothesis):
    """CE + coverage-greedy selection for multi-hop query coverage."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        coverage_weight: float = 0.2,
    ):
        self._model_name = model_name
        self._coverage_weight = coverage_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-coverage-greedy-{self._coverage_weight}"

    @property
    def description(self) -> str:
        return f"CE + coverage-greedy selection (coverage_weight={self._coverage_weight})"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # CE scoring
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # Normalize CE to [0, 1]
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max - ce_min > 1e-12:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.ones_like(ce_scores)

        # Extract query terms (non-stopword)
        query_terms = set(
            w.lower().strip(".,?!;:\"'()") for w in query.split()
            if w.lower().strip(".,?!;:\"'()") not in _STOPWORDS and len(w) > 2
        )

        # Per-document term coverage
        doc_terms = []
        for r in results:
            text_lower = r.text.lower()
            covered = set(t for t in query_terms if t in text_lower)
            doc_terms.append(covered)

        # Greedy selection
        selected = []
        remaining = set(range(n))
        covered_terms = set()

        for _ in range(n):
            best_idx = None
            best_score = -float('inf')

            for idx in remaining:
                # New terms this doc would cover
                new_terms = doc_terms[idx] - covered_terms
                if query_terms:
                    coverage_bonus = len(new_terms) / len(query_terms)
                else:
                    coverage_bonus = 0.0

                score = (1 - self._coverage_weight) * ce_norm[idx] + self._coverage_weight * coverage_bonus

                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)
            covered_terms.update(doc_terms[best_idx])

        reranked = [results[i] for i in selected]

        lines = ["Retrieved context (coverage-greedy CE):"]
        for i, idx in enumerate(selected, 1):
            lines.append(f"\n[{i}] (ce: {ce_scores[idx]:.4f}, new_terms: {len(doc_terms[idx] - (covered_terms if i > 1 else set()))})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in selected],
                "selection_order": selected,
                "total_query_terms": len(query_terms),
                "final_coverage": len(covered_terms),
            },
        )
