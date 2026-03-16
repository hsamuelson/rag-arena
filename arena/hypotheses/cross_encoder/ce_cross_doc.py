"""Cross-document cross-encoder hypothesis.

For multi-hop QA, the answer requires combining info from 2+ documents.
Standard CE scores each doc independently. This hypothesis scores
PAIRS of documents together, asking: "Does this pair of documents
jointly answer the query?"

Process:
1. Standard CE for initial ranking
2. For top-5, create pairs (doc_i, doc_j) and score with CE
3. Boost documents that appear in high-scoring pairs

This directly models the multi-hop reasoning pattern.
"""

import numpy as np
from itertools import combinations

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 800  # shorter per doc since we concatenate pairs


class CECrossDocHypothesis(Hypothesis):
    """Cross-document CE: score document pairs for multi-hop relevance."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k_pairs: int = 5,
        pair_weight: float = 0.3,
    ):
        self._model_name = model_name
        self._top_k_pairs = top_k_pairs
        self._pair_weight = pair_weight
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-cross-doc-{self._top_k_pairs}k"

    @property
    def description(self) -> str:
        return f"Cross-document CE: score top-{self._top_k_pairs} document pairs for multi-hop"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()
        n = len(results)

        # Step 1: Individual CE scores
        individual_pairs = [(query, r.text[:_MAX_CHARS * 2]) for r in results]
        individual_scores = model.predict(individual_pairs).tolist()

        # Step 2: Pair scoring for top-k documents
        top_indices = sorted(range(n), key=lambda i: individual_scores[i], reverse=True)[:self._top_k_pairs]

        pair_scores = {}
        if len(top_indices) >= 2:
            pair_pairs = []
            pair_indices = []
            for i, j in combinations(top_indices, 2):
                # Concatenate both docs
                combined = f"{results[i].text[:_MAX_CHARS]}\n\n{results[j].text[:_MAX_CHARS]}"
                pair_pairs.append((query, combined))
                pair_indices.append((i, j))

            if pair_pairs:
                pair_ce_scores = model.predict(pair_pairs).tolist()
                for (i, j), score in zip(pair_indices, pair_ce_scores):
                    pair_scores[(i, j)] = score

        # Step 3: Compute pair-based document boost
        doc_pair_boost = np.zeros(n, dtype=np.float64)
        if pair_scores:
            max_pair = max(pair_scores.values())
            min_pair = min(pair_scores.values())
            range_pair = max_pair - min_pair if max_pair > min_pair else 1.0

            for (i, j), score in pair_scores.items():
                normalized = (score - min_pair) / range_pair
                doc_pair_boost[i] = max(doc_pair_boost[i], normalized)
                doc_pair_boost[j] = max(doc_pair_boost[j], normalized)

        # Step 4: Combine individual + pair scores
        ind_arr = np.array(individual_scores, dtype=np.float64)
        ind_min, ind_max = ind_arr.min(), ind_arr.max()
        if ind_max - ind_min > 1e-12:
            ind_norm = (ind_arr - ind_min) / (ind_max - ind_min)
        else:
            ind_norm = np.ones_like(ind_arr)

        final_scores = (1 - self._pair_weight) * ind_norm + self._pair_weight * doc_pair_boost

        ranked_indices = sorted(range(n), key=lambda i: final_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = ["Retrieved context (cross-document CE):"]
        for i, idx in enumerate(ranked_indices, 1):
            boost = doc_pair_boost[idx]
            lines.append(f"\n[{i}] (score: {final_scores[idx]:.4f}, individual: {individual_scores[idx]:.4f}, pair_boost: {boost:.3f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "individual_scores": [individual_scores[i] for i in ranked_indices],
                "pair_boosts": [float(doc_pair_boost[i]) for i in ranked_indices],
                "final_scores": [float(final_scores[i]) for i in ranked_indices],
                "n_pairs_scored": len(pair_scores),
            },
        )
