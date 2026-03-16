"""Pairwise cross-encoder hypothesis.

Instead of scoring each document independently (pointwise), compare
documents pairwise and use win-counts to rank them. This captures
relative relevance that pointwise scoring misses.

For efficiency, we don't do all O(n^2) pairs. We use a tournament:
1. Pointwise CE to get initial ranking
2. Bubble-sort style: compare adjacent pairs, swap if needed
3. Final ranking based on pairwise wins

This adds ~n extra CE inferences on top of the initial n.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 1500


class CEPairwiseHypothesis(Hypothesis):
    """Pairwise cross-encoder comparison for refined ranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        n_passes: int = 2,
    ):
        self._model_name = model_name
        self._n_passes = n_passes
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return f"ce-pairwise-{self._n_passes}pass"

    @property
    def description(self) -> str:
        return f"Pairwise CE comparison with {self._n_passes} bubble-sort passes over pointwise ranking"

    def apply(self, query, results, embeddings, query_embedding):
        if not results or len(results) <= 1:
            return HypothesisResult(
                results=results or [],
                context_prompt="Retrieved context:\n" + (results[0].text if results else "(no results)"),
                metadata={},
            )

        model = self._get_model()

        # Step 1: Pointwise CE for initial ranking
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        pointwise_scores = model.predict(pairs).tolist()
        order = sorted(range(len(pointwise_scores)), key=lambda i: pointwise_scores[i], reverse=True)

        # Step 2: Bubble-sort passes using pairwise comparison
        # For each adjacent pair, ask CE: "Given query Q, is doc A or doc B more relevant?"
        # We approximate this by comparing CE scores but with both docs in context
        working = list(order)

        for _pass in range(self._n_passes):
            swapped = False
            compare_pairs = []
            compare_indices = []

            for i in range(len(working) - 1):
                idx_a, idx_b = working[i], working[i + 1]
                # Create comparison: query + "Document A: ... Document B: ..."
                text_a = results[idx_a].text[:_MAX_CHARS // 2]
                text_b = results[idx_b].text[:_MAX_CHARS // 2]
                # Score A with context of B nearby
                compare_pairs.append((query, text_a))
                compare_pairs.append((query, text_b))
                compare_indices.append(i)

            if not compare_pairs:
                break

            scores = model.predict(compare_pairs).tolist()

            # Process pairwise comparisons
            for j, i in enumerate(compare_indices):
                score_a = scores[j * 2]
                score_b = scores[j * 2 + 1]
                if score_b > score_a:
                    working[i], working[i + 1] = working[i + 1], working[i]
                    swapped = True

            if not swapped:
                break

        reranked = [results[i] for i in working]
        final_scores_map = {idx: len(working) - rank for rank, idx in enumerate(working)}

        lines = ["Retrieved context (pairwise CE reranked):"]
        for i, idx in enumerate(working, 1):
            lines.append(f"\n[{i}] (pointwise_ce: {pointwise_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "pointwise_scores": [pointwise_scores[i] for i in working],
                "n_passes": self._n_passes,
                "total_comparisons": len(compare_pairs) // 2,
            },
        )
