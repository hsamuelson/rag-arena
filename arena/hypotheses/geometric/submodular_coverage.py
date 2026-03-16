"""Submodular coverage reranking (Novel #11).

Model document selection as a submodular function maximisation problem.
The utility function measures *coverage* — how much of the query's
"information need" is satisfied by the selected subset.

Geometric intuition
-------------------
Think of each document as a "facility" that can serve nearby information
needs.  The facility location objective scores a subset S by:

    f(S) = sum_j max_{i in S} sim(i, j)

where j ranges over all candidate documents (or query-related target
points).  Each target point is "served" by its closest selected document.
This function is monotone submodular, so the greedy algorithm (pick the
document that maximises marginal gain at each step) gives a (1 - 1/e)
approximation to the optimum.

Unlike MMR or DPP, this objective directly models *coverage* — ensuring
every corner of the information space has a nearby representative — rather
than diversity per se.

References:
  - Nemhauser, Wolsey & Fisher (1978): submodular maximisation guarantees
  - Lin & Bilmes (2011): submodular document summarisation
  - Wei, Iyer & Bilmes (2015): submodularity in ML
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class SubmodularCoverageHypothesis(Hypothesis):
    """Greedy submodular facility-location selection for coverage."""

    def __init__(self, query_weight: float = 0.3):
        """
        Args:
            query_weight: Weight given to covering the query direction
                vs covering the full document set.  Higher values bias
                selection toward query-relevant documents.
        """
        self.query_weight = query_weight

    @property
    def name(self) -> str:
        return f"submodular-coverage-{self.query_weight}q"

    @property
    def description(self) -> str:
        return (
            "Submodular facility-location coverage — greedy selection "
            "maximising coverage of the information space with (1-1/e) guarantee"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or len(results) < 4:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        n = len(results)

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        # Similarity matrix between documents (clipped to non-negative)
        S_doc = E @ E.T
        S_doc = np.maximum(S_doc, 0.0)

        # Query similarity column (if available)
        if query_embedding is not None:
            qnorm = np.linalg.norm(query_embedding)
            if qnorm > 1e-12:
                q = query_embedding / qnorm
                q_sim = np.maximum(E @ q, 0.0)  # (n,)
            else:
                q_sim = np.zeros(n)
        else:
            q_sim = np.zeros(n)

        # Build the target similarity matrix.
        # Columns = "demand points" = all docs + a virtual query point.
        # For efficiency, we treat each document as a demand point and add
        # the query as an extra demand point with weight query_weight.
        #
        # Facility location objective:
        #   f(S) = (1-w) * sum_j max_{i in S} S_doc[i,j]
        #          + w * max_{i in S} q_sim[i]

        w = self.query_weight

        # Greedy selection
        selected: list[int] = []
        remaining = set(range(n))
        marginal_gains: list[float] = []

        # Track current coverage: for each demand point j,
        # max_coverage[j] = max_{i in S} S_doc[i, j]
        max_coverage = np.full(n, -np.inf)
        max_q_coverage = -np.inf

        for _ in range(n):
            best_idx = -1
            best_gain = -float("inf")

            for idx in remaining:
                # Marginal gain from adding idx
                doc_gain = float(np.sum(
                    np.maximum(S_doc[idx] - max_coverage, 0.0)
                ))
                q_gain = float(max(q_sim[idx] - max_q_coverage, 0.0))

                total_gain = (1.0 - w) * doc_gain + w * q_gain

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_idx = idx

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            marginal_gains.append(best_gain)

            # Update coverage
            max_coverage = np.maximum(max_coverage, S_doc[best_idx])
            max_q_coverage = max(max_q_coverage, q_sim[best_idx])

        reranked = [results[i] for i in selected]

        # Final coverage value
        total_coverage = float(
            (1.0 - w) * np.sum(max_coverage) + w * max_q_coverage
        )

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": selected,
                "marginal_gains": marginal_gains,
                "total_coverage": total_coverage,
                "coverage_per_step": [
                    sum(marginal_gains[:i + 1]) for i in range(len(marginal_gains))
                ],
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (submodular-coverage selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
