"""Influence function reranking via leave-one-out perturbation.

Hypothesis (Novel #15): Inspired by influence functions in machine learning,
which estimate the effect of removing a training point on model predictions.
Here we estimate each document's "influence" on the retrieved context by
computing how much removing it shifts the centroid of the retrieved set
relative to the query.

Geometric intuition: Consider the centroid of all retrieved documents as a
summary of the context the LLM receives. Some documents are inert — removing
them barely changes the centroid's position relative to the query. Others are
highly influential — their removal shifts the centroid noticeably, either
toward or away from the query. Documents whose removal shifts the centroid
*away* from the query are load-bearing: they are pulling the context toward
the query and should be ranked higher.

Algorithm:
1. Compute the full centroid of all retrieved embeddings.
2. For each document i, compute the leave-one-out centroid (without doc i).
3. Measure how the query-centroid distance changes: if it increases when
   doc i is removed, doc i has positive influence (it helps).
4. Rank by influence: documents with highest positive influence first.
5. Combine influence with original score for final ranking.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class InfluenceFunctionHypothesis(Hypothesis):
    """Rerank by leave-one-out influence on query-centroid alignment."""

    def __init__(self, influence_weight: float = 0.6):
        self.influence_weight = influence_weight

    @property
    def name(self) -> str:
        return "influence-function"

    @property
    def description(self) -> str:
        return (
            "Influence function reranking — estimate each document's impact "
            "on query-centroid alignment via leave-one-out perturbation"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 3:
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

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        q = query_embedding / q_norm

        # Full centroid and its distance to query
        full_centroid = E.mean(axis=0)
        full_dist = float(np.linalg.norm(full_centroid - q))

        # Leave-one-out influence for each document
        # LOO centroid_i = (sum - E[i]) / (n-1)
        # We can compute this efficiently:
        #   centroid_without_i = (n * full_centroid - E[i]) / (n - 1)
        influences = np.zeros(n)
        loo_dists = np.zeros(n)

        for i in range(n):
            loo_centroid = (n * full_centroid - E[i]) / (n - 1)
            loo_dist = float(np.linalg.norm(loo_centroid - q))
            loo_dists[i] = loo_dist
            # Positive influence = removing doc increases distance (doc was helping)
            influences[i] = loo_dist - full_dist

        # Normalise influence scores to [0, 1] range
        inf_min, inf_max = influences.min(), influences.max()
        if inf_max - inf_min > 1e-12:
            norm_influences = (influences - inf_min) / (inf_max - inf_min)
        else:
            norm_influences = np.ones(n) * 0.5

        # Normalise original scores similarly
        raw_scores = np.array([r.score for r in results])
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max - s_min > 1e-12:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(n) * 0.5

        # Combined score: weighted blend of influence and original relevance
        w = self.influence_weight
        combined = w * norm_influences + (1 - w) * norm_scores

        order = np.argsort(combined)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "full_centroid_distance": full_dist,
                "influences": influences[order].tolist(),
                "loo_distances": loo_dists[order].tolist(),
                "combined_scores": combined[order].tolist(),
                "most_influential_doc": int(np.argmax(influences)),
                "least_influential_doc": int(np.argmin(influences)),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (influence-ranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
