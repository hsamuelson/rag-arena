"""Centroid drift reranking.

Hypothesis: Standard top-K retrieval produces a result set whose centroid
drifts away from the query — the retrieved cluster's centre of mass is
biased toward the densest region of the corpus, not toward the query.
Correcting this drift by penalising documents that pull the centroid
away from the query improves answer quality.

Geometric intuition: If you retrieve 10 documents, 7 might cluster
around one sub-topic. The centroid of your context drifts toward that
sub-topic, starving the LLM of the other 30% of relevant information.

Algorithm:
1. Start with top-K by cosine similarity
2. Iteratively swap the document that most reduces |centroid - query|
3. Track the "drift vector" = centroid - query and report it
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class CentroidDriftHypothesis(Hypothesis):
    """Rerank to minimise drift between query and retrieved-set centroid."""

    def __init__(self, max_swaps: int = 20, drift_weight: float = 0.5):
        self.max_swaps = max_swaps
        self.drift_weight = drift_weight

    @property
    def name(self) -> str:
        return f"centroid-drift-{self.drift_weight}w"

    @property
    def description(self) -> str:
        return (
            "Centroid drift correction — rerank to minimise distance between "
            "query embedding and centroid of retrieved set"
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
        selected_mask = np.ones(n, dtype=bool)

        # Normalise everything
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        q = query_embedding / q_norm

        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norms = np.maximum(emb_norms, 1e-12)
        E = embeddings / emb_norms

        # Initial centroid and drift
        centroid = E.mean(axis=0)
        initial_drift = float(np.linalg.norm(centroid - q))

        # Iterative swap: try removing the doc that most reduces drift,
        # keeping at least half the documents
        min_docs = max(3, n // 2)
        removed: list[int] = []

        for _ in range(self.max_swaps):
            if selected_mask.sum() <= min_docs:
                break

            active_indices = np.where(selected_mask)[0]
            centroid = E[selected_mask].mean(axis=0)
            current_drift = float(np.linalg.norm(centroid - q))

            best_remove = -1
            best_drift = current_drift

            for idx in active_indices:
                # Tentative centroid without this document
                mask_copy = selected_mask.copy()
                mask_copy[idx] = False
                new_centroid = E[mask_copy].mean(axis=0)
                new_drift = float(np.linalg.norm(new_centroid - q))

                if new_drift < best_drift:
                    best_drift = new_drift
                    best_remove = idx

            if best_remove < 0 or best_drift >= current_drift - 1e-6:
                break  # No improvement possible

            selected_mask[best_remove] = False
            removed.append(int(best_remove))

        # Build reranked list: selected first (sorted by relevance), removed last
        selected_indices = list(np.where(selected_mask)[0])
        selected_indices.sort(key=lambda i: results[i].score, reverse=True)
        final_order = selected_indices + removed

        reranked = [results[i] for i in final_order]
        final_centroid = E[selected_mask].mean(axis=0)
        final_drift = float(np.linalg.norm(final_centroid - q))

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked[:len(selected_indices)]),
            metadata={
                "initial_drift": initial_drift,
                "final_drift": final_drift,
                "drift_reduction": initial_drift - final_drift,
                "docs_removed": len(removed),
                "docs_kept": len(selected_indices),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (centroid-drift corrected):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
