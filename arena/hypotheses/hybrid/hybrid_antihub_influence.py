"""Hybrid: Anti-Hubness + Influence Function fusion.

Combines the two best-performing novel reranking signals:
1. CSLS anti-hubness correction (penalises universal nearest neighbours)
2. Influence function scoring (leave-one-out centroid perturbation)

Fuses via weighted sum of normalised scores.

Round 1 results: anti-hubness=0.613, influence-fn=0.594 on HotpotQA.
Hypothesis: combining orthogonal signals should beat either alone.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class HybridAntihubInfluenceHypothesis(Hypothesis):
    """Fuse anti-hubness CSLS with influence function scores."""

    def __init__(self, hubness_weight: float = 0.5, reference_k: int = 5):
        self.hubness_weight = hubness_weight
        self.reference_k = reference_k

    @property
    def name(self) -> str:
        return f"hybrid-antihub-influence-{self.hubness_weight}w"

    @property
    def description(self) -> str:
        return "Hybrid anti-hubness + influence function fusion"

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 4:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        n = len(results)
        k = min(self.reference_k, n - 1)

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        # --- Signal 1: Anti-hubness CSLS ---
        sim_matrix = E @ E.T
        np.fill_diagonal(sim_matrix, -1)

        mean_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(sim_matrix[i])[::-1][:k]
            mean_sim[i] = top_k_sims.mean()

        query_sims = E @ q
        csls_scores = 2 * query_sims - mean_sim

        # --- Signal 2: Influence function ---
        centroid = E.mean(axis=0)
        centroid_sim = float(centroid @ q / max(np.linalg.norm(centroid), 1e-12))

        influence_scores = np.zeros(n)
        for i in range(n):
            # Centroid without doc i
            loo_centroid = (E.sum(axis=0) - E[i]) / max(n - 1, 1)
            loo_norm = np.linalg.norm(loo_centroid)
            if loo_norm > 1e-12:
                loo_sim = float((loo_centroid / loo_norm) @ q)
            else:
                loo_sim = 0.0
            # Positive influence = removing it hurts (centroid moves away from query)
            influence_scores[i] = centroid_sim - loo_sim

        # --- Fusion ---
        csls_norm = self._min_max_norm(csls_scores)
        influence_norm = self._min_max_norm(influence_scores)

        w = self.hubness_weight
        fused = w * csls_norm + (1 - w) * influence_norm

        order = np.argsort(fused)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "csls_scores": csls_scores.tolist(),
                "influence_scores": influence_scores.tolist(),
                "fused_scores": fused[order].tolist(),
                "rerank_order": order.tolist(),
            },
        )

    def _min_max_norm(self, x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (hybrid anti-hubness + influence):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
