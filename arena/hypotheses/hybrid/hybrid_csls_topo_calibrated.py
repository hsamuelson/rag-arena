"""Hybrid: CSLS + Topological Persistence + Score Calibration.

Triple fusion of the top 3 reranking signals from Round 1:
1. CSLS anti-hubness (0.613 nDCG)
2. Topological persistence (0.595 nDCG)
3. Score calibration z-scores (0.594 nDCG)

Each signal captures a different aspect:
- CSLS: corrects for neighbourhood density bias
- Persistence: identifies topologically significant documents
- Calibration: normalises scores by local distribution

Fusion via rank-weighted combination.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class HybridCSLSTopoCalibratedHypothesis(Hypothesis):
    """Triple fusion: CSLS + topological persistence + calibrated z-scores."""

    def __init__(self, csls_weight: float = 0.4, topo_weight: float = 0.3,
                 calib_weight: float = 0.3, reference_k: int = 5):
        self.csls_weight = csls_weight
        self.topo_weight = topo_weight
        self.calib_weight = calib_weight
        self.reference_k = reference_k

    @property
    def name(self) -> str:
        return "hybrid-csls-topo-calibrated"

    @property
    def description(self) -> str:
        return "Triple fusion: CSLS anti-hubness + topological persistence + score calibration"

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

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        raw_sims = E @ q

        # --- Signal 1: CSLS anti-hubness ---
        doc_doc_sims = E @ E.T
        np.fill_diagonal(doc_doc_sims, -1)

        mean_nn_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(doc_doc_sims[i])[::-1][:k]
            mean_nn_sim[i] = top_k_sims.mean()

        csls_scores = 2 * raw_sims - mean_nn_sim

        # --- Signal 2: Topological persistence ---
        # Build filtration: add docs in order of decreasing query similarity
        sorted_by_sim = np.argsort(raw_sims)[::-1]

        # Union-Find for connected components
        parent = list(range(n))
        rank_uf = [0] * n
        birth = [0.0] * n
        persistence = np.zeros(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y, death_val):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            # Younger component dies (higher birth = added later = younger)
            if birth[rx] > birth[ry]:
                rx, ry = ry, rx
            # ry dies
            life = death_val - birth[ry]
            for idx in range(n):
                if find(idx) == ry:
                    persistence[idx] += life
            if rank_uf[rx] < rank_uf[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank_uf[rx] == rank_uf[ry]:
                rank_uf[rx] += 1

        added = set()
        for step, idx in enumerate(sorted_by_sim):
            birth[idx] = float(step)
            added.add(idx)
            # Connect to already-added neighbours
            for other in added:
                if other != idx and doc_doc_sims[idx, other] > 0.5:
                    union(idx, other, float(step))

        # Normalise persistence
        if persistence.max() > 0:
            persistence = persistence / persistence.max()

        # --- Signal 3: Score calibration z-scores ---
        local_means = np.zeros(n)
        local_stds = np.zeros(n)
        for i in range(n):
            top_k_sims_i = np.sort(doc_doc_sims[i])[::-1][:k]
            local_means[i] = top_k_sims_i.mean()
            local_stds[i] = max(top_k_sims_i.std(), 1e-6)

        z_scores = (raw_sims - local_means) / local_stds
        query_mean = raw_sims.mean()
        query_std = max(raw_sims.std(), 1e-6)
        query_z = (raw_sims - query_mean) / query_std
        calib_scores = 0.5 * z_scores + 0.5 * query_z

        # --- Fusion via normalised weighted sum ---
        csls_norm = self._min_max_norm(csls_scores)
        topo_norm = self._min_max_norm(persistence)
        calib_norm = self._min_max_norm(calib_scores)

        fused = (self.csls_weight * csls_norm +
                 self.topo_weight * topo_norm +
                 self.calib_weight * calib_norm)

        order = np.argsort(fused)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "csls_scores": csls_scores.tolist(),
                "persistence_scores": persistence.tolist(),
                "calibrated_scores": calib_scores.tolist(),
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
        lines = ["Retrieved context (CSLS + topo + calibrated):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
