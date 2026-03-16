"""Hybrid: RRF fusion of top 5 reranking signals.

Takes the 5 best-performing individual reranking methods and fuses
their rankings via Reciprocal Rank Fusion. This is the "ensemble
of winners" approach.

Signals:
1. Raw cosine similarity (original ranking)
2. CSLS anti-hubness scores
3. Influence function scores
4. Topological persistence scores
5. Score calibration z-scores

RRF formula: score(d) = sum_i 1/(k + rank_i(d))
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class HybridRRFTop5Hypothesis(Hypothesis):
    """RRF ensemble of the 5 best reranking signals."""

    def __init__(self, rrf_k: int = 60, reference_k: int = 5):
        self.rrf_k = rrf_k
        self.reference_k = reference_k

    @property
    def name(self) -> str:
        return f"hybrid-rrf-top5-{self.rrf_k}k"

    @property
    def description(self) -> str:
        return "RRF ensemble of top 5 reranking signals (cosine, CSLS, influence, topo, calibration)"

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
        k_nn = min(self.reference_k, n - 1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        raw_sims = E @ q

        doc_doc_sims = E @ E.T
        np.fill_diagonal(doc_doc_sims, -1)

        # Signal 1: Raw cosine (original ranking)
        signal_raw = raw_sims.copy()

        # Signal 2: CSLS anti-hubness
        mean_nn_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(doc_doc_sims[i])[::-1][:k_nn]
            mean_nn_sim[i] = top_k_sims.mean()
        signal_csls = 2 * raw_sims - mean_nn_sim

        # Signal 3: Influence function
        centroid = E.mean(axis=0)
        c_norm = np.linalg.norm(centroid)
        centroid_sim = float((centroid / max(c_norm, 1e-12)) @ q) if c_norm > 1e-12 else 0.0
        signal_influence = np.zeros(n)
        for i in range(n):
            loo_c = (E.sum(axis=0) - E[i]) / max(n - 1, 1)
            loo_n = np.linalg.norm(loo_c)
            loo_s = float((loo_c / max(loo_n, 1e-12)) @ q) if loo_n > 1e-12 else 0.0
            signal_influence[i] = centroid_sim - loo_s

        # Signal 4: Topological persistence (simplified)
        sorted_idx = np.argsort(raw_sims)[::-1]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        component_birth = {}
        signal_topo = np.zeros(n)
        added = set()
        for step, idx in enumerate(sorted_idx):
            added.add(idx)
            component_birth[idx] = step
            for other in added:
                if other != idx and doc_doc_sims[idx, other] > 0.5:
                    ri, ro = find(idx), find(other)
                    if ri != ro:
                        younger = ri if component_birth.get(ri, step) > component_birth.get(ro, step) else ro
                        older = ro if younger == ri else ri
                        life = step - component_birth.get(younger, step)
                        signal_topo[younger] += life
                        parent[younger] = older

        # Signal 5: Score calibration
        local_means = np.zeros(n)
        local_stds = np.zeros(n)
        for i in range(n):
            top_k = np.sort(doc_doc_sims[i])[::-1][:k_nn]
            local_means[i] = top_k.mean()
            local_stds[i] = max(top_k.std(), 1e-6)
        z_scores = (raw_sims - local_means) / local_stds
        q_mean, q_std = raw_sims.mean(), max(raw_sims.std(), 1e-6)
        signal_calib = 0.5 * z_scores + 0.5 * (raw_sims - q_mean) / q_std

        # RRF fusion
        signals = [signal_raw, signal_csls, signal_influence, signal_topo, signal_calib]
        rrf_scores = np.zeros(n)

        for signal in signals:
            ranking = np.argsort(signal)[::-1]
            for rank, idx in enumerate(ranking):
                rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

        order = np.argsort(rrf_scores)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "rrf_scores": rrf_scores[order].tolist(),
                "rerank_order": order.tolist(),
                "signal_names": ["raw", "csls", "influence", "topo", "calibration"],
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (RRF top-5 ensemble):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
