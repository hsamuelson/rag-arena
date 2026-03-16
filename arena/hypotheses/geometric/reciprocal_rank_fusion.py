"""Reciprocal Rank Fusion (RRF) from multiple embedding perspectives.

Hypothesis: A single cosine similarity score captures only one "view" of
query-document relevance. By constructing multiple views — raw similarity,
mean-centred similarity, and cluster-centroid similarity — and fusing their
rankings with RRF, we obtain a more robust ranking that is less sensitive
to embedding space artefacts.

This simulates the multi-signal retrieval strategy used by systems like
CRAG (Corrective RAG) without requiring multiple retrieval backends.

Algorithm:
1. View A: raw cosine similarity (original scores)
2. View B: mean-centred cosine similarity (remove shared bias)
3. View C: similarity to the centroid of top-3 docs (cluster proxy)
4. For each document, compute RRF score = sum(1 / (k + rank_i)) across views
5. Re-rank by fused RRF score

Geometric intuition: Each view projects the relevance landscape onto a
different axis. RRF is rank-level ensembling — a document that appears
near the top across all views is genuinely relevant, while one that ranks
high only in raw cosine may be an artefact of anisotropy or hubness.

References:
  - Cormack et al. (2009): Reciprocal Rank Fusion outperforms Condorcet
  - Shi et al. (2024): CRAG — Corrective RAG with multi-signal retrieval
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class RRFMultiPerspectiveHypothesis(Hypothesis):
    """Fuse multiple embedding similarity views using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60, top_centroid: int = 3):
        """
        Args:
            k: RRF constant (higher = more weight to lower-ranked docs).
            top_centroid: Number of top docs used to form the centroid for view C.
        """
        self.k = k
        self.top_centroid = top_centroid

    @property
    def name(self) -> str:
        return "rrf-multi-perspective"

    @property
    def description(self) -> str:
        return (
            "Reciprocal Rank Fusion across multiple embedding views — "
            "raw cosine, mean-centred, and centroid similarity"
        )

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

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # View A: raw cosine similarity
        sims_raw = E @ q

        # View B: mean-centred cosine similarity
        mean_emb = E.mean(axis=0)
        E_centred = E - mean_emb
        q_centred = q - mean_emb
        # Re-normalise
        cn = np.linalg.norm(E_centred, axis=1, keepdims=True)
        cn = np.maximum(cn, 1e-12)
        E_centred = E_centred / cn
        qcn = max(np.linalg.norm(q_centred), 1e-12)
        q_centred = q_centred / qcn
        sims_centred = E_centred @ q_centred

        # View C: similarity to top-3 centroid
        top_idx = np.argsort(sims_raw)[::-1][:self.top_centroid]
        centroid = E[top_idx].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims_centroid = E @ centroid

        # Compute ranks for each view (0-indexed)
        views = [sims_raw, sims_centred, sims_centroid]
        view_names = ["raw", "centred", "centroid"]
        ranks = []
        for sims in views:
            order = np.argsort(sims)[::-1]
            rank_arr = np.empty(n, dtype=int)
            for rank, idx in enumerate(order):
                rank_arr[idx] = rank
            ranks.append(rank_arr)

        # RRF fusion: score_i = sum(1 / (k + rank_i_v)) for each view v
        rrf_scores = np.zeros(n)
        for rank_arr in ranks:
            rrf_scores += 1.0 / (self.k + rank_arr)

        # Re-rank by RRF score (descending)
        new_order = np.argsort(rrf_scores)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        # Measure rank changes
        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "rrf_scores": rrf_scores[new_order].tolist(),
                "view_rank_correlations": {
                    f"{view_names[i]}_vs_{view_names[j]}": float(
                        np.corrcoef(ranks[i], ranks[j])[0, 1]
                    )
                    for i in range(len(views))
                    for j in range(i + 1, len(views))
                },
                "rank_changes": rank_changes,
                "k": self.k,
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (RRF multi-perspective fusion):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
