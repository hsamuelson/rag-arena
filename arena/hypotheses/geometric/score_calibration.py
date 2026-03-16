"""Score distribution calibration.

Hypothesis: Raw cosine similarity scores are not comparable across queries.
A score of 0.85 for a specific factual query (dense local cluster) means
something very different from 0.85 for a broad exploratory query (sparse
spread). At scale, this inconsistency causes some queries to retrieve
too many distractors while others miss relevant docs.

Fix: Normalise scores by the local density/distribution of the embedding
neighbourhood. A document scoring 0.85 in a neighbourhood where the
mean similarity is 0.90 is less impressive than 0.85 where mean is 0.70.

Algorithm:
1. For each retrieved doc, estimate local density (mean distance to k-NN)
2. Convert raw similarity to z-score: (sim - local_mean) / local_std
3. Use calibrated scores for ranking and cutoff decisions
4. Apply adaptive cutoff based on calibrated score gap

References:
  - Zelnik-Manor & Perona (2004): Self-Tuning Spectral Clustering
  - Local scaling for distance metrics
  - Score normalisation in federated search / multi-index retrieval
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class ScoreCalibrationHypothesis(Hypothesis):
    """Calibrate retrieval scores by local embedding density."""

    def __init__(self, k_neighbors: int = 5, cutoff_z: float = -0.5):
        self.k_neighbors = k_neighbors
        self.cutoff_z = cutoff_z  # Z-score below which to demote results

    @property
    def name(self) -> str:
        return f"score-calibration-{self.k_neighbors}k"

    @property
    def description(self) -> str:
        return (
            "Score calibration — normalise similarity scores by local "
            "embedding density for cross-query consistency"
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
        k = min(self.k_neighbors, n - 1)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        # Compute query-doc similarities (raw)
        raw_sims = E @ q

        # Compute local density for each doc (mean similarity to its k-NN)
        doc_doc_sims = E @ E.T
        np.fill_diagonal(doc_doc_sims, -1)

        local_means = np.zeros(n)
        local_stds = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(doc_doc_sims[i])[::-1][:k]
            local_means[i] = top_k_sims.mean()
            local_stds[i] = max(top_k_sims.std(), 1e-6)

        # Calibrated score: z-score relative to local neighbourhood
        # High z = this doc is unusually similar to query for its neighbourhood
        z_scores = (raw_sims - local_means) / local_stds

        # Also compute query-relative calibration
        # How does the query-doc similarity compare to the query's overall profile?
        query_mean_sim = raw_sims.mean()
        query_std_sim = max(raw_sims.std(), 1e-6)
        query_z_scores = (raw_sims - query_mean_sim) / query_std_sim

        # Combined calibrated score: average of both z-scores
        calibrated = 0.5 * z_scores + 0.5 * query_z_scores

        # Rank by calibrated score
        order = np.argsort(calibrated)[::-1]
        reranked = [results[i] for i in order]

        # Find natural cutoff
        cutoff_idx = n
        for i, idx in enumerate(order):
            if calibrated[idx] < self.cutoff_z:
                cutoff_idx = i
                break

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked[:max(3, cutoff_idx)]),
            metadata={
                "raw_similarities": raw_sims[order].tolist(),
                "z_scores": z_scores[order].tolist(),
                "calibrated_scores": calibrated[order].tolist(),
                "local_densities": local_means[order].tolist(),
                "suggested_cutoff": cutoff_idx,
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = [f"Retrieved context ({len(results)} calibrated results):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
