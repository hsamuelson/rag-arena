"""Anti-hubness correction.

Hypothesis: In high-dimensional embedding spaces at scale, certain documents
become "hubs" — they appear as nearest neighbours for a disproportionate
number of queries, regardless of actual relevance. This hubness effect
(Radovanovic et al., 2010) worsens as corpus size grows, causing irrelevant
hub documents to crowd out relevant results.

Fix: Compute the N_k(x) count (how often each document appears in top-K of
other documents) and penalise high-hubness documents.

Geometric intuition: Hubs are points near the "centre" of the embedding
distribution that happen to be close to many other points due to the
concentration of measure in high dimensions. They're false positives
masquerading as relevant results.

References:
  - Radovanovic et al. (2010): Hubs in Space, JMLR
  - Dinu & Baroni (2015): Hubness and word embeddings
  - DeepMind (2025): Theoretical Limitations of Embedding-Based Retrieval
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class AntiHubnessHypothesis(Hypothesis):
    """Penalise hub documents that appear as universal nearest neighbours."""

    def __init__(self, hubness_penalty: float = 0.4, reference_k: int = 5):
        self.hubness_penalty = hubness_penalty
        self.reference_k = reference_k

    @property
    def name(self) -> str:
        return f"anti-hubness-{self.hubness_penalty}p"

    @property
    def description(self) -> str:
        return (
            "Anti-hubness correction — penalise documents that are hubs "
            "(universal nearest neighbours) in the retrieved embedding space"
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
        k = min(self.reference_k, n - 1)

        # Compute pairwise cosine similarities among retrieved docs
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        sim_matrix = E @ E.T
        np.fill_diagonal(sim_matrix, -1)  # exclude self

        # Count hubness: how many times each doc appears in top-K of others
        hubness = np.zeros(n)
        for i in range(n):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
            for j in top_k_indices:
                hubness[j] += 1

        # Normalise hubness to [0, 1]
        h_max = hubness.max()
        if h_max > 0:
            hubness_norm = hubness / h_max
        else:
            hubness_norm = np.zeros(n)

        # Also compute CSLS (Cross-domain Similarity Local Scaling)
        # which corrects for hubness at query time
        # CSLS(q, d) = 2*cos(q,d) - mean_k(cos(d, nn_k(d)))
        mean_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(sim_matrix[i])[::-1][:k]
            mean_sim[i] = top_k_sims.mean()

        if query_embedding is not None:
            q_norm = np.linalg.norm(query_embedding)
            if q_norm > 1e-12:
                q = query_embedding / q_norm
                query_sims = E @ q
                csls_scores = 2 * query_sims - mean_sim
            else:
                csls_scores = np.array([r.score for r in results])
        else:
            csls_scores = np.array([r.score for r in results])

        # Adjusted scores: blend original score with CSLS, penalise hubness
        original_scores = np.array([r.score for r in results])
        adjusted = (
            (1 - self.hubness_penalty) * original_scores
            + self.hubness_penalty * self._min_max_norm(csls_scores)
        )

        order = np.argsort(adjusted)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "hubness_counts": hubness.tolist(),
                "hubness_max": float(h_max),
                "hubness_mean": float(hubness.mean()),
                "csls_scores": csls_scores.tolist(),
                "rerank_order": order.tolist(),
            },
        )

    def _min_max_norm(self, x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (anti-hubness corrected):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
