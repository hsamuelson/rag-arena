"""Random projection ensemble reranking.

Hypothesis (Novel #16): A single high-dimensional similarity score is
fragile — it conflates many semantic dimensions into one number. Inspired
by the Johnson-Lindenstrauss lemma, which guarantees that random projections
approximately preserve pairwise distances, we create an ensemble of rankings
in multiple randomly projected subspaces and fuse them.

Geometric intuition: Each random projection slices the embedding space along
a different random hyperplane, emphasising different latent semantic aspects.
A document that ranks highly in most projections is robustly similar to the
query across many semantic dimensions. A document that ranks highly in only
one projection may be a false positive along that particular axis.

Algorithm:
1. Generate P random Gaussian projection matrices (D -> d, where d << D).
2. Project both query and document embeddings into each subspace.
3. Rank documents by cosine similarity in each projected space.
4. Fuse rankings using Reciprocal Rank Fusion (RRF).
5. The consensus ranking is more robust than any single projection.

Reference:
  - Johnson & Lindenstrauss (1984): random projections preserve distances
  - Cormack et al. (2009): Reciprocal Rank Fusion
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class RandomProjectionEnsembleHypothesis(Hypothesis):
    """Rerank via consensus of multiple random-projection similarity rankings."""

    def __init__(
        self,
        n_projections: int = 5,
        target_dim: int = 32,
        rrf_k: int = 60,
        seed: int = 42,
    ):
        self.n_projections = n_projections
        self.target_dim = target_dim
        self.rrf_k = rrf_k
        self.seed = seed

    @property
    def name(self) -> str:
        return "random-projection-ensemble"

    @property
    def description(self) -> str:
        return (
            "Random projection ensemble — fuse rankings from multiple JL "
            "projections via RRF for robust similarity estimation"
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
        D = embeddings.shape[1]

        # Normalise
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

        target_dim = min(self.target_dim, D)
        rng = np.random.RandomState(self.seed)

        # Generate rankings in each random projection
        all_rankings = []  # list of arrays, each of shape (n,) with rank indices
        all_sims = []

        for p in range(self.n_projections):
            # Random Gaussian projection matrix, scaled by 1/sqrt(d)
            P = rng.randn(D, target_dim) / np.sqrt(target_dim)

            # Project
            E_proj = E @ P  # (n, target_dim)
            q_proj = q @ P  # (target_dim,)

            # Normalise projected vectors
            E_proj_norms = np.linalg.norm(E_proj, axis=1, keepdims=True)
            E_proj_norms = np.maximum(E_proj_norms, 1e-12)
            E_proj = E_proj / E_proj_norms

            q_proj_norm = np.linalg.norm(q_proj)
            if q_proj_norm > 1e-12:
                q_proj = q_proj / q_proj_norm

            # Cosine similarities in projected space
            sims = E_proj @ q_proj  # (n,)
            ranking = np.argsort(sims)[::-1]  # best first

            all_rankings.append(ranking)
            all_sims.append(sims)

        # Reciprocal Rank Fusion
        rrf_scores = np.zeros(n)
        for ranking in all_rankings:
            for rank, doc_idx in enumerate(ranking):
                rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)

        # Final ranking by RRF score
        order = np.argsort(rrf_scores)[::-1]
        reranked = [results[i] for i in order]

        # Agreement metric: what fraction of projections agree on top-3?
        top3_sets = [set(ranking[:3].tolist()) for ranking in all_rankings]
        pairwise_overlaps = []
        for i in range(len(top3_sets)):
            for j in range(i + 1, len(top3_sets)):
                overlap = len(top3_sets[i] & top3_sets[j]) / 3.0
                pairwise_overlaps.append(overlap)
        mean_agreement = (
            float(np.mean(pairwise_overlaps)) if pairwise_overlaps else 1.0
        )

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "n_projections": self.n_projections,
                "target_dim": target_dim,
                "original_dim": D,
                "rrf_scores": rrf_scores[order].tolist(),
                "top3_agreement": mean_agreement,
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (projection-ensemble ranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
