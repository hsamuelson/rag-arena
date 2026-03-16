"""LID-Gated Adaptive Pool Depth + CE reranking.

Uses Local Intrinsic Dimensionality (LID) of the query neighbourhood to
adaptively set the candidate pool depth before CE reranking. When the local
manifold around the query is complex (high LID), the initial retrieval is
less reliable → expand the pool to give CE more candidates. When LID is low,
the dense retriever is confident → keep a small pool.

Why robust: LID measures local manifold complexity, which is orthogonal to
embedder quality. Higher-dimensional embedders (snowflake 1024d) tend to
have higher ambient LID, making this gating MORE useful, not less.

References:
  - Amsaleg et al. (2015): TwoNN estimator
  - Houle (2017): Local Intrinsic Dimensionality
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class LIDGatedPoolCEHypothesis(Hypothesis):
    """Adaptive pool depth gated by query-neighbourhood LID, then CE rerank."""

    def __init__(
        self,
        initial_pool=20,
        low_lid_pool=30,
        high_lid_pool=80,
        lid_threshold=8.0,
        lid_k=5,
        ce_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        final_k=10,
    ):
        self._initial_pool = initial_pool
        self._low_lid_pool = low_lid_pool
        self._high_lid_pool = high_lid_pool
        self._lid_threshold = lid_threshold
        self._lid_k = lid_k
        self._ce_model_name = ce_model
        self._final_k = final_k
        self._ce_model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_ce_model(self):
        if self._ce_model is None:
            from sentence_transformers import CrossEncoder
            self._ce_model = CrossEncoder(self._ce_model_name)
        return self._ce_model

    @property
    def name(self):
        return "lid-gated-pool-ce"

    @property
    def description(self):
        return (
            f"LID-gated adaptive pool: initial {self._initial_pool}, "
            f"expand to {self._high_lid_pool} if LID > {self._lid_threshold}, "
            f"else {self._low_lid_pool} → CE rerank to {self._final_k}"
        )

    def _estimate_query_lid(self, query_embedding, embeddings):
        """Estimate LID of the query point relative to retrieved documents.

        Uses MLE estimator over k nearest neighbours:
        LID = -k / sum(log(d_j / d_k)) for j=1..k-1
        """
        n = len(embeddings)
        k = min(self._lid_k, n - 1)
        if k < 2:
            return 0.0

        # Cosine distances from query to each doc
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        E = embeddings / np.maximum(norms, 1e-12)
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        cosine_sims = E @ q
        dists = 1.0 - cosine_sims  # cosine distance
        dists = np.maximum(dists, 1e-12)

        # Sort and use k nearest
        sorted_dists = np.sort(dists)[:k]
        d_k = sorted_dists[-1]

        if d_k < 1e-10:
            return 0.0

        m = k - 1
        log_ratios = np.log(sorted_dists[:m] / d_k)
        sum_log = log_ratios.sum()

        if abs(sum_log) < 1e-12:
            return 0.0

        return -m / sum_log

    def apply(self, query, results, embeddings, query_embedding):
        # Initial shallow retrieval to estimate LID
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._initial_pool
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Get query embedding if needed
        if query_embedding is None and self._backend is not None:
            try:
                query_embedding = self._backend.embed_query(query)
            except Exception:
                pass

        # Estimate LID from initial pool
        query_lid = 0.0
        if embeddings is not None and query_embedding is not None:
            query_lid = self._estimate_query_lid(query_embedding, embeddings)

        # Gate: decide pool depth
        if query_lid > self._lid_threshold:
            target_pool = self._high_lid_pool
        else:
            target_pool = self._low_lid_pool

        # Re-retrieve with adapted pool if needed
        if target_pool > len(results) and self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, target_pool
                )
            except Exception:
                pass

        # CE rerank
        model = self._get_ce_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (LID-gated pool {len(results)}, LID={query_lid:.1f} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "query_lid": float(query_lid),
                "lid_threshold": self._lid_threshold,
                "pool_depth_used": len(results),
                "target_pool": target_pool,
                "expanded": target_pool == self._high_lid_pool,
                "ce_scores": [ce_scores[i] for i in top],
            },
        )
