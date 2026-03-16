"""Residual Query Retrieval: rerank by what the top results *don't* cover.

Standard CE reranking scores each document independently against the query.
But multi-hop questions often need a "bridge" document that covers a different
facet of the query — one that the top-scoring documents already satisfy.

This hypothesis computes a *residual* query vector: the component of the
query embedding that is orthogonal to the subspace spanned by the top-K
CE-scored documents. Documents that are similar to this residual cover
semantic content that the top results are missing, so we boost them.

Algorithm:
1. Retrieve top-50 from the backend (deep pool).
2. CE-score all 50 candidates.
3. Compute query and document embeddings.
4. Take the top-5 by CE score as the "covered" subspace.
5. Project the query embedding onto the span of those 5 doc embeddings
   and compute the residual (the orthogonal complement).
6. For the remaining 45 docs, add a residual-similarity bonus:
       final_score = ce_score + lambda * cosine(doc_emb, residual)
7. Sort by final score, return top-10.

This promotes "bridge" documents from deep in the pool (e.g., rank 30)
up to the final result set when they cover the missing semantic signal.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class ResidualQueryHypothesis(Hypothesis):
    """Deep pool CE reranking with residual query boosting."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        top_k_subspace: int = 5,
        residual_weight: float = 0.3,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._top_k_subspace = top_k_subspace
        self._residual_weight = residual_weight
        self._model = None  # lazy-loaded
        self._backend = None

    def set_backend(self, backend):
        """Inject the backend for deep retrieval and embedding access."""
        self._backend = backend

    def _get_model(self):
        """Lazy-load the CrossEncoder to avoid slow imports at startup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "residual-query"

    @property
    def description(self) -> str:
        return (
            f"Deep pool ({self._pool_size}) CE rerank with residual query boosting "
            f"(top-{self._top_k_subspace} subspace, lambda={self._residual_weight})"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        # --- Step 1: Deep retrieval ---
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to whatever was passed in

        if not results:
            return HypothesisResult(
                results=[],
                context_prompt="Retrieved context:\n(no results)",
                metadata={},
            )

        # --- Step 2: CE-score all candidates ---
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # --- Step 3: Get embeddings ---
        # Prefer embeddings from the backend (already returned with retrieval).
        # If we don't have them, compute them via embed_batch.
        if embeddings is None and self._backend is not None:
            try:
                doc_texts = [r.text for r in results]
                embeddings = self._backend.embed_batch(doc_texts)
            except Exception:
                pass

        if query_embedding is None and self._backend is not None:
            try:
                query_embedding = self._backend.embed_query(query)
            except Exception:
                pass

        # If we still don't have embeddings, fall back to pure CE reranking.
        if embeddings is None or query_embedding is None:
            return self._fallback_ce_only(results, ce_scores)

        # --- Step 4: Identify top-K by CE as the "covered" subspace ---
        n_subspace = min(self._top_k_subspace, len(results))
        ce_ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        subspace_indices = ce_ranked[:n_subspace]

        # D_sub: (n_subspace, dim)
        D_sub = embeddings[subspace_indices].astype(np.float64)

        # --- Step 5: Compute the residual ---
        q = query_embedding.astype(np.float64)

        # Project q onto the column space of D_sub.T  (i.e., the row space of D_sub).
        # proj = D_sub.T @ pinv(D_sub.T) @ q
        # Equivalently: proj = D_sub.T @ (D_sub @ D_sub.T)^{-1} @ D_sub @ q
        # Using pinv for numerical stability:
        #   pinv(D_sub.T) has shape (n_subspace, dim)
        #   D_sub.T has shape (dim, n_subspace)
        # So proj = D_sub.T @ pinv(D_sub.T) @ q  →  (dim,)
        D_sub_T = D_sub.T  # (dim, n_subspace)
        pinv_D_sub_T = np.linalg.pinv(D_sub_T)  # (n_subspace, dim)
        proj = D_sub_T @ (pinv_D_sub_T @ q)  # (dim,)

        residual = q - proj

        residual_norm = np.linalg.norm(residual)
        if residual_norm < 1e-2:
            # The query is almost entirely covered by the top-K docs.
            # No meaningful residual — just do standard CE reranking.
            return self._fallback_ce_only(results, ce_scores)

        # Normalize to unit length.
        residual = residual / residual_norm

        # --- Step 6: Compute residual similarity bonus ---
        # Normalize doc embeddings for cosine similarity.
        emb_f64 = embeddings.astype(np.float64)
        norms = np.linalg.norm(emb_f64, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        emb_normed = emb_f64 / norms

        residual_sims = emb_normed @ residual  # (N,)

        # --- Step 7: Combined score ---
        final_scores = [
            ce_scores[i] + self._residual_weight * residual_sims[i]
            for i in range(len(results))
        ]

        # Sort by final score, take top final_k.
        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_indices = ranked_indices[: self._final_k]

        reranked = [results[i] for i in top_indices]

        # --- Format context ---
        lines = [
            f"Retrieved context (residual query, pool {len(results)} "
            f"→ {len(top_indices)}, CE + residual reranked):"
        ]
        for rank, idx in enumerate(top_indices, 1):
            lines.append(
                f"\n[{rank}] (final: {final_scores[idx]:.4f}, "
                f"ce: {ce_scores[idx]:.4f}, "
                f"residual_sim: {residual_sims[idx]:.4f})"
            )
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top_indices],
                "residual_sims": [float(residual_sims[i]) for i in top_indices],
                "final_scores": [final_scores[i] for i in top_indices],
                "residual_norm": float(residual_norm),
                "pool_size": len(results),
                "final_k": len(top_indices),
                "subspace_k": n_subspace,
                "residual_weight": self._residual_weight,
            },
        )

    def _fallback_ce_only(
        self,
        results: list[RetrievalResult],
        ce_scores: list[float],
    ) -> HypothesisResult:
        """Pure CE reranking when residual computation is not possible."""
        ranked_indices = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top_indices = ranked_indices[: self._final_k]
        reranked = [results[i] for i in top_indices]

        lines = [f"Retrieved context (residual query fallback — CE only, pool {len(results)} → {len(top_indices)}):"]
        for rank, idx in enumerate(top_indices, 1):
            lines.append(f"\n[{rank}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top_indices],
                "pool_size": len(results),
                "final_k": len(top_indices),
                "fallback": True,
            },
        )
