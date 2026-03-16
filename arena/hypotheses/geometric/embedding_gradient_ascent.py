"""Embedding Gradient Ascent: step the query toward high-relevance regions.

Cross-encoder scores tell us which documents are truly relevant, but the
original query embedding may sit far from the cluster of relevant docs.
This hypothesis uses CE scores as a relevance signal to compute a gradient
in embedding space — the direction from the query toward the weighted
centroid of high-scoring documents — then steps the query along that
gradient. Documents near the stepped query that were previously underranked
get boosted, combining the precision of CE scoring with geometric
repositioning of the query.

Algorithm:
1. Deep retrieval: fetch top-50 candidates from the backend.
2. CE score all 50 documents.
3. Compute a relevance gradient at the query point using CE-weighted
   document offsets.
4. Step the query embedding along the gradient and re-normalise.
5. Final score = ce_score + lambda * cosine(doc, stepped_query).
6. Return top-10 by final score.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class EmbeddingGradientAscentHypothesis(Hypothesis):
    """Step the query embedding toward high-CE-score regions, then rescore."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        alpha: float = 0.5,
        lam: float = 0.3,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._alpha = alpha
        self._lam = lam
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
        return "embedding-gradient-ascent"

    @property
    def description(self) -> str:
        return (
            f"Deep pool ({self._pool_size}) + CE scoring + gradient ascent step "
            f"(alpha={self._alpha}) toward high-relevance embedding region, "
            f"final rescore with lambda={self._lam}"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        # --- Step 1: Deep retrieval from backend ---
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

        # --- Step 2: CE score all candidates ---
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # --- Step 3: Get embeddings ---
        if self._backend is not None:
            query_emb = self._backend.embed_query(query)
            doc_embs = self._backend.embed_batch([r.text[:_MAX_CHARS] for r in results])
        elif embeddings is not None and query_embedding is not None:
            query_emb = query_embedding
            doc_embs = embeddings
        else:
            # Cannot do gradient ascent without embeddings; fall back to pure CE ranking.
            return self._fallback_ce_only(results, ce_scores)

        query_emb = np.asarray(query_emb, dtype=np.float64)
        doc_embs = np.asarray(doc_embs, dtype=np.float64)

        # --- Step 4: Compute relevance gradient ---
        scores_arr = np.array(ce_scores, dtype=np.float64)
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max - s_min > 1e-9:
            weights = (scores_arr - s_min) / (s_max - s_min)
        else:
            weights = np.ones_like(scores_arr)

        # gradient = weighted mean of (doc_emb_i - query_emb)
        offsets = doc_embs - query_emb[np.newaxis, :]  # (N, D)
        weight_sum = weights.sum()
        if weight_sum > 1e-9:
            gradient = (weights[:, np.newaxis] * offsets).sum(axis=0) / weight_sum
        else:
            gradient = np.zeros_like(query_emb)

        # --- Step 5: Step the query embedding ---
        query_new = query_emb + self._alpha * gradient
        norm = np.linalg.norm(query_new)
        if norm > 1e-9:
            query_new = query_new / norm

        # --- Step 6: Cosine similarity of each doc to the stepped query ---
        # Normalise doc embeddings for cosine computation.
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_norms = np.maximum(doc_norms, 1e-9)
        doc_embs_normed = doc_embs / doc_norms
        cosine_scores = (doc_embs_normed @ query_new).tolist()  # (N,)

        # --- Step 7: Final combined score ---
        final_scores = [
            ce + self._lam * cos
            for ce, cos in zip(ce_scores, cosine_scores)
        ]

        # --- Step 8: Return top-K by final score ---
        ranked_indices = sorted(
            range(len(final_scores)), key=lambda i: final_scores[i], reverse=True
        )
        top_indices = ranked_indices[: self._final_k]

        reranked = [results[i] for i in top_indices]

        # Build context prompt.
        lines = [
            f"Retrieved context (gradient ascent: pool {len(results)} "
            f"-> {len(top_indices)}, alpha={self._alpha}, lambda={self._lam}):"
        ]
        for rank, idx in enumerate(top_indices, 1):
            lines.append(
                f"\n[{rank}] (ce: {ce_scores[idx]:.4f}, "
                f"cos_new: {cosine_scores[idx]:.4f}, "
                f"final: {final_scores[idx]:.4f})"
            )
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top_indices],
                "cosine_new_scores": [cosine_scores[i] for i in top_indices],
                "final_scores": [final_scores[i] for i in top_indices],
                "pool_size": len(results),
                "final_k": len(top_indices),
                "alpha": self._alpha,
                "lambda": self._lam,
                "gradient_norm": float(np.linalg.norm(gradient)),
            },
        )

    def _fallback_ce_only(
        self,
        results: list[RetrievalResult],
        ce_scores: list[float],
    ) -> HypothesisResult:
        """Pure CE ranking when embeddings are unavailable."""
        ranked_indices = sorted(
            range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True
        )
        top_indices = ranked_indices[: self._final_k]
        reranked = [results[i] for i in top_indices]

        lines = ["Retrieved context (gradient ascent fallback — CE only):"]
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
