"""Spectral Query Decomposition — eigendecompose the doc similarity matrix.

Instead of scoring documents only against the original query, we discover
latent "facets" of the retrieved set by eigen-decomposing the pairwise
cosine-similarity matrix among the top-50 candidates.  Each dominant
eigenvector defines a weighted combination of document embeddings that acts
as a facet query.  A document's spectral score is its maximum cosine
similarity to any facet query, capturing whether it is strongly aligned
with *any* latent topic in the pool.

Final ranking fuses cross-encoder scores with spectral facet scores:

    final_score = ce_score + lambda * facet_similarity

This rewards documents that a cross-encoder judges relevant *and* that
occupy a dominant direction in embedding space, suppressing outlier
documents that score well on CE alone but sit in a thin, noisy region.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class SpectralQueryDecompHypothesis(Hypothesis):
    """Retrieve deep pool, decompose doc similarity spectrally, CE + facet rerank."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        variance_threshold: float = 0.90,
        max_facets: int = 5,
        spectral_weight: float = 0.3,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._variance_threshold = variance_threshold
        self._max_facets = max_facets
        self._spectral_weight = spectral_weight
        self._model = None  # lazy-loaded CE
        self._backend = None

    # -- backend injection (same pattern as deep_pool_ce) ------------------

    def set_backend(self, backend):
        """Inject the backend for deep retrieval and embedding access."""
        self._backend = backend

    # -- lazy cross-encoder ------------------------------------------------

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    # -- properties --------------------------------------------------------

    @property
    def name(self) -> str:
        return "spectral-query-decomp"

    @property
    def description(self) -> str:
        return (
            f"Retrieve top-{self._pool_size}, eigendecompose doc similarity "
            f"to find facet queries, CE + spectral rerank to top-{self._final_k}"
        )

    # -- core algorithm ----------------------------------------------------

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        # 1. Deep retrieval via backend (fall back to provided results)
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to caller-supplied results

        if not results:
            return HypothesisResult(
                results=[],
                context_prompt="Retrieved context:\n(no results)",
                metadata={},
            )

        n_docs = len(results)

        # 2. Obtain doc embeddings and query embedding
        if embeddings is None and self._backend is not None:
            embeddings = self._backend.embed_batch([r.text[:_MAX_CHARS] for r in results])
        if query_embedding is None and self._backend is not None:
            query_embedding = self._backend.embed_query(query)

        # 3. Compute spectral facet similarities (needs embeddings)
        facet_similarities = np.zeros(n_docs)
        n_facets_used = 0

        if embeddings is not None and len(embeddings) >= 2:
            facet_similarities, n_facets_used = self._spectral_facet_scores(embeddings)

        # 4. CE score all docs
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = np.array(model.predict(pairs), dtype=np.float64)

        # 5. Combine: final_score = ce_score + lambda * facet_similarity
        final_scores = ce_scores + self._spectral_weight * facet_similarities

        # 6. Rank and take top-K
        ranked_indices = np.argsort(-final_scores).tolist()
        top_indices = ranked_indices[: self._final_k]

        reranked = [results[i] for i in top_indices]

        # 7. Build context prompt
        lines = [
            f"Retrieved context (spectral query decomp, pool {n_docs} "
            f"-> {len(top_indices)}, {n_facets_used} facets):"
        ]
        for rank, idx in enumerate(top_indices, 1):
            lines.append(
                f"\n[{rank}] (ce: {ce_scores[idx]:.4f}, "
                f"facet: {facet_similarities[idx]:.4f}, "
                f"final: {final_scores[idx]:.4f})"
            )
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [float(ce_scores[i]) for i in top_indices],
                "facet_similarities": [float(facet_similarities[i]) for i in top_indices],
                "final_scores": [float(final_scores[i]) for i in top_indices],
                "n_facets": n_facets_used,
                "pool_size": n_docs,
                "final_k": len(top_indices),
                "spectral_weight": self._spectral_weight,
            },
        )

    # -- spectral decomposition helper -------------------------------------

    def _spectral_facet_scores(self, embeddings: np.ndarray) -> tuple[np.ndarray, int]:
        """Eigendecompose doc-doc similarity and score docs against facet queries.

        Returns:
            (facet_similarities, n_facets) where facet_similarities is shape (N,).
        """
        # Normalize embeddings for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        emb_normed = embeddings / norms

        # 3. Pairwise cosine similarity matrix S (N x N)
        S = emb_normed @ emb_normed.T

        # 4. Eigendecompose (S is symmetric, use eigh for stability)
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # eigh returns ascending order; flip to descending
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Clamp negative eigenvalues (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        # 5. Select top-m eigenvectors capturing >= variance_threshold of total variance
        total_var = eigenvalues.sum()
        if total_var < 1e-12:
            return np.zeros(len(embeddings)), 0

        cumulative = np.cumsum(eigenvalues) / total_var
        # Number of components to reach the threshold, capped at max_facets
        m = int(np.searchsorted(cumulative, self._variance_threshold) + 1)
        m = min(m, self._max_facets, len(eigenvalues))

        # 6. Build facet queries: facet_query_i = normalize(sum_j v_i[j] * emb_normed[j])
        facet_queries = []
        for i in range(m):
            v = eigenvectors[:, i]  # (N,) weights for each doc
            facet_q = v @ emb_normed  # (D,) weighted combination
            fq_norm = np.linalg.norm(facet_q)
            if fq_norm > 1e-10:
                facet_q = facet_q / fq_norm
            facet_queries.append(facet_q)

        if not facet_queries:
            return np.zeros(len(embeddings)), 0

        facet_queries = np.stack(facet_queries, axis=0)  # (m, D)

        # 7. Score each doc: max over facets of cosine(doc_emb, facet_query)
        # cosines shape: (N, m) = emb_normed (N, D) @ facet_queries.T (D, m)
        cosines = emb_normed @ facet_queries.T
        facet_similarities = cosines.max(axis=1)  # (N,)

        return facet_similarities, m
