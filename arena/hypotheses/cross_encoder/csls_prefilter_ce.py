"""CSLS Pre-Filter + Deep Pool CE reranking.

Uses CSLS (Cross-domain Similarity Local Scaling) as a candidate *selector*
rather than a reranker. Retrieves top-100, uses CSLS to filter to top-30
(removing hub documents that inflate the candidate pool), then applies CE
reranking on the cleaner set.

Why robust: Hubness is universal in high-dimensional spaces and worsens at
higher dimensions (1024d snowflake > 768d nomic). CSLS was only tested as
a reranker before; using it as a pre-filter avoids the score fusion problem
that hurt CSLS+CE blending.

References:
  - Conneau et al. (2018): CSLS for cross-lingual word retrieval
  - Radovanovic et al. (2010): Hubs in Space
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CSLSPrefilterCEHypothesis(Hypothesis):
    """Top-100 → CSLS filter to 30 → CE rerank to 10."""

    def __init__(
        self,
        initial_pool=100,
        csls_k=30,
        reference_k=5,
        ce_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        final_k=10,
    ):
        self._initial_pool = initial_pool
        self._csls_k = csls_k
        self._reference_k = reference_k
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
        return f"csls-prefilter-ce-{self._initial_pool}-{self._csls_k}"

    @property
    def description(self):
        return (
            f"CSLS pre-filter: top-{self._initial_pool} → CSLS filter to "
            f"{self._csls_k} → CE rerank to {self._final_k}"
        )

    def _csls_filter(self, results, embeddings, query_embedding):
        """Apply CSLS scoring to filter candidates, keeping top csls_k."""
        n = len(results)
        k = min(self._reference_k, n - 1)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        E = embeddings / np.maximum(norms, 1e-12)
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # Query-doc similarities
        raw_sims = E @ q

        # Doc-doc similarities for hubness correction
        doc_doc_sims = E @ E.T
        np.fill_diagonal(doc_doc_sims, -1)

        mean_nn_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(doc_doc_sims[i])[::-1][:k]
            mean_nn_sim[i] = top_k_sims.mean()

        csls_scores = 2 * raw_sims - mean_nn_sim

        # Select top csls_k by CSLS score
        keep = min(self._csls_k, n)
        top_indices = np.argsort(csls_scores)[::-1][:keep]

        filtered_results = [results[i] for i in top_indices]
        filtered_embeddings = embeddings[top_indices]

        return filtered_results, filtered_embeddings, csls_scores[top_indices]

    def apply(self, query, results, embeddings, query_embedding):
        # Retrieve deep pool
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

        pool_size = len(results)

        # CSLS pre-filter (only if we have embeddings)
        csls_scores_used = None
        if embeddings is not None and query_embedding is not None and len(results) >= 4:
            results, embeddings, csls_scores_used = self._csls_filter(
                results, embeddings, query_embedding
            )

        # CE rerank the filtered set
        model = self._get_ce_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (CSLS pre-filter {pool_size} → {len(results)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "initial_pool": pool_size,
                "after_csls_filter": len(results),
                "final_k": len(top),
                "ce_scores": [ce_scores[i] for i in top],
                "csls_scores": csls_scores_used.tolist() if csls_scores_used is not None else None,
            },
        )
