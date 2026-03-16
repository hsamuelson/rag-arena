"""Hub-Aware Deep Pool Replacement + CE reranking.

Retrieves top-70, identifies hub documents (those appearing as nearest
neighbours of many other docs in the pool), replaces them with deeper
candidates, then CE reranks the cleaned pool.

Why robust: Hub documents waste CE budget regardless of embedder quality.
The effect is STRONGER at higher dimensions (snowflake 1024d > nomic 768d)
due to concentration of measure. Unlike anti-hubness which only penalises
hubs in scoring, this approach actively replaces them with fresh candidates.

References:
  - Radovanovic et al. (2010): Hubs in Space, JMLR
  - Dinu & Baroni (2015): Hubness and word embeddings
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class HubAwareDeepPoolCEHypothesis(Hypothesis):
    """Top-70 → replace hub docs with deeper candidates → CE rerank 50 → top-10."""

    def __init__(
        self,
        initial_pool=70,
        replacement_pool=120,
        target_pool=50,
        hub_threshold_percentile=90,
        reference_k=5,
        ce_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        final_k=10,
    ):
        self._initial_pool = initial_pool
        self._replacement_pool = replacement_pool
        self._target_pool = target_pool
        self._hub_percentile = hub_threshold_percentile
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
        return "hub-aware-deep-pool-ce"

    @property
    def description(self):
        return (
            f"Hub-aware pool: top-{self._initial_pool} → detect/replace hubs → "
            f"CE rerank {self._target_pool} → {self._final_k}"
        )

    def _compute_hubness(self, embeddings):
        """Compute hubness count for each document in the pool."""
        n = len(embeddings)
        k = min(self._reference_k, n - 1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        E = embeddings / np.maximum(norms, 1e-12)
        sim_matrix = E @ E.T
        np.fill_diagonal(sim_matrix, -1)

        hubness = np.zeros(n)
        for i in range(n):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:k]
            for j in top_k_indices:
                hubness[j] += 1

        return hubness

    def apply(self, query, results, embeddings, query_embedding):
        # Retrieve initial pool
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._initial_pool
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        n_hubs_removed = 0
        initial_size = len(results)

        # Detect hubs if we have embeddings
        if embeddings is not None and len(results) >= 6:
            hubness = self._compute_hubness(embeddings)
            hub_threshold = np.percentile(hubness, self._hub_percentile)

            # Identify hub indices (high hubness count)
            is_hub = hubness >= max(hub_threshold, 1)
            hub_indices = set(np.where(is_hub)[0])
            n_hubs_removed = len(hub_indices)

            if n_hubs_removed > 0 and n_hubs_removed < len(results):
                # Keep non-hub docs
                non_hub_results = [r for i, r in enumerate(results) if i not in hub_indices]
                hub_doc_ids = {results[i].doc_id for i in hub_indices}

                # Retrieve deeper pool for replacements
                replacement_results = []
                if self._backend is not None:
                    try:
                        deeper_results, _ = self._backend.retrieve_with_embeddings(
                            query, self._replacement_pool
                        )
                        # Get candidates not already in the pool
                        existing_ids = {r.doc_id for r in non_hub_results}
                        replacement_results = [
                            r for r in deeper_results
                            if r.doc_id not in existing_ids
                        ]
                    except Exception:
                        pass

                # Fill up to target pool size
                n_needed = self._target_pool - len(non_hub_results)
                if n_needed > 0 and replacement_results:
                    results = non_hub_results + replacement_results[:n_needed]
                else:
                    results = non_hub_results[:self._target_pool]
            else:
                results = results[:self._target_pool]
        else:
            results = results[:self._target_pool]

        # CE rerank
        model = self._get_ce_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (hub-aware pool, {n_hubs_removed} hubs replaced → {len(results)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "initial_pool": initial_size,
                "hubs_removed": n_hubs_removed,
                "pool_after_replacement": len(results),
                "ce_scores": [ce_scores[i] for i in top],
            },
        )
