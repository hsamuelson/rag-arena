"""Void Detection / Sparse Region Probing.

Bridge documents for multi-hop questions often live in the semantic space
BETWEEN the query and the top results, or between pairs of top results.
Standard retrieval misses these because no query or document sits exactly
in those gap regions.

Algorithm:
1. Deep-retrieve top-50 from backend.
2. Cross-encoder score all 50, sort by CE score.
3. Take the top-5 by CE score as "anchor" documents.
4. Generate probe points in the void regions:
   - Pairwise anchor midpoints: normalize((anchor_i + anchor_j) / 2)
   - Query-anchor bridges: normalize((query_emb + anchor_i) / 2)
   - Centroid probe: normalize((query_emb + mean(anchors)) / 2)
5. For each non-anchor doc, compute max cosine similarity to any probe.
6. Final score = ce_score + lambda * max_probe_similarity.
7. Return top-10 by final score.

The probe similarity boosts documents that fill the voids — the semantic
gaps between our best results where bridge evidence is likely to hide.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or batch of vectors."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms


class VoidDetectionHypothesis(Hypothesis):
    """Probe the voids between top results to find bridge documents."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        n_anchors: int = 5,
        lam: float = 0.2,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._n_anchors = n_anchors
        self._lam = lam
        self._model = None  # lazy-loaded
        self._backend = None

    def set_backend(self, backend):
        """Inject the backend for deep retrieval."""
        self._backend = backend

    def _get_model(self):
        """Lazy-load the CrossEncoder to avoid slow imports at startup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "void-detection"

    @property
    def description(self) -> str:
        return (
            "Probe semantic voids between top CE results to surface "
            "bridge documents for multi-hop reasoning"
        )

    def _generate_probes(
        self,
        query_emb: np.ndarray,
        anchor_embs: np.ndarray,
    ) -> np.ndarray:
        """Generate probe points in the gaps between query and anchors.

        Returns:
            (P, D) array of normalized probe vectors.
        """
        probes = []

        n = len(anchor_embs)

        # Pairwise anchor midpoints: C(n, 2) probes
        for i in range(n):
            for j in range(i + 1, n):
                midpoint = (anchor_embs[i] + anchor_embs[j]) / 2.0
                probes.append(midpoint)

        # Query-anchor bridges: n probes
        for i in range(n):
            bridge = (query_emb + anchor_embs[i]) / 2.0
            probes.append(bridge)

        # Centroid probe: 1 probe
        centroid = anchor_embs.mean(axis=0)
        centroid_probe = (query_emb + centroid) / 2.0
        probes.append(centroid_probe)

        probes = np.array(probes)
        return _normalize(probes)

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        # Deep retrieval from backend
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to normal results

        if not results:
            return HypothesisResult(
                results=[],
                context_prompt="Retrieved context:\n(no results)",
                metadata={},
            )

        # Step 2: CE score all candidates
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Sort indices by CE score descending
        ranked_indices = sorted(
            range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True
        )

        # If we lack embeddings, fall back to pure CE reranking
        if embeddings is None or query_embedding is None:
            top_indices = ranked_indices[: self._final_k]
            reranked = [results[i] for i in top_indices]
            lines = [f"Retrieved context (void-detection, CE-only fallback, pool {len(results)}):"]
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

        # Step 3: Get embeddings for query and docs
        query_emb = query_embedding.astype(np.float64)
        doc_embs = embeddings.astype(np.float64)

        # Step 4: Top anchors by CE score
        n_anchors = min(self._n_anchors, len(ranked_indices))
        anchor_indices = ranked_indices[:n_anchors]
        remaining_indices = ranked_indices[n_anchors:]

        anchor_embs = doc_embs[anchor_indices]

        # Step 5: Generate probe points
        probes = self._generate_probes(query_emb, anchor_embs)

        # Step 6: For each remaining doc, compute max probe similarity
        # probes: (P, D), remaining docs: (R, D)
        remaining_embs = doc_embs[remaining_indices] if remaining_indices else np.empty((0, doc_embs.shape[1]))
        remaining_embs_norm = _normalize(remaining_embs)

        if len(remaining_indices) > 0 and len(probes) > 0:
            # (R, P) similarity matrix
            sim_matrix = remaining_embs_norm @ probes.T
            max_probe_sim = sim_matrix.max(axis=1)  # (R,)
        else:
            max_probe_sim = np.zeros(len(remaining_indices))

        # Step 7: Compute final scores for remaining docs
        remaining_final_scores = []
        for k, idx in enumerate(remaining_indices):
            final = ce_scores[idx] + self._lam * float(max_probe_sim[k])
            remaining_final_scores.append((idx, final))

        # Anchors keep their CE score as final score (they're already top)
        anchor_final_scores = [(idx, ce_scores[idx]) for idx in anchor_indices]

        # Combine and sort all by final score
        all_scored = anchor_final_scores + remaining_final_scores
        all_scored.sort(key=lambda x: x[1], reverse=True)

        # Step 8: Take top-K
        top_entries = all_scored[: self._final_k]
        top_indices = [idx for idx, _ in top_entries]
        top_final = [score for _, score in top_entries]

        reranked = [results[idx] for idx in top_indices]

        # Format context
        lines = [
            f"Retrieved context (void-detection, pool {len(results)}, "
            f"{n_anchors} anchors, {len(probes)} probes, lambda={self._lam}):"
        ]
        for rank, (idx, fscore) in enumerate(top_entries, 1):
            is_anchor = idx in anchor_indices
            tag = "anchor" if is_anchor else "probed"
            lines.append(
                f"\n[{rank}] (final: {fscore:.4f}, ce: {ce_scores[idx]:.4f}, {tag})"
            )
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "final_scores": top_final,
                "ce_scores": [ce_scores[idx] for idx in top_indices],
                "pool_size": len(results),
                "final_k": len(top_entries),
                "n_anchors": n_anchors,
                "n_probes": len(probes),
                "lambda": self._lam,
                "anchor_indices": anchor_indices,
            },
        )
