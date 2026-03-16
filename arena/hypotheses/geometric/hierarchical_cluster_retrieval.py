"""Hierarchical cluster retrieval (RAPTOR-style).

Hypothesis: Flat retrieval treats all documents independently, missing the
hierarchical semantic structure of the retrieved set. By building a two-level
clustering hierarchy and scoring documents based on both their direct
similarity AND their cluster's similarity to the query, we capture
"topic-level" relevance that individual document scores miss.

This simulates RAPTOR's tree-based retrieval without requiring pre-built
summary trees — we construct the hierarchy on-the-fly from retrieved
embeddings.

Algorithm:
1. Cluster retrieved embeddings at two levels using agglomerative clustering:
   - Level 1: fine-grained clusters (more clusters)
   - Level 2: coarse clusters (fewer clusters, merging level-1 clusters)
2. Compute cluster centroids (summary embeddings) at both levels
3. Score each document: alpha * doc_sim + beta * L1_cluster_sim + gamma * L2_cluster_sim
4. Re-rank by combined hierarchical score

Geometric intuition: A document in a highly relevant cluster gets a boost
even if its individual similarity is slightly lower. This is analogous to
how RAPTOR retrieves from summary nodes — a cluster centroid acts as a
summary embedding.

References:
  - Sarthi et al. (2024): RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
  - Ward's method for hierarchical clustering
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class HierarchicalClusterHypothesis(Hypothesis):
    """Two-level hierarchical clustering with centroid-boosted scoring."""

    def __init__(
        self,
        n_clusters_l1: int = 4,
        n_clusters_l2: int = 2,
        alpha: float = 0.6,
        beta: float = 0.25,
        gamma: float = 0.15,
    ):
        """
        Args:
            n_clusters_l1: Number of fine-grained (level 1) clusters.
            n_clusters_l2: Number of coarse (level 2) clusters.
            alpha: Weight for direct document similarity.
            beta: Weight for level-1 cluster centroid similarity.
            gamma: Weight for level-2 cluster centroid similarity.
        """
        self.n_clusters_l1 = n_clusters_l1
        self.n_clusters_l2 = n_clusters_l2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @property
    def name(self) -> str:
        return "hierarchical-cluster"

    @property
    def description(self) -> str:
        return (
            "RAPTOR-style hierarchical cluster retrieval — two-level "
            "agglomerative clustering with centroid-boosted scoring"
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
        n_l1 = min(self.n_clusters_l1, n - 1)
        n_l2 = min(self.n_clusters_l2, n_l1)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # Direct document similarities
        doc_sims = E @ q

        # Agglomerative clustering (Ward-like using cosine distance)
        labels_l1 = self._agglomerative(E, n_l1)
        labels_l2 = self._agglomerative(E, n_l2)

        # Compute cluster centroids
        centroids_l1 = self._compute_centroids(E, labels_l1, n_l1)
        centroids_l2 = self._compute_centroids(E, labels_l2, n_l2)

        # Cluster-level similarities to query
        l1_sims = np.array([centroids_l1[labels_l1[i]] @ q for i in range(n)])
        l2_sims = np.array([centroids_l2[labels_l2[i]] @ q for i in range(n)])

        # Combined hierarchical score
        combined = self.alpha * doc_sims + self.beta * l1_sims + self.gamma * l2_sims

        new_order = np.argsort(combined)[::-1].tolist()
        reranked = [results[i] for i in new_order]

        rank_changes = sum(1 for i, idx in enumerate(new_order) if idx != i)

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format_hierarchical(reranked, labels_l1, labels_l2, new_order),
            metadata={
                "n_clusters_l1": n_l1,
                "n_clusters_l2": n_l2,
                "combined_scores": combined[new_order].tolist(),
                "doc_sims": doc_sims[new_order].tolist(),
                "l1_cluster_sims": l1_sims[new_order].tolist(),
                "l2_cluster_sims": l2_sims[new_order].tolist(),
                "rank_changes": rank_changes,
            },
        )

    def _agglomerative(self, E: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple agglomerative clustering using average linkage on cosine distance."""
        n = E.shape[0]
        if n_clusters >= n:
            return np.arange(n)

        # Start with each point in its own cluster
        labels = np.arange(n)
        cluster_members: dict[int, list[int]] = {i: [i] for i in range(n)}
        active_clusters = set(range(n))

        # Pairwise cosine distances
        dist_matrix = 1.0 - E @ E.T
        np.fill_diagonal(dist_matrix, np.inf)

        while len(active_clusters) > n_clusters:
            # Find closest pair of active clusters (average linkage)
            best_dist = np.inf
            best_pair = (-1, -1)

            active_list = sorted(active_clusters)
            for i_idx, ci in enumerate(active_list):
                for cj in active_list[i_idx + 1:]:
                    # Average distance between cluster members
                    members_i = cluster_members[ci]
                    members_j = cluster_members[cj]
                    dist = np.mean([dist_matrix[a, b] for a in members_i for b in members_j])
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (ci, cj)

            ci, cj = best_pair
            # Merge cj into ci
            cluster_members[ci].extend(cluster_members[cj])
            for idx in cluster_members[cj]:
                labels[idx] = ci
            del cluster_members[cj]
            active_clusters.discard(cj)

        # Relabel to 0..n_clusters-1
        unique_labels = sorted(set(labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])

    def _compute_centroids(
        self, E: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> dict[int, np.ndarray]:
        """Compute normalised centroids for each cluster."""
        centroids = {}
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centroid = E[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                centroids[c] = centroid / max(norm, 1e-12)
            else:
                centroids[c] = np.zeros(E.shape[1])
        return centroids

    def _format_hierarchical(
        self,
        results: list[RetrievalResult],
        labels_l1: np.ndarray,
        labels_l2: np.ndarray,
        order: list[int],
    ) -> str:
        lines = ["Retrieved context (hierarchical cluster retrieval):"]
        for rank, (res, orig_idx) in enumerate(zip(results, order)):
            l1 = labels_l1[orig_idx]
            l2 = labels_l2[orig_idx]
            lines.append(f"\n[{rank + 1}] (score: {res.score:.3f}) [topic {l2 + 1}.{l1 + 1}]")
            lines.append(res.text)
        return "\n".join(lines)

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (hierarchical cluster retrieval):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
