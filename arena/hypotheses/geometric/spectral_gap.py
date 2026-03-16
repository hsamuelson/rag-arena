"""Spectral gap reranking.

Hypothesis: The eigenvalue spectrum of the retrieved embedding matrix
reveals natural cluster boundaries. Large gaps between consecutive
eigenvalues indicate distinct semantic "topics" in the retrieved set.
By ensuring representation from each spectral cluster, we get more
complete coverage than cosine top-K (which over-represents the
dominant cluster).

Geometric intuition: Eigenvalue λ_k drops sharply at cluster boundaries.
If λ_1 >> λ_2 >> λ_3 ≈ λ_4 ≈ ..., there are ~2 significant clusters.
We should retrieve proportionally from each cluster, not just the
densest one.

Related work:
  - Spectral clustering (Ng, Jordan, Weiss 2001)
  - Eigengap heuristic for number of clusters
  - PCA variance analysis (already in SEMDA — this extends it)
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class SpectralGapHypothesis(Hypothesis):
    """Use eigenvalue gaps to detect clusters and ensure balanced retrieval."""

    def __init__(self, max_clusters: int = 5):
        self.max_clusters = max_clusters

    @property
    def name(self) -> str:
        return f"spectral-gap-{self.max_clusters}c"

    @property
    def description(self) -> str:
        return (
            "Spectral gap clustering — detect natural topic boundaries via "
            "eigenvalue gaps and retrieve proportionally from each cluster"
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
                context_prompt=self._format_flat(results),
                metadata={"fallback": True},
            )

        n = len(results)

        # Build similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        S = E @ E.T  # cosine similarity matrix
        S = np.maximum(S, 0)  # clip negative similarities

        # Laplacian: L = D - S (normalised)
        D = np.diag(S.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.maximum(np.sqrt(S.sum(axis=1)), 1e-12))
        L_norm = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Find number of clusters via eigengap heuristic
        n_clusters = self._eigengap_k(eigenvalues)

        # Spectral clustering: use first k eigenvectors
        features = eigenvectors[:, :n_clusters]
        # Normalise rows
        row_norms = np.linalg.norm(features, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        features = features / row_norms

        # Simple k-means in spectral space
        labels = self._kmeans(features, n_clusters)

        # Proportional selection: take top docs from each cluster
        cluster_docs: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            cluster_docs.setdefault(label, []).append(i)

        # Sort within clusters by relevance
        for label in cluster_docs:
            cluster_docs[label].sort(key=lambda i: results[i].score, reverse=True)

        # Round-robin selection from clusters (largest relevance first)
        selected: list[int] = []
        cluster_keys = sorted(
            cluster_docs.keys(),
            key=lambda k: max(results[i].score for i in cluster_docs[k]),
            reverse=True,
        )
        pointers = {k: 0 for k in cluster_keys}

        while len(selected) < n:
            added = False
            for k in cluster_keys:
                if pointers[k] < len(cluster_docs[k]):
                    selected.append(cluster_docs[k][pointers[k]])
                    pointers[k] += 1
                    added = True
            if not added:
                break

        reranked = [results[i] for i in selected]

        # Format with cluster labels
        context = self._format_clustered(reranked, labels, selected, n_clusters)

        return HypothesisResult(
            results=reranked,
            context_prompt=context,
            metadata={
                "n_clusters": n_clusters,
                "eigenvalues": eigenvalues[:min(10, n)].tolist(),
                "eigengaps": np.diff(eigenvalues[:min(10, n)]).tolist(),
                "cluster_sizes": {int(k): len(v) for k, v in cluster_docs.items()},
            },
        )

    def _eigengap_k(self, eigenvalues: np.ndarray) -> int:
        """Find number of clusters via largest eigengap."""
        max_k = min(self.max_clusters, len(eigenvalues) - 1)
        if max_k < 2:
            return 2

        # Look at gaps between consecutive eigenvalues (skip λ_0 ≈ 0)
        gaps = np.diff(eigenvalues[1 : max_k + 2])
        if len(gaps) == 0:
            return 2

        # Number of clusters = index of largest gap + 2 (1-indexed, skip first)
        k = int(np.argmax(gaps)) + 2
        return max(2, min(k, max_k))

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means clustering (no sklearn dependency)."""
        n = X.shape[0]
        rng = np.random.RandomState(42)

        # k-means++ initialisation
        centres = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centres], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centres.append(X[rng.choice(n, p=probs)])
        centres = np.array(centres)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centres])
            new_labels = np.argmin(dists, axis=0)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # Update
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centres[j] = X[mask].mean(axis=0)

        return labels

    def _format_clustered(
        self,
        results: list[RetrievalResult],
        all_labels: np.ndarray,
        selected: list[int],
        n_clusters: int,
    ) -> str:
        lines = [f"Retrieved context ({n_clusters} semantic clusters detected):"]

        current_cluster = -1
        for rank, (res, orig_idx) in enumerate(zip(results, selected)):
            label = all_labels[orig_idx]
            if label != current_cluster:
                current_cluster = label
                lines.append(f"\n── Cluster {label + 1} ──")
            lines.append(f"  [{rank + 1}] (score: {res.score:.3f})")
            lines.append(f"  {res.text}")

        return "\n".join(lines)

    def _format_flat(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
