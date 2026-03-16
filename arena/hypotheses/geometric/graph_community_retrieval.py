"""Graph community retrieval (GraphRAG-style).

Hypothesis: Retrieved documents form an implicit graph where edges represent
semantic similarity. Community structure in this graph reveals coherent
information clusters. By detecting communities and selecting the best
representative from each, we get broader topic coverage. Remaining slots
are filled from the most relevant community for depth.

This simulates GraphRAG's community-level retrieval without requiring a
pre-built knowledge graph — we construct a k-NN similarity graph on the
fly from retrieved embeddings.

Algorithm:
1. Build a k-NN graph from cosine similarities (k = sqrt(n))
2. Compute the normalised graph Laplacian
3. Detect communities via spectral embedding + k-means
4. Select the best document (by query similarity) from each community
5. Fill remaining slots from the top-scoring community

Geometric intuition: The k-NN graph captures local manifold structure
that global pairwise similarity misses. Communities in this graph
correspond to dense regions of the embedding manifold — semantically
coherent document groups that may discuss different facets of the query.

References:
  - Edge et al. (2024): GraphRAG — graph-based retrieval-augmented generation
  - Von Luxburg (2007): A tutorial on spectral clustering
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class GraphCommunityHypothesis(Hypothesis):
    """k-NN graph community detection for diverse, structured retrieval."""

    def __init__(self, max_communities: int = 4):
        """
        Args:
            max_communities: Maximum number of communities to detect.
        """
        self.max_communities = max_communities

    @property
    def name(self) -> str:
        return "graph-community"

    @property
    def description(self) -> str:
        return (
            "GraphRAG-style community retrieval — k-NN graph with spectral "
            "community detection for structured, diverse retrieval"
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

        # Normalise embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)

        # Document-query similarities
        doc_sims = E @ q

        # Build k-NN graph
        knn = max(2, int(np.sqrt(n)))
        S = E @ E.T
        np.fill_diagonal(S, -np.inf)

        # Adjacency matrix: keep only top-k neighbours (symmetric)
        A = np.zeros((n, n))
        for i in range(n):
            top_k_idx = np.argsort(S[i])[-knn:]
            for j in top_k_idx:
                sim_val = max(S[i, j], 0.0)
                A[i, j] = sim_val
                A[j, i] = sim_val

        # Normalised Laplacian
        degrees = A.sum(axis=1)
        degrees = np.maximum(degrees, 1e-12)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # Eigendecomposition for community detection
        eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

        # Determine number of communities via eigengap
        n_comm = self._find_n_communities(eigenvalues)

        # Spectral embedding: use first n_comm eigenvectors (skip first if ~0)
        spec_features = eigenvectors[:, :n_comm]
        row_norms = np.linalg.norm(spec_features, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-12)
        spec_features = spec_features / row_norms

        # k-means clustering in spectral space
        labels = self._kmeans(spec_features, n_comm)

        # Organise documents by community
        communities: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            communities.setdefault(label, []).append(i)

        # Sort communities by their best document's query similarity
        community_best_sim = {
            c: max(doc_sims[i] for i in members)
            for c, members in communities.items()
        }
        sorted_communities = sorted(
            community_best_sim.keys(),
            key=lambda c: community_best_sim[c],
            reverse=True,
        )

        # Selection strategy:
        # Phase 1: best doc from each community
        selected: list[int] = []
        for c in sorted_communities:
            members = communities[c]
            best_member = max(members, key=lambda i: doc_sims[i])
            selected.append(best_member)

        # Phase 2: fill remaining from the best community
        best_community = sorted_communities[0]
        remaining_from_best = sorted(
            [i for i in communities[best_community] if i not in selected],
            key=lambda i: doc_sims[i],
            reverse=True,
        )
        selected.extend(remaining_from_best)

        # Phase 3: fill any remaining slots from other communities
        already_selected = set(selected)
        for c in sorted_communities[1:]:
            for i in sorted(communities[c], key=lambda i: doc_sims[i], reverse=True):
                if i not in already_selected:
                    selected.append(i)
                    already_selected.add(i)

        reranked = [results[i] for i in selected]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format_communities(reranked, labels, selected, n_comm),
            metadata={
                "n_communities": n_comm,
                "community_sizes": {int(c): len(m) for c, m in communities.items()},
                "community_best_sims": {
                    int(c): float(s) for c, s in community_best_sim.items()
                },
                "knn_k": knn,
                "graph_density": float(np.count_nonzero(A) / (n * n)),
                "eigenvalues_top5": eigenvalues[:min(5, n)].tolist(),
            },
        )

    def _find_n_communities(self, eigenvalues: np.ndarray) -> int:
        """Eigengap heuristic for number of communities."""
        max_k = min(self.max_communities, len(eigenvalues) - 1)
        if max_k < 2:
            return 2

        gaps = np.diff(eigenvalues[1:max_k + 2])
        if len(gaps) == 0:
            return 2

        k = int(np.argmax(gaps)) + 2
        return max(2, min(k, max_k))

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        """Simple k-means (no external deps)."""
        n = X.shape[0]
        rng = np.random.RandomState(42)

        # k-means++ init
        centres = [X[rng.randint(n)]]
        for _ in range(1, k):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centres], axis=0)
            probs = dists / (dists.sum() + 1e-12)
            centres.append(X[rng.choice(n, p=probs)])
        centres = np.array(centres)

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centres])
            new_labels = np.argmin(dists, axis=0)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centres[j] = X[mask].mean(axis=0)

        return labels

    def _format_communities(
        self,
        results: list[RetrievalResult],
        all_labels: np.ndarray,
        selected: list[int],
        n_communities: int,
    ) -> str:
        lines = [f"Retrieved context ({n_communities} graph communities):"]
        current_community = -1
        for rank, (res, orig_idx) in enumerate(zip(results, selected)):
            label = all_labels[orig_idx]
            if label != current_community:
                current_community = label
                lines.append(f"\n-- Community {label + 1} --")
            lines.append(f"  [{rank + 1}] (score: {res.score:.3f})")
            lines.append(f"  {res.text}")
        return "\n".join(lines)

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (graph community retrieval):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
