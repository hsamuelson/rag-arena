"""Mutual information reranking via maximum spanning tree.

Hypothesis (Novel #19): Documents are not independent — they form an
information graph where edges represent shared semantic content. The most
valuable documents for answering a query are those that sit at the backbone
of this information structure: hub nodes that connect many topics.

Geometric intuition: Estimate pairwise mutual information between documents
using embedding correlations as a proxy (high correlation in embedding space
implies shared information content). Build a weighted graph and find its
maximum weight spanning tree (MST). The MST reveals the skeleton of the
information landscape. Documents with high degree in the MST are information
hubs — they connect to many other topics and provide the most efficient
coverage of the semantic space. Selecting these hub nodes gives the LLM
a maximally informative context window.

Algorithm:
1. Compute pairwise cosine similarities between all retrieved document
   embeddings, treating these as MI proxies.
2. Build a complete weighted graph.
3. Find the maximum weight spanning tree (Kruskal/Prim).
4. Compute node degree in the MST.
5. Score documents by MST degree (information hub score) combined with
   query similarity, preferring high-degree nodes.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class MutualInformationHypothesis(Hypothesis):
    """Rerank by MST-degree centrality in the document MI graph."""

    def __init__(self, hub_weight: float = 0.4):
        self.hub_weight = hub_weight

    @property
    def name(self) -> str:
        return "mutual-information"

    @property
    def description(self) -> str:
        return (
            "Mutual information reranking — find information hubs via "
            "maximum spanning tree of the document similarity graph"
        )

    @staticmethod
    def _maximum_spanning_tree(n: int, weights: np.ndarray) -> list[tuple[int, int]]:
        """Kruskal's algorithm for maximum weight spanning tree.

        Args:
            n: Number of nodes.
            weights: (n, n) symmetric weight matrix.

        Returns:
            List of edges (i, j) in the MST.
        """
        # Collect all edges with weights
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((weights[i, j], i, j))

        # Sort by weight descending (maximum spanning tree)
        edges.sort(key=lambda x: x[0], reverse=True)

        # Union-Find
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        mst_edges = []
        for w, i, j in edges:
            if union(i, j):
                mst_edges.append((i, j))
                if len(mst_edges) == n - 1:
                    break

        return mst_edges

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if embeddings is None or query_embedding is None or len(results) < 3:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )

        n = len(results)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        q = query_embedding / q_norm

        # Pairwise cosine similarities as MI proxy
        sim_matrix = E @ E.T
        np.fill_diagonal(sim_matrix, 0.0)

        # Shift to positive weights (MI is non-negative)
        # Use (sim + 1) / 2 to map [-1, 1] -> [0, 1]
        weights = (sim_matrix + 1.0) / 2.0

        # Find maximum spanning tree
        mst_edges = self._maximum_spanning_tree(n, weights)

        # Compute MST degree for each node
        degrees = np.zeros(n, dtype=int)
        edge_weights_sum = np.zeros(n)
        for i, j in mst_edges:
            degrees[i] += 1
            degrees[j] += 1
            edge_weights_sum[i] += weights[i, j]
            edge_weights_sum[j] += weights[i, j]

        # Normalise degrees to [0, 1]
        deg_max = degrees.max()
        if deg_max > 0:
            norm_degrees = degrees.astype(float) / deg_max
        else:
            norm_degrees = np.zeros(n)

        # Query-document similarities
        raw_sims = E @ q

        # Normalise scores to [0, 1]
        s_min, s_max = raw_sims.min(), raw_sims.max()
        if s_max - s_min > 1e-12:
            norm_sims = (raw_sims - s_min) / (s_max - s_min)
        else:
            norm_sims = np.ones(n) * 0.5

        # Combined score: hub importance + query relevance
        w = self.hub_weight
        combined = w * norm_degrees + (1 - w) * norm_sims

        order = np.argsort(combined)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "mst_edges": mst_edges,
                "mst_degrees": degrees[order].tolist(),
                "hub_scores": norm_degrees[order].tolist(),
                "query_sims": raw_sims[order].tolist(),
                "combined_scores": combined[order].tolist(),
                "total_mst_weight": float(
                    sum(weights[i, j] for i, j in mst_edges)
                ),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (MI-hub ranked):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
