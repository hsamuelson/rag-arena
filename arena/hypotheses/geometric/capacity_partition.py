"""Capacity-aware partitioning.

Hypothesis: DeepMind (Aug 2025) proved fixed-dimension embeddings have a
hard capacity ceiling. At dim=768, retrieval degrades beyond ~1.7M docs.
Rather than fighting this ceiling, we can work within it by partitioning
the corpus into sub-corpora that stay below the ceiling, routing queries
to the right partition, and merging results.

This is analogous to database sharding but in embedding space.

Algorithm:
1. Cluster the corpus embeddings into partitions (each below ceiling)
2. For each query, identify the top-P partitions by centroid similarity
3. Retrieve top-K/P from each partition
4. Merge via RRF with partition-distance weighting
5. This keeps each partition within the capacity bound

References:
  - DeepMind (2025): On the Theoretical Limitations of Embedding-Based Retrieval
  - IVF indices (FAISS): similar concept at the index level
  - The key insight: making this explicit at the retrieval strategy level
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class CapacityPartitionHypothesis(Hypothesis):
    """Partition retrieved docs to stay within embedding capacity bounds.

    Note: In the arena, we work with the already-retrieved set. This
    hypothesis simulates what a capacity-aware index would do by
    clustering the retrieved candidates and ensuring balanced
    representation from each cluster. At scale, the partitioning
    would happen at index time.
    """

    def __init__(self, partition_size: int = 5, n_partitions: int | None = None):
        self.partition_size = partition_size
        self._n_partitions = n_partitions

    @property
    def name(self) -> str:
        return f"capacity-partition-{self.partition_size}ps"

    @property
    def description(self) -> str:
        return (
            "Capacity-aware partitioning — cluster docs into sub-groups below "
            "the embedding capacity ceiling and retrieve balanced results"
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
        n_parts = self._n_partitions or max(2, n // self.partition_size)
        n_parts = min(n_parts, n)

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        # Cluster into partitions via k-means
        labels = self._kmeans(E, n_parts)

        # Compute partition centroids and their distance to query
        partition_info: dict[int, dict] = {}
        for p in range(n_parts):
            mask = labels == p
            if not mask.any():
                continue
            centroid = E[mask].mean(axis=0)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
            query_sim = float(centroid @ q)
            partition_info[p] = {
                "centroid_query_sim": query_sim,
                "size": int(mask.sum()),
                "indices": np.where(mask)[0].tolist(),
            }

        # Sort partitions by proximity to query
        sorted_partitions = sorted(
            partition_info.keys(),
            key=lambda p: partition_info[p]["centroid_query_sim"],
            reverse=True,
        )

        # Balanced retrieval: take proportional results from each partition
        # More from closer partitions
        selected: list[int] = []
        total_budget = n
        remaining_budget = total_budget

        for rank, p in enumerate(sorted_partitions):
            info = partition_info[p]
            indices = info["indices"]

            # Budget: more for closer partitions (exponential decay)
            weight = 1.0 / (rank + 1)
            budget = max(1, int(weight * total_budget / sum(1.0 / (r + 1) for r in range(len(sorted_partitions)))))
            budget = min(budget, len(indices), remaining_budget)

            # Take top-budget from this partition by original score
            partition_scores = [(i, results[i].score) for i in indices]
            partition_scores.sort(key=lambda x: x[1], reverse=True)

            for i, _ in partition_scores[:budget]:
                selected.append(i)
                remaining_budget -= 1

            if remaining_budget <= 0:
                break

        # Fill any remaining budget
        for i in range(n):
            if i not in set(selected) and remaining_budget > 0:
                selected.append(i)
                remaining_budget -= 1

        reranked = [results[i] for i in selected]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "n_partitions": len(partition_info),
                "partition_sizes": {
                    int(p): info["size"] for p, info in partition_info.items()
                },
                "partition_query_sims": {
                    int(p): round(info["centroid_query_sim"], 4)
                    for p, info in partition_info.items()
                },
                "selected_order": selected,
            },
        )

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
        n = X.shape[0]
        k = min(k, n)
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

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (capacity-partitioned):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
