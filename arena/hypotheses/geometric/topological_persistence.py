"""Topological persistence reranking (Novel #12).

Use concepts from persistent homology / topological data analysis (TDA)
to score documents by their topological significance in the retrieved set.

Geometric intuition
-------------------
Build a Vietoris-Rips-like filtration by adding documents in order of
decreasing similarity to the query.  As each document is added, it either:

(a) Creates a new connected component (a "birth" event) — this document
    introduces genuinely new information not connected to anything
    already selected.
(b) Merges two existing components (a "death" event for the younger one)
    — this document bridges two previously separate information clusters.

A connected component that persists for many steps (high persistence =
death_time - birth_time) represents a robust, well-separated information
cluster.  Documents that *create* these long-lived components are
topologically significant — they anchor distinct information threads.

Algorithm
---------
1. Sort documents by descending cosine similarity to query.
2. Build a Union-Find structure.
3. Add documents one by one.  When adding doc i, connect it to all
   previously-added docs within a radius threshold.
4. Track births (new components) and deaths (merges).
5. Score each document by the persistence of the component it birthed.
6. Rerank by persistence score (descending), breaking ties by relevance.

References:
  - Edelsbrunner & Harer (2010): Computational Topology
  - Carlsson (2009): Topology and Data
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class _UnionFind:
    """Simple union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Union x and y. Returns True if they were in different sets."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


class TopologicalPersistenceHypothesis(Hypothesis):
    """Rerank documents by topological persistence in a filtration."""

    def __init__(self, radius_percentile: float = 30.0):
        """
        Args:
            radius_percentile: Percentile of pairwise distances to use as
                the connectivity radius.  Lower values produce more
                components (finer topology).
        """
        self.radius_percentile = radius_percentile

    @property
    def name(self) -> str:
        return f"topological-persistence-{self.radius_percentile}r"

    @property
    def description(self) -> str:
        return (
            "Topological persistence — score documents by the lifetime of "
            "connected components they create in a similarity filtration"
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

        # Normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        qnorm = np.linalg.norm(query_embedding)
        if qnorm < 1e-12:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        q = query_embedding / qnorm

        # Pairwise cosine distances
        sim = E @ E.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        dist = np.maximum(dist, 0.0)

        # Connectivity radius
        upper_tri = dist[np.triu_indices(n, k=1)]
        if len(upper_tri) == 0:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results),
                metadata={"fallback": True},
            )
        radius = float(np.percentile(upper_tri, self.radius_percentile))
        radius = max(radius, 1e-8)

        # Query similarity for filtration order
        q_sim = E @ q
        filtration_order = np.argsort(-q_sim).tolist()  # descending similarity

        # Build filtration with Union-Find
        uf = _UnionFind(n)
        birth_time: dict[int, int] = {}   # component_root -> birth step
        doc_birth: dict[int, int] = {}    # doc_index -> step it was added
        persistence: dict[int, float] = {}  # doc_index -> persistence value
        added = set()

        # Track which doc birthed which component
        component_creator: dict[int, int] = {}  # root -> creator doc index

        for step, doc_idx in enumerate(filtration_order):
            added.add(doc_idx)
            doc_birth[doc_idx] = step

            # This doc initially creates its own component
            birth_time[doc_idx] = step
            component_creator[doc_idx] = doc_idx

            # Connect to all previously-added docs within radius
            for prev_idx in added:
                if prev_idx == doc_idx:
                    continue
                if dist[doc_idx, prev_idx] <= radius:
                    root_doc = uf.find(doc_idx)
                    root_prev = uf.find(prev_idx)

                    if root_doc != root_prev:
                        # Merge: younger component dies
                        birth_doc = birth_time.get(root_doc, step)
                        birth_prev = birth_time.get(root_prev, step)

                        if birth_doc >= birth_prev:
                            # doc's component is younger, it dies
                            dying_root = root_doc
                            surviving_root = root_prev
                        else:
                            dying_root = root_prev
                            surviving_root = root_doc

                        dying_birth = birth_time.get(dying_root, step)
                        dying_creator = component_creator.get(dying_root, dying_root)
                        pers = step - dying_birth
                        persistence[dying_creator] = float(pers)

                        uf.union(doc_idx, prev_idx)
                        new_root = uf.find(doc_idx)

                        # Surviving component keeps older birth time
                        birth_time[new_root] = min(birth_doc, birth_prev)
                        if birth_doc < birth_prev:
                            component_creator[new_root] = component_creator.get(
                                root_doc, root_doc
                            )
                        else:
                            component_creator[new_root] = component_creator.get(
                                root_prev, root_prev
                            )

        # Components still alive at the end get persistence = n - birth
        for doc_idx in range(n):
            if doc_idx not in persistence:
                root = uf.find(doc_idx)
                creator = component_creator.get(root, root)
                if creator == doc_idx:
                    birth = birth_time.get(root, doc_birth.get(doc_idx, 0))
                    persistence[doc_idx] = float(n - birth)

        # Fill missing with zero
        for i in range(n):
            if i not in persistence:
                persistence[i] = 0.0

        # Combine persistence with relevance for final scoring
        pers_arr = np.array([persistence[i] for i in range(n)])
        pers_max = pers_arr.max()
        if pers_max > 1e-12:
            pers_norm = pers_arr / pers_max
        else:
            pers_norm = np.zeros(n)

        rel = np.array([r.score for r in results], dtype=np.float64)
        r_min, r_max = rel.min(), rel.max()
        if r_max - r_min > 1e-12:
            rel_norm = (rel - r_min) / (r_max - r_min)
        else:
            rel_norm = np.ones(n)

        # Combined score: persistence * relevance
        combined = 0.5 * pers_norm + 0.5 * rel_norm
        order = np.argsort(-combined).tolist()

        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": order,
                "persistence_scores": {i: persistence[i] for i in range(n)},
                "radius": radius,
                "n_final_components": len(set(uf.find(i) for i in range(n))),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (topological-persistence reranking):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
