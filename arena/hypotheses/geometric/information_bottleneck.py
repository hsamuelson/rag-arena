"""Information Bottleneck reranking (Novel #7).

Compress the retrieved set by finding the minimal subset that preserves
maximum mutual information with the query.  Inspired by the information
bottleneck method (Tishby, Pereira & Bialek, 1999).

Geometric intuition
-------------------
Each retrieved document carries some "information" about the query.
Redundant documents share the same information.  The goal is to greedily
build a subset S where every new document maximises *information gain* —
high similarity to the query, but low redundancy with the already-selected
set.  Unlike MMR (which uses a single lambda trade-off), this formulation
uses an entropy-inspired score:

    IG(d | S) = relevance(d) * (1 - max_redundancy(d, S))

where relevance is cosine similarity to query and redundancy is max cosine
similarity to any already-selected document.  This encourages selection of
documents that reduce uncertainty about the query while avoiding overlap.

The final ordering reflects the sequence of greedy selection, so the most
information-dense documents appear first.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class InformationBottleneckHypothesis(Hypothesis):
    """Greedy information-gain selection to compress the retrieved set."""

    def __init__(self, beta: float = 1.0):
        """
        Args:
            beta: Exponent on the redundancy penalty. Higher beta
                penalises redundancy more aggressively.
        """
        self.beta = beta

    @property
    def name(self) -> str:
        return f"info-bottleneck-{self.beta}b"

    @property
    def description(self) -> str:
        return (
            "Information bottleneck — greedy selection maximising information "
            "gain about the query while minimising redundancy"
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

        # Relevance: cosine similarity to query (shifted to [0, 1])
        relevance = E @ q
        relevance = (relevance + 1.0) / 2.0  # map [-1, 1] -> [0, 1]

        # Pairwise similarity matrix
        sim = E @ E.T
        sim = (sim + 1.0) / 2.0  # map to [0, 1]

        # Greedy sequential selection
        selected: list[int] = []
        remaining = set(range(n))
        info_gains: list[float] = []

        for _ in range(n):
            best_idx = -1
            best_ig = -float("inf")

            for idx in remaining:
                if not selected:
                    redundancy = 0.0
                else:
                    # Max similarity to any already-selected doc
                    redundancy = float(max(sim[idx, s] for s in selected))

                novelty = (1.0 - redundancy) ** self.beta
                ig = float(relevance[idx]) * novelty

                if ig > best_ig:
                    best_ig = ig
                    best_idx = idx

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            info_gains.append(best_ig)

        reranked = [results[i] for i in selected]

        # Compute total information captured
        total_relevance = float(np.sum(relevance))
        selected_relevance = float(sum(relevance[i] for i in selected))

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": selected,
                "info_gains": info_gains,
                "total_relevance": total_relevance,
                "selected_relevance": selected_relevance,
                "compression_ratio": selected_relevance / max(total_relevance, 1e-12),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (information-bottleneck selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
