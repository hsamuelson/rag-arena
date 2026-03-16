"""Kernel herding reranking (Novel #13).

From Bayesian quadrature and kernel methods: select documents that minimise
the Maximum Mean Discrepancy (MMD) between the selected subset and a target
distribution centred on the query.

Geometric intuition
-------------------
The MMD measures the distance between two distributions in a reproducing
kernel Hilbert space (RKHS).  Kernel herding iteratively selects the point
(document) that most reduces the MMD between the empirical distribution of
selected documents and the target (query neighbourhood).  This produces a
well-distributed subset that optimally approximates the query's information
neighbourhood — not just the closest points, but points that collectively
"cover" the query region in RKHS.

The herding sequence converges to the target distribution at rate O(1/T),
which is faster than i.i.d. sampling (O(1/sqrt(T))).

Algorithm
---------
1. Define target mean embedding: mu_target = k(query, .) — the kernel
   evaluation centred on the query.
2. Initialise running mean mu_S = 0.
3. At each step, pick the document d that maximises:
   k(query, d) - (1/|S|) sum_{s in S} k(s, d)
   This is the document whose RKHS direction most reduces the gap
   between mu_S and mu_target.
4. Update mu_S and repeat.

Uses an RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2 sigma^2)).

References:
  - Chen, Welling & Smola (2010): Super-Samples from Kernel Herding
  - Bach, Lacoste-Julien & Obozinski (2012): Herding and Quadrature
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class KernelHerdingHypothesis(Hypothesis):
    """Select documents via kernel herding to minimise MMD to query."""

    def __init__(self, sigma: float | None = None, query_weight: float = 0.7):
        """
        Args:
            sigma: Bandwidth of the RBF kernel.  None = adaptive (median
                heuristic on pairwise distances).
            query_weight: Weight of the query attraction term vs the
                repulsion from already-selected docs.
        """
        self._sigma = sigma
        self.query_weight = query_weight

    @property
    def name(self) -> str:
        sigma_str = f"{self._sigma}" if self._sigma else "auto"
        return f"kernel-herding-{sigma_str}s"

    @property
    def description(self) -> str:
        return (
            "Kernel herding — iteratively select documents minimising MMD "
            "to the query distribution in RKHS (RBF kernel)"
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
        d = embeddings.shape[1]

        # Normalise embeddings
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
        q = (query_embedding / qnorm).reshape(1, -1)

        # Compute sigma via median heuristic if not specified
        if self._sigma is not None:
            sigma = self._sigma
        else:
            # Pairwise squared distances
            sq_dists = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=2)
            upper = sq_dists[np.triu_indices(n, k=1)]
            if len(upper) > 0:
                sigma = float(np.sqrt(np.median(upper) / 2.0))
            else:
                sigma = 1.0
            sigma = max(sigma, 1e-6)

        two_sigma_sq = 2.0 * sigma * sigma

        # Precompute kernel values
        # k(q, d_i) for all docs
        sq_dist_q = np.sum((E - q) ** 2, axis=1)  # (n,)
        k_query = np.exp(-sq_dist_q / two_sigma_sq)  # (n,)

        # k(d_i, d_j) for all pairs
        sq_dist_dd = np.sum((E[:, None, :] - E[None, :, :]) ** 2, axis=2)
        K = np.exp(-sq_dist_dd / two_sigma_sq)  # (n, n)

        # Kernel herding: greedy selection
        selected: list[int] = []
        remaining = set(range(n))
        # Running sum of k(selected_docs, d_i)
        k_sum = np.zeros(n)
        mmd_reductions: list[float] = []

        w = self.query_weight

        for step in range(n):
            best_idx = -1
            best_score = -float("inf")

            t = step + 1  # number of docs after this selection

            for idx in remaining:
                # Herding criterion:
                # Pick d that maximises: w * k(q, d) - (1/t) * sum_{s in S} k(s, d)
                # (attraction to query minus repulsion from selected set)
                attraction = w * k_query[idx]
                if step > 0:
                    repulsion = k_sum[idx] / step
                else:
                    repulsion = 0.0

                score = attraction - repulsion

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            mmd_reductions.append(best_score)

            # Update running kernel sum
            k_sum += K[best_idx]

        # Append any remaining (shouldn't happen with well-behaved data)
        for i in range(n):
            if i not in set(selected):
                selected.append(i)

        reranked = [results[i] for i in selected]

        # Estimate final MMD^2
        if len(selected) > 0:
            sel = np.array(selected[:min(len(selected), n)])
            mu_S = np.mean(K[np.ix_(sel, sel)])
            mu_cross = np.mean(k_query[sel])
            mmd_sq = float(mu_S - 2 * mu_cross + 1.0)  # k(q,q) = 1
        else:
            mmd_sq = float("inf")

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "selected_indices": selected,
                "sigma": sigma,
                "mmd_squared": mmd_sq,
                "herding_scores": mmd_reductions,
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (kernel-herding selection):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
