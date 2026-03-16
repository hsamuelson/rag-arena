"""Pure CSLS (Cross-domain Similarity Local Scaling) reranking.

Unlike anti-hubness which blends CSLS with original scores,
this uses CSLS as the sole ranking signal. The theory says
CSLS(q, d) = 2*cos(q,d) - mean_k(cos(d, NN_k(d)))
corrects for neighbourhood density and should be used directly.

Testing whether pure CSLS beats the blended version.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult


class PureCSLSHypothesis(Hypothesis):
    """Rank purely by CSLS score (no blending with original cosine)."""

    def __init__(self, reference_k: int = 5):
        self.reference_k = reference_k

    @property
    def name(self) -> str:
        return f"pure-csls-{self.reference_k}k"

    @property
    def description(self) -> str:
        return "Pure CSLS reranking — use cross-domain similarity local scaling as sole ranking signal"

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
        k = min(self.reference_k, n - 1)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = embeddings / norms

        q_norm = np.linalg.norm(query_embedding)
        q = query_embedding / max(q_norm, 1e-12)

        raw_sims = E @ q

        doc_doc_sims = E @ E.T
        np.fill_diagonal(doc_doc_sims, -1)

        mean_nn_sim = np.zeros(n)
        for i in range(n):
            top_k_sims = np.sort(doc_doc_sims[i])[::-1][:k]
            mean_nn_sim[i] = top_k_sims.mean()

        csls_scores = 2 * raw_sims - mean_nn_sim

        order = np.argsort(csls_scores)[::-1]
        reranked = [results[i] for i in order]

        return HypothesisResult(
            results=reranked,
            context_prompt=self._format(reranked),
            metadata={
                "csls_scores": csls_scores[order].tolist(),
                "raw_sims": raw_sims[order].tolist(),
                "mean_nn_sims": mean_nn_sim[order].tolist(),
                "rerank_order": order.tolist(),
            },
        )

    def _format(self, results: list[RetrievalResult]) -> str:
        lines = ["Retrieved context (pure CSLS):"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
