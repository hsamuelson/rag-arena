"""BM25 Score as Text Feature for Cross-Encoder.

Research basis: "Pathway to Relevance" (Lu et al., EMNLP 2025) shows CEs
internally reconstruct BM25. TREC 2023 DL Track found that prepending
BM25 scores as text (e.g., "[Relevance: 0.72]") improves robustness.

This is a zero-cost inference-time technique: just modify the CE input text.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class BM25TextFeatureCEHypothesis(Hypothesis):
    """Prepend retrieval score as text feature to CE input."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return "bm25-text-feature-ce"

    @property
    def description(self):
        return "Prepend [Relevance: X.XX] to CE input (EMNLP 2025 finding)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()

        # Prepend retrieval score as text feature
        pairs = []
        for r in results:
            score_prefix = f"[Relevance: {r.score:.3f}] "
            pairs.append((query, score_prefix + r.text[:_MAX_CHARS - len(score_prefix)]))

        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        reranked = [results[i] for i in ranked]

        lines = ["Retrieved context (BM25 text feature + CE reranked):"]
        for i, idx in enumerate(ranked, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"ce_scores": [ce_scores[i] for i in ranked]},
        )
