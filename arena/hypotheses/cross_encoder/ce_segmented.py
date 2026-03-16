"""Segmented cross-encoder hypothesis.

Instead of truncating long documents to ~512 tokens, split them into
overlapping segments and score each segment independently. Use the
maximum segment score as the document score.

Rationale: Cross-encoders are limited to ~512 tokens. Long documents
may have the relevant passage buried past the truncation point.
Segmentation ensures every part of the document gets scored.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_SEGMENT_CHARS = 1500  # ~375 tokens, leaves room for query
_OVERLAP_CHARS = 300   # overlap between segments


class CESegmentedHypothesis(Hypothesis):
    """Cross-encoder with document segmentation — max-score aggregation."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        segment_chars: int = _SEGMENT_CHARS,
        overlap_chars: int = _OVERLAP_CHARS,
        aggregation: str = "max",  # "max" or "mean"
    ):
        self._model_name = model_name
        self._segment_chars = segment_chars
        self._overlap_chars = overlap_chars
        self._aggregation = aggregation
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def _segment(self, text):
        """Split text into overlapping segments."""
        if len(text) <= self._segment_chars:
            return [text]
        segments = []
        start = 0
        step = self._segment_chars - self._overlap_chars
        while start < len(text):
            end = min(start + self._segment_chars, len(text))
            segments.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return segments

    @property
    def name(self) -> str:
        return f"ce-segmented-{self._aggregation}"

    @property
    def description(self) -> str:
        return f"Cross-encoder with document segmentation ({self._aggregation}-score aggregation)"

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="Retrieved context:\n(no results)", metadata={})

        model = self._get_model()

        # Build all query-segment pairs
        all_pairs = []
        doc_segment_map = []  # (doc_idx, segment_idx) for each pair
        for doc_idx, r in enumerate(results):
            segments = self._segment(r.text)
            for seg_idx, seg in enumerate(segments):
                all_pairs.append((query, seg))
                doc_segment_map.append((doc_idx, seg_idx))

        # Score all segments in one batch
        all_scores = model.predict(all_pairs).tolist()

        # Aggregate per document
        doc_scores = {}
        doc_segment_scores = {}
        for pair_idx, (doc_idx, seg_idx) in enumerate(doc_segment_map):
            if doc_idx not in doc_segment_scores:
                doc_segment_scores[doc_idx] = []
            doc_segment_scores[doc_idx].append(all_scores[pair_idx])

        for doc_idx, seg_scores in doc_segment_scores.items():
            if self._aggregation == "max":
                doc_scores[doc_idx] = max(seg_scores)
            else:
                doc_scores[doc_idx] = sum(seg_scores) / len(seg_scores)

        # Rank by aggregated score
        ranked_indices = sorted(doc_scores.keys(), key=lambda i: doc_scores[i], reverse=True)
        reranked = [results[i] for i in ranked_indices]

        lines = [f"Retrieved context (segmented CE, {self._aggregation}):"]
        for i, idx in enumerate(ranked_indices, 1):
            n_segs = len(doc_segment_scores[idx])
            lines.append(f"\n[{i}] (score: {doc_scores[idx]:.4f}, segments: {n_segs})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "doc_scores": [doc_scores[i] for i in ranked_indices],
                "segment_counts": [len(doc_segment_scores[i]) for i in ranked_indices],
                "total_segments_scored": len(all_pairs),
            },
        )
