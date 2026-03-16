"""Tests for hypothesis implementations."""

import numpy as np
import pytest

from arena.backends.base import RetrievalResult
from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
from arena.hypotheses.pca_diversity import PCADiversityHypothesis
from arena.hypotheses.pca_grouped import PCAGroupedHypothesis


def _make_results(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(doc_id=f"doc_{i}", text=f"Document {i} content", score=1.0 - i * 0.1)
        for i in range(n)
    ]


def _make_embeddings(n: int, d: int = 8, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, d).astype(np.float32)


class TestFlatBaseline:
    def test_preserves_order(self):
        hyp = FlatBaselineHypothesis()
        results = _make_results(5)
        out = hyp.apply("test query", results, None, None)
        assert [r.doc_id for r in out.results] == [r.doc_id for r in results]

    def test_formats_context(self):
        hyp = FlatBaselineHypothesis()
        results = _make_results(3)
        out = hyp.apply("test query", results, None, None)
        assert "Retrieved context:" in out.context_prompt
        assert "Document 0" in out.context_prompt
        assert "[1]" in out.context_prompt

    def test_name(self):
        assert FlatBaselineHypothesis().name == "flat-baseline"


class TestPCADiversity:
    def test_reranks_with_embeddings(self):
        hyp = PCADiversityHypothesis(n_components=2, diversity_weight=0.5)
        results = _make_results(6)
        embeddings = _make_embeddings(6)
        out = hyp.apply("test", results, embeddings, None)
        assert len(out.results) == 6
        assert "pca_axes" in out.metadata

    def test_falls_back_without_embeddings(self):
        hyp = PCADiversityHypothesis()
        results = _make_results(5)
        out = hyp.apply("test", results, None, None)
        assert out.metadata.get("fallback") is True

    def test_falls_back_with_few_results(self):
        hyp = PCADiversityHypothesis()
        results = _make_results(2)
        embeddings = _make_embeddings(2)
        out = hyp.apply("test", results, embeddings, None)
        assert out.metadata.get("fallback") is True

    def test_diversity_changes_order(self):
        """With high diversity weight, order should differ from pure relevance."""
        hyp_diverse = PCADiversityHypothesis(n_components=3, diversity_weight=0.9)
        hyp_relevant = PCADiversityHypothesis(n_components=3, diversity_weight=0.0)

        results = _make_results(10)
        # Create embeddings with clear clusters
        embeddings = np.zeros((10, 8), dtype=np.float32)
        embeddings[0:3, 0:3] = np.eye(3)       # cluster 1
        embeddings[3:6, 3:6] = np.eye(3)       # cluster 2
        embeddings[6:9, 0:3] = np.eye(3) * 0.5 # near cluster 1
        embeddings[9, 5:8] = [1, 1, 1]         # outlier

        out_diverse = hyp_diverse.apply("test", results, embeddings, None)
        out_relevant = hyp_relevant.apply("test", results, embeddings, None)

        diverse_order = [r.doc_id for r in out_diverse.results]
        relevant_order = [r.doc_id for r in out_relevant.results]

        # They should produce different orderings
        assert diverse_order != relevant_order

    def test_pca_axes_structure(self):
        hyp = PCADiversityHypothesis(n_components=2)
        results = _make_results(5)
        embeddings = _make_embeddings(5)
        out = hyp.apply("test", results, embeddings, None)

        axes = out.metadata["pca_axes"]
        assert len(axes) == 2
        for axis in axes:
            assert "explained_variance" in axis
            assert "positive_pole_idx" in axis
            assert "negative_pole_idx" in axis
            assert 0 <= axis["explained_variance"] <= 1.0


class TestPCAGrouped:
    def test_groups_with_embeddings(self):
        hyp = PCAGroupedHypothesis(n_components=2)
        results = _make_results(6)
        embeddings = _make_embeddings(6)
        out = hyp.apply("test", results, embeddings, None)
        assert "Dimension" in out.context_prompt
        assert "variance" in out.context_prompt

    def test_falls_back_without_embeddings(self):
        hyp = PCAGroupedHypothesis()
        results = _make_results(5)
        out = hyp.apply("test", results, None, None)
        assert out.metadata.get("fallback") is True

    def test_group_assignments(self):
        hyp = PCAGroupedHypothesis(n_components=2)
        results = _make_results(6)
        embeddings = _make_embeddings(6)
        out = hyp.apply("test", results, embeddings, None)
        assignments = out.metadata.get("group_assignments", [])
        assert len(assignments) == 6
        # Each assignment should be a valid axis index
        assert all(0 <= a < 2 for a in assignments)
