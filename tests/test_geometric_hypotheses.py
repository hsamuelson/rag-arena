"""Tests for geometry-based hypothesis implementations."""

import numpy as np
import pytest

from arena.backends.base import RetrievalResult
from arena.hypotheses.dpp_selection import DPPSelectionHypothesis
from arena.hypotheses.convex_hull_coverage import ConvexHullCoverageHypothesis
from arena.hypotheses.centroid_drift import CentroidDriftHypothesis
from arena.hypotheses.spectral_gap import SpectralGapHypothesis
from arena.hypotheses.local_intrinsic_dimension import LocalIntrinsicDimensionHypothesis
from arena.hypotheses.cone_retrieval import ConeRetrievalHypothesis
from arena.hypotheses.geodesic_interpolation import GeodesicInterpolationHypothesis


def _make_results(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(doc_id=f"doc_{i}", text=f"Document {i} content", score=1.0 - i * 0.05)
        for i in range(n)
    ]


def _make_clustered_embeddings(n_per_cluster: int = 5, n_clusters: int = 3, d: int = 16) -> np.ndarray:
    """Create embeddings with clear cluster structure."""
    rng = np.random.RandomState(42)
    embeddings = []
    for c in range(n_clusters):
        centre = rng.randn(d)
        centre = centre / np.linalg.norm(centre) * 3.0
        for _ in range(n_per_cluster):
            point = centre + rng.randn(d) * 0.2
            embeddings.append(point)
    return np.array(embeddings, dtype=np.float32)


def _make_query_embedding(d: int = 16) -> np.ndarray:
    rng = np.random.RandomState(99)
    q = rng.randn(d).astype(np.float32)
    return q / np.linalg.norm(q)


# ── DPP Selection ─────────────────────────────────────────────


class TestDPPSelection:
    def test_selects_diverse_subset(self):
        hyp = DPPSelectionHypothesis(relevance_weight=0.7)
        results = _make_results(15)
        embeddings = _make_clustered_embeddings(5, 3)
        out = hyp.apply("test", results, embeddings, None)
        assert len(out.results) == 15
        assert "selected_indices" in out.metadata

    def test_kernel_logdet_is_finite(self):
        hyp = DPPSelectionHypothesis()
        results = _make_results(10)
        embeddings = _make_clustered_embeddings(5, 2, d=8)
        out = hyp.apply("test", results, embeddings, None)
        assert np.isfinite(out.metadata["kernel_logdet"])

    def test_fallback_without_embeddings(self):
        hyp = DPPSelectionHypothesis()
        results = _make_results(5)
        out = hyp.apply("test", results, None, None)
        assert out.metadata.get("fallback") is True

    def test_respects_subset_size(self):
        hyp = DPPSelectionHypothesis(subset_size=5)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 8).astype(np.float32)
        out = hyp.apply("test", results, embeddings, None)
        assert len(out.results) <= 5


# ── Convex Hull Coverage ──────────────────────────────────────


class TestConvexHullCoverage:
    def test_produces_coverage_metric(self):
        hyp = ConvexHullCoverageHypothesis(n_components=3)
        results = _make_results(10)
        embeddings = _make_clustered_embeddings(5, 2, d=16)
        query_emb = _make_query_embedding(16)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "angular_coverage" in out.metadata
        assert 0 <= out.metadata["angular_coverage"] <= 1.0

    def test_fallback_without_query(self):
        hyp = ConvexHullCoverageHypothesis()
        results = _make_results(5)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(5, 8).astype(np.float32)
        out = hyp.apply("test", results, embeddings, None)
        assert out.metadata.get("fallback") is True

    def test_reports_hull_containment(self):
        hyp = ConvexHullCoverageHypothesis(n_components=3)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        query_emb = _make_query_embedding(16)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "query_inside_hull" in out.metadata


# ── Centroid Drift ────────────────────────────────────────────


class TestCentroidDrift:
    def test_reduces_drift(self):
        hyp = CentroidDriftHypothesis(max_swaps=10)
        results = _make_results(10)
        embeddings = _make_clustered_embeddings(5, 2, d=8)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert out.metadata["final_drift"] <= out.metadata["initial_drift"] + 1e-6

    def test_keeps_minimum_docs(self):
        hyp = CentroidDriftHypothesis(max_swaps=50)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 8).astype(np.float32)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert out.metadata["docs_kept"] >= 3

    def test_reports_drift_reduction(self):
        hyp = CentroidDriftHypothesis()
        results = _make_results(8)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(8, 8).astype(np.float32)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "drift_reduction" in out.metadata


# ── Spectral Gap ──────────────────────────────────────────────


class TestSpectralGap:
    def test_detects_clusters(self):
        hyp = SpectralGapHypothesis(max_clusters=4)
        results = _make_results(15)
        embeddings = _make_clustered_embeddings(5, 3, d=16)
        out = hyp.apply("test", results, embeddings, None)
        assert 2 <= out.metadata["n_clusters"] <= 4

    def test_round_robin_from_clusters(self):
        hyp = SpectralGapHypothesis()
        results = _make_results(15)
        embeddings = _make_clustered_embeddings(5, 3, d=16)
        out = hyp.apply("test", results, embeddings, None)
        # Should have results from multiple clusters
        assert len(out.results) == 15

    def test_reports_eigenvalues(self):
        hyp = SpectralGapHypothesis()
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 8).astype(np.float32)
        out = hyp.apply("test", results, embeddings, None)
        assert "eigenvalues" in out.metadata
        assert "eigengaps" in out.metadata

    def test_clustered_presentation(self):
        hyp = SpectralGapHypothesis()
        results = _make_results(10)
        embeddings = _make_clustered_embeddings(5, 2, d=8)
        out = hyp.apply("test", results, embeddings, None)
        assert "Cluster" in out.context_prompt


# ── Local Intrinsic Dimensionality ────────────────────────────


class TestLID:
    def test_computes_lid_values(self):
        hyp = LocalIntrinsicDimensionHypothesis(k_neighbors=3)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        out = hyp.apply("test", results, embeddings, None)
        assert "lid_values" in out.metadata
        assert len(out.metadata["lid_values"]) == 10

    def test_lid_values_are_positive(self):
        hyp = LocalIntrinsicDimensionHypothesis(k_neighbors=4)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        out = hyp.apply("test", results, embeddings, None)
        for lid in out.metadata["lid_values"]:
            assert lid >= 0

    def test_reranks_results(self):
        hyp = LocalIntrinsicDimensionHypothesis(lid_penalty=0.5)
        results = _make_results(10)
        embeddings = _make_clustered_embeddings(5, 2, d=8)
        out = hyp.apply("test", results, embeddings, None)
        assert len(out.results) == 10

    def test_clustered_vs_uniform_lid(self):
        """Clustered data should have different LID than uniform."""
        hyp = LocalIntrinsicDimensionHypothesis(k_neighbors=3)
        results = _make_results(15)

        clustered = _make_clustered_embeddings(5, 3, d=16)
        out_c = hyp.apply("test", results, clustered, None)

        rng = np.random.RandomState(42)
        uniform = rng.randn(15, 16).astype(np.float32)
        out_u = hyp.apply("test", results, uniform, None)

        # LID distributions should differ
        assert out_c.metadata["lid_mean"] != out_u.metadata["lid_mean"]


# ── Cone Retrieval ────────────────────────────────────────────


class TestConeRetrieval:
    def test_adapts_cutoff(self):
        hyp = ConeRetrievalHypothesis()
        # Create results with a clear score gap
        results = [
            RetrievalResult(doc_id=f"doc_{i}", text=f"Doc {i}", score=s)
            for i, s in enumerate([0.95, 0.93, 0.90, 0.88, 0.50, 0.48, 0.45, 0.43])
        ]
        out = hyp.apply("test", results, None, None)
        # Should cut off around the gap at index 4
        assert out.metadata.get("cone_cutoff", 8) <= 6

    def test_keeps_minimum_results(self):
        hyp = ConeRetrievalHypothesis(min_results=3)
        results = _make_results(5)
        out = hyp.apply("test", results, None, None)
        assert len(out.results) >= 3

    def test_angular_analysis_with_embeddings(self):
        hyp = ConeRetrievalHypothesis()
        results = _make_results(8)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(8, 16).astype(np.float32)
        query_emb = _make_query_embedding(16)
        out = hyp.apply("test", results, embeddings, query_emb)
        if "cone_half_angle_deg" in out.metadata:
            assert out.metadata["cone_half_angle_deg"] > 0


# ── Geodesic Interpolation ───────────────────────────────────


class TestGeodesicInterpolation:
    def test_produces_path(self):
        hyp = GeodesicInterpolationHypothesis(n_waypoints=3)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        query_emb = _make_query_embedding(16)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "geodesic_path_length" in out.metadata
        assert out.metadata["geodesic_path_length"] > 0

    def test_slerp_interpolation(self):
        """Verify slerp produces unit vectors between endpoints."""
        hyp = GeodesicInterpolationHypothesis()
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([0.0, 1.0, 0.0])
        mid = hyp._slerp(v0, v1, 0.5)
        # Should be unit vector at 45 degrees
        assert abs(np.linalg.norm(mid) - 1.0) < 1e-6
        assert abs(mid @ v0 - mid @ v1) < 1e-6  # equidistant

    def test_reports_query_dest_angle(self):
        hyp = GeodesicInterpolationHypothesis()
        results = _make_results(8)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(8, 8).astype(np.float32)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "query_dest_angle_deg" in out.metadata
        assert 0 <= out.metadata["query_dest_angle_deg"] <= 180

    def test_all_results_included(self):
        hyp = GeodesicInterpolationHypothesis(n_waypoints=3)
        results = _make_results(8)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(8, 8).astype(np.float32)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert len(out.results) == 8
        # All original doc_ids should be present
        original_ids = {r.doc_id for r in results}
        returned_ids = {r.doc_id for r in out.results}
        assert original_ids == returned_ids
