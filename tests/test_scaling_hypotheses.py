"""Tests for scaling-focused hypothesis implementations."""

import numpy as np
import pytest

from arena.backends.base import RetrievalResult
from arena.hypotheses.anti_hubness import AntiHubnessHypothesis
from arena.hypotheses.mean_bias_correction import MeanBiasCorrectionHypothesis
from arena.hypotheses.capacity_partition import CapacityPartitionHypothesis
from arena.hypotheses.score_calibration import ScoreCalibrationHypothesis


def _make_results(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(doc_id=f"doc_{i}", text=f"Document {i} content", score=1.0 - i * 0.05)
        for i in range(n)
    ]


def _make_query_embedding(d: int = 16) -> np.ndarray:
    rng = np.random.RandomState(99)
    q = rng.randn(d).astype(np.float32)
    return q / np.linalg.norm(q)


def _make_biased_embeddings(n: int, d: int = 16) -> np.ndarray:
    """Create embeddings with strong mean bias (anisotropy)."""
    rng = np.random.RandomState(42)
    bias = np.ones(d) * 2.0  # strong shared direction
    return (rng.randn(n, d) * 0.5 + bias).astype(np.float32)


def _make_hub_embeddings(n: int, d: int = 16) -> np.ndarray:
    """Create embeddings where doc 0 is a hub (close to everything)."""
    rng = np.random.RandomState(42)
    embs = rng.randn(n, d).astype(np.float32)
    # Make doc 0 the mean of all others (guaranteed hub)
    embs[0] = embs[1:].mean(axis=0)
    return embs


# ── Anti-Hubness ──────────────────────────────────────────────


class TestAntiHubness:
    def test_detects_hubs(self):
        hyp = AntiHubnessHypothesis(hubness_penalty=0.4, reference_k=3)
        results = _make_results(10)
        embeddings = _make_hub_embeddings(10)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "hubness_counts" in out.metadata
        # Doc 0 should have highest hubness
        assert out.metadata["hubness_counts"][0] >= out.metadata["hubness_mean"]

    def test_computes_csls(self):
        hyp = AntiHubnessHypothesis()
        results = _make_results(8)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(8, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "csls_scores" in out.metadata
        assert len(out.metadata["csls_scores"]) == 8

    def test_fallback(self):
        hyp = AntiHubnessHypothesis()
        out = hyp.apply("test", _make_results(3), None, None)
        assert out.metadata.get("fallback") is True


# ── Mean Bias Correction ─────────────────────────────────────


class TestMeanBiasCorrection:
    def test_changes_ranking_with_biased_embeddings(self):
        hyp = MeanBiasCorrectionHypothesis()
        results = _make_results(10)
        embeddings = _make_biased_embeddings(10)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert out.metadata.get("rank_changes", 0) > 0

    def test_reports_bias_magnitude(self):
        hyp = MeanBiasCorrectionHypothesis()
        results = _make_results(8)
        embeddings = _make_biased_embeddings(8)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert out.metadata["mean_bias_norm"] > 0

    def test_preserves_result_count(self):
        hyp = MeanBiasCorrectionHypothesis()
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert len(out.results) == 10

    def test_fallback(self):
        hyp = MeanBiasCorrectionHypothesis()
        out = hyp.apply("test", _make_results(5), None, None)
        assert out.metadata.get("fallback") is True


# ── Capacity Partition ────────────────────────────────────────


class TestCapacityPartition:
    def test_creates_partitions(self):
        hyp = CapacityPartitionHypothesis(partition_size=3)
        results = _make_results(12)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(12, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "n_partitions" in out.metadata
        assert out.metadata["n_partitions"] >= 2

    def test_balanced_representation(self):
        hyp = CapacityPartitionHypothesis(partition_size=5, n_partitions=3)
        results = _make_results(15)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(15, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert len(out.results) == 15

    def test_reports_partition_query_similarity(self):
        hyp = CapacityPartitionHypothesis(n_partitions=3)
        results = _make_results(9)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(9, 8).astype(np.float32)
        query_emb = _make_query_embedding(8)
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "partition_query_sims" in out.metadata

    def test_fallback(self):
        hyp = CapacityPartitionHypothesis()
        out = hyp.apply("test", _make_results(3), None, None)
        assert out.metadata.get("fallback") is True


# ── Score Calibration ─────────────────────────────────────────


class TestScoreCalibration:
    def test_calibrates_scores(self):
        hyp = ScoreCalibrationHypothesis(k_neighbors=3)
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "z_scores" in out.metadata
        assert "calibrated_scores" in out.metadata
        assert "local_densities" in out.metadata

    def test_suggests_cutoff(self):
        hyp = ScoreCalibrationHypothesis()
        results = _make_results(10)
        rng = np.random.RandomState(42)
        embeddings = rng.randn(10, 16).astype(np.float32)
        query_emb = _make_query_embedding()
        out = hyp.apply("test", results, embeddings, query_emb)
        assert "suggested_cutoff" in out.metadata
        assert 0 <= out.metadata["suggested_cutoff"] <= 10

    def test_dense_vs_sparse_calibration(self):
        """Dense clusters should have different calibration than sparse."""
        hyp = ScoreCalibrationHypothesis(k_neighbors=3)
        results = _make_results(10)

        # Dense cluster
        rng = np.random.RandomState(42)
        dense = rng.randn(10, 16).astype(np.float32) * 0.1
        query_emb = _make_query_embedding()
        out_dense = hyp.apply("test", results, dense, query_emb)

        # Sparse spread
        sparse = rng.randn(10, 16).astype(np.float32) * 5.0
        out_sparse = hyp.apply("test", results, sparse, query_emb)

        # Local densities should differ
        assert out_dense.metadata["local_densities"] != out_sparse.metadata["local_densities"]
