"""Tests for scoring metrics."""

from arena.metrics.scoring import (
    exact_match,
    token_f1,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mrr,
    compute_scorecard,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_case_insensitive(self):
        assert exact_match("Hello World", "hello world") == 1.0

    def test_strips_articles(self):
        assert exact_match("the cat", "cat") == 1.0

    def test_strips_punctuation(self):
        assert exact_match("hello, world!", "hello world") == 1.0

    def test_different(self):
        assert exact_match("hello", "goodbye") == 0.0

    def test_empty(self):
        assert exact_match("", "") == 1.0


class TestTokenF1:
    def test_identical(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_partial_overlap(self):
        f1 = token_f1("the cat sat on the mat", "the cat ate the fish")
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        assert token_f1("hello", "goodbye") == 0.0

    def test_empty_both(self):
        assert token_f1("", "") == 1.0

    def test_empty_prediction(self):
        assert token_f1("", "hello world") == 0.0

    def test_empty_ground_truth(self):
        assert token_f1("hello world", "") == 0.0


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_none_found(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5

    def test_k_limits(self):
        assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], 3) == 1.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a", "b"], 2) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], ["a", "b"], 2) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], [], 3) == 1.0

    def test_no_relevant_found(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], 2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_not_found(self):
        assert mrr(["x", "y", "z"], ["a"]) == 0.0


class TestScorecard:
    def test_compute_scorecard(self):
        results = [
            {
                "prediction": "the cat",
                "ground_truth": "the cat",
                "retrieved_ids": ["a", "b"],
                "relevant_ids": ["a"],
                "category": "simple",
                "latency_ms": 10.0,
            },
            {
                "prediction": "the dog",
                "ground_truth": "a dog",
                "retrieved_ids": ["c", "d"],
                "relevant_ids": ["c", "d"],
                "category": "simple",
                "latency_ms": 20.0,
            },
        ]
        sc = compute_scorecard(results, k=2)
        assert sc.num_samples == 2
        assert sc.exact_match > 0  # At least one is exact
        assert sc.token_f1 > 0
        assert sc.recall_at_k > 0
        assert sc.latency_ms == 15.0
        assert "simple" in sc.category_scores

    def test_empty_results(self):
        sc = compute_scorecard([])
        assert sc.num_samples == 0
