"""Tests for benchmark implementations."""

from arena.benchmarks.synthetic_multihop import SyntheticMultiHopBenchmark, SyntheticConfig


class TestSyntheticMultiHop:
    def test_load_generates_data(self):
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(
            num_chains=3, hops_per_chain=2, distractors_per_chain=2, seed=0
        ))
        bench.load()

        corpus = bench.corpus()
        samples = bench.samples()

        # 3 chains × (2 hops + 2 distractors) = 12 docs
        assert len(corpus) == 12

        # 3 chains × 2 hop depths = 6 questions
        assert len(samples) == 6

    def test_categories(self):
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(hops_per_chain=3))
        assert bench.categories() == ["1-hop", "2-hop", "3-hop"]

    def test_corpus_has_required_fields(self):
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(num_chains=2, hops_per_chain=2))
        bench.load()
        for doc in bench.corpus():
            assert "id" in doc
            assert "text" in doc
            assert len(doc["text"]) > 0

    def test_samples_have_relevant_ids(self):
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(
            num_chains=2, hops_per_chain=3, distractors_per_chain=1
        ))
        bench.load()
        for sample in bench.samples():
            assert len(sample.corpus_doc_ids) > 0
            assert len(sample.question) > 0
            assert len(sample.ground_truth) > 0

    def test_multi_hop_needs_more_docs(self):
        """Higher hop questions should need more relevant docs."""
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(
            num_chains=1, hops_per_chain=3
        ))
        bench.load()
        samples_by_cat = {}
        for s in bench.samples():
            samples_by_cat[s.category] = s

        assert len(samples_by_cat["1-hop"].corpus_doc_ids) == 1
        assert len(samples_by_cat["2-hop"].corpus_doc_ids) == 2
        assert len(samples_by_cat["3-hop"].corpus_doc_ids) == 3

    def test_deterministic(self):
        """Same seed should produce identical data."""
        b1 = SyntheticMultiHopBenchmark(SyntheticConfig(seed=42))
        b2 = SyntheticMultiHopBenchmark(SyntheticConfig(seed=42))
        b1.load()
        b2.load()
        assert [d["id"] for d in b1.corpus()] == [d["id"] for d in b2.corpus()]
        assert [s.question for s in b1.samples()] == [s.question for s in b2.samples()]

    def test_name(self):
        assert SyntheticMultiHopBenchmark().name == "synthetic-multihop"

    def test_scaling(self):
        """Should handle large corpus sizes without issues."""
        bench = SyntheticMultiHopBenchmark(SyntheticConfig(
            num_chains=100, hops_per_chain=4, distractors_per_chain=20
        ))
        bench.load()
        # 100 × (4 + 20) = 2400 docs
        assert len(bench.corpus()) == 2400
        # 100 × 4 = 400 questions
        assert len(bench.samples()) == 400
