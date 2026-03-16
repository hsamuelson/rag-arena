"""CLI entry point for the RAG Arena."""

import argparse
import sys
from pathlib import Path

from .config import ArenaConfig
from .backends import DirectEmbeddingBackend, BM25Backend, HybridBackend
from .benchmarks import (
    LoCoMoBenchmark,
    SyntheticMultiHopBenchmark,
    NaturalQuestionsBenchmark,
    HotpotQABenchmark,
    BEIRSubsetBenchmark,
)
from .hypotheses import (
    FlatBaselineHypothesis,
    PCADiversityHypothesis,
    PCAGroupedHypothesis,
    DPPSelectionHypothesis,
    ConvexHullCoverageHypothesis,
    CentroidDriftHypothesis,
    SpectralGapHypothesis,
    LocalIntrinsicDimensionHypothesis,
    HyDEHypothesis,
    ConeRetrievalHypothesis,
    GeodesicInterpolationHypothesis,
    AntiHubnessHypothesis,
    MeanBiasCorrectionHypothesis,
    QueryDecompositionHypothesis,
    CapacityPartitionHypothesis,
    ScoreCalibrationHypothesis,
    # Memory retrieval baselines
    RRFMultiPerspectiveHypothesis,
    HierarchicalClusterHypothesis,
    GraphCommunityHypothesis,
    # Novel hypotheses
    IsotropyEnhancementHypothesis,
    QueryDriftCorrectionHypothesis,
    AdaptiveContextWindowHypothesis,
    CrossEncoderProxyHypothesis,
    MahalanobisRetrievalHypothesis,
    DensityPeakSelectionHypothesis,
    InformationBottleneckHypothesis,
    SpectralRerankingHypothesis,
    LeverageScoreSamplingHypothesis,
    OptimalTransportHypothesis,
    SubmodularCoverageHypothesis,
    TopologicalPersistenceHypothesis,
    KernelHerdingHypothesis,
    VarianceReductionHypothesis,
    InfluenceFunctionHypothesis,
    RandomProjectionEnsembleHypothesis,
    CurvatureAwareHypothesis,
    AnchorExpansionHypothesis,
    MutualInformationHypothesis,
    EmbeddingTriangulationHypothesis,
    HybridAntihubInfluenceHypothesis,
    HybridCSLSTopoCalibratedHypothesis,
    HybridRRFTop5Hypothesis,
    PureCSLSHypothesis,
    CrossEncoderRerankerHypothesis,
    # Embedder-robust experiments (Round 8)
    CSLSPrefilterCEHypothesis,
    LIDGatedPoolCEHypothesis,
    HubAwareDeepPoolCEHypothesis,
    GatedMRAMCEHypothesis,
    CrossModelMaxSimHypothesis,
    RoutedRerankerHypothesis,
)
from .runners import ArenaRunner


BACKENDS = {
    "direct": DirectEmbeddingBackend,
    "bm25": BM25Backend,
    "hybrid": HybridBackend,
}

BENCHMARKS = {
    # ── Quick / synthetic ──
    "synthetic": SyntheticMultiHopBenchmark,
    # ── Conversational memory ──
    "locomo": LoCoMoBenchmark,
    # ── Standard IR benchmarks (industry baselines) ──
    "nq": NaturalQuestionsBenchmark,
    "hotpotqa": HotpotQABenchmark,
    "beir": BEIRSubsetBenchmark,
}

HYPOTHESES = {
    # ── Baselines ──
    "flat": FlatBaselineHypothesis,
    # ── PCA / SEMDA (from Caldera) ──
    "pca-diversity": PCADiversityHypothesis,
    "pca-grouped": PCAGroupedHypothesis,
    # ── Geometric: volume & coverage ──
    "dpp": DPPSelectionHypothesis,
    "hull-coverage": ConvexHullCoverageHypothesis,
    # ── Geometric: centroid & drift ──
    "centroid-drift": CentroidDriftHypothesis,
    # ── Spectral ──
    "spectral-gap": SpectralGapHypothesis,
    # ── Manifold ──
    "lid-weighted": LocalIntrinsicDimensionHypothesis,
    "geodesic": GeodesicInterpolationHypothesis,
    # ── Adaptive ──
    "cone": ConeRetrievalHypothesis,
    # ── Scaling-focused (2025-2026 research) ──
    "anti-hubness": AntiHubnessHypothesis,
    "mean-bias": MeanBiasCorrectionHypothesis,
    "capacity-partition": CapacityPartitionHypothesis,
    "score-calibration": ScoreCalibrationHypothesis,
    # ── Generative (require LLM + backend) ──
    "hyde": HyDEHypothesis,
    "query-decomp": QueryDecompositionHypothesis,
    # ── Memory retrieval baselines ──
    "rrf-multi": RRFMultiPerspectiveHypothesis,
    "hierarchical": HierarchicalClusterHypothesis,
    "graph-community": GraphCommunityHypothesis,
    # ── Novel hypotheses (20) ──
    "isotropy": IsotropyEnhancementHypothesis,
    "query-drift": QueryDriftCorrectionHypothesis,
    "adaptive-window": AdaptiveContextWindowHypothesis,
    "cross-proxy": CrossEncoderProxyHypothesis,
    "mahalanobis": MahalanobisRetrievalHypothesis,
    "density-peak": DensityPeakSelectionHypothesis,
    "info-bottleneck": InformationBottleneckHypothesis,
    "spectral-rerank": SpectralRerankingHypothesis,
    "leverage-score": LeverageScoreSamplingHypothesis,
    "optimal-transport": OptimalTransportHypothesis,
    "submodular": SubmodularCoverageHypothesis,
    "topo-persistence": TopologicalPersistenceHypothesis,
    "kernel-herding": KernelHerdingHypothesis,
    "variance-reduce": VarianceReductionHypothesis,
    "influence-fn": InfluenceFunctionHypothesis,
    "rand-proj-ensemble": RandomProjectionEnsembleHypothesis,
    "curvature": CurvatureAwareHypothesis,
    "anchor-expand": AnchorExpansionHypothesis,
    "mutual-info": MutualInformationHypothesis,
    "triangulation": EmbeddingTriangulationHypothesis,
    # ── Hybrid (Round 2) ──
    "hybrid-antihub-influence": HybridAntihubInfluenceHypothesis,
    "hybrid-csls-topo-calib": HybridCSLSTopoCalibratedHypothesis,
    "hybrid-rrf-top5": HybridRRFTop5Hypothesis,
    "pure-csls": PureCSLSHypothesis,
    # ── Cross-encoder reranking (gold standard) ──
    "cross-encoder": CrossEncoderRerankerHypothesis,
    # ── Embedder-robust experiments (Round 8) ──
    "csls-prefilter-ce": CSLSPrefilterCEHypothesis,
    "lid-gated-pool-ce": LIDGatedPoolCEHypothesis,
    "hub-aware-pool-ce": HubAwareDeepPoolCEHypothesis,
    "gated-mram-ce": GatedMRAMCEHypothesis,
    "cross-model-maxsim": CrossModelMaxSimHypothesis,
    "routed-reranker": RoutedRerankerHypothesis,
}

# The 10 hypotheses for the scaling experiment
SCALING_HYPOTHESES = [
    "flat",
    "dpp",
    "anti-hubness",
    "mean-bias",
    "capacity-partition",
    "score-calibration",
    "spectral-gap",
    "centroid-drift",
    "lid-weighted",
    "pca-diversity",
]

# Memory retrieval baselines + 20 novel hypotheses for the overnight experiment
NOVEL_HYPOTHESES = [
    "flat",  # control
    # Memory retrieval baselines (3)
    "rrf-multi",
    "hierarchical",
    "graph-community",
    # Novel hypotheses (20)
    "isotropy",
    "query-drift",
    "adaptive-window",
    "cross-proxy",
    "mahalanobis",
    "density-peak",
    "info-bottleneck",
    "spectral-rerank",
    "leverage-score",
    "optimal-transport",
    "submodular",
    "topo-persistence",
    "kernel-herding",
    "variance-reduce",
    "influence-fn",
    "rand-proj-ensemble",
    "curvature",
    "anchor-expand",
    "mutual-info",
    "triangulation",
]


def main():
    parser = argparse.ArgumentParser(
        description="RAG Arena — test retrieval hypotheses against benchmarks"
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run an experiment")
    run_p.add_argument(
        "--benchmark", "-b", choices=list(BENCHMARKS.keys()),
        default="synthetic", help="Benchmark to use"
    )
    run_p.add_argument(
        "--backend", choices=list(BACKENDS.keys()),
        default="direct", help="Backend to use"
    )
    run_p.add_argument(
        "--hypotheses", "-H", nargs="+", choices=list(HYPOTHESES.keys()),
        default=None, help="Hypotheses to test (default: scaling set of 10)"
    )
    run_p.add_argument("--max-samples", "-n", type=int, default=None)
    run_p.add_argument("--skip-llm", action="store_true", help="Skip LLM generation (retrieval metrics only)")
    run_p.add_argument("--config", "-c", type=str, default=None, help="Path to config YAML")
    run_p.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    run_p.add_argument("--quiet", "-q", action="store_true")

    # ── bench ─────────────────────────────────────────────────
    bench_p = sub.add_parser("bench", help="Run full local benchmark suite against industry baselines")
    bench_p.add_argument("--max-samples", "-n", type=int, default=50, help="Samples per benchmark")
    bench_p.add_argument("--skip-llm", action="store_true", default=True)
    bench_p.add_argument("--config", "-c", type=str, default=None)
    bench_p.add_argument("--output-dir", type=str, default=None)

    # ── compare ───────────────────────────────────────────────
    cmp_p = sub.add_parser("compare", help="Compare saved experiment results")
    cmp_p.add_argument("files", nargs="+", help="JSON result files to compare")

    # ── baselines ─────────────────────────────────────────────
    sub.add_parser("baselines", help="Print published industry baselines for reference")

    # ── init-config ───────────────────────────────────────────
    init_p = sub.add_parser("init-config", help="Generate default config file")
    init_p.add_argument("--output", "-o", default="arena.yaml")

    args = parser.parse_args()

    if args.command == "run":
        _run(args)
    elif args.command == "bench":
        _bench(args)
    elif args.command == "compare":
        _compare(args)
    elif args.command == "baselines":
        _print_baselines()
    elif args.command == "init-config":
        _init_config(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run(args):
    if args.config:
        config = ArenaConfig.from_yaml(Path(args.config))
    else:
        config = ArenaConfig()

    backend = BACKENDS[args.backend](config)
    benchmark = BENCHMARKS[args.benchmark]()

    hyp_keys = args.hypotheses or SCALING_HYPOTHESES
    hypotheses = [HYPOTHESES[h]() for h in hyp_keys]

    runner = ArenaRunner(config)
    results = runner.run_arena(
        benchmark=benchmark,
        backend=backend,
        hypotheses=hypotheses,
        max_samples=args.max_samples,
        skip_llm=args.skip_llm,
        verbose=not args.quiet,
    )

    runner.print_comparison(results)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.results_dir / f"{benchmark.name}_{backend.name}.json"

    runner.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")


def _bench(args):
    """Run full benchmark suite: all 10 hypotheses across multiple benchmarks."""
    if args.config:
        config = ArenaConfig.from_yaml(Path(args.config))
    else:
        config = ArenaConfig()

    output_dir = Path(args.output_dir) if args.output_dir else config.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = DirectEmbeddingBackend(config)
    hypotheses = [HYPOTHESES[h]() for h in SCALING_HYPOTHESES]
    runner = ArenaRunner(config)

    # Run across benchmarks
    bench_configs = [
        ("synthetic", SyntheticMultiHopBenchmark()),
        ("hotpotqa", HotpotQABenchmark(max_samples=args.max_samples)),
        ("beir", BEIRSubsetBenchmark()),
    ]

    all_results = []
    for bench_name, benchmark in bench_configs:
        print(f"\n{'#'*70}")
        print(f"# BENCHMARK: {bench_name}")
        print(f"{'#'*70}")

        results = runner.run_arena(
            benchmark=benchmark,
            backend=backend,
            hypotheses=hypotheses,
            max_samples=args.max_samples,
            skip_llm=args.skip_llm,
            verbose=True,
        )
        all_results.extend(results)

        runner.save_results(results, output_dir / f"{bench_name}_results.json")
        runner.print_comparison(results)

    # Print grand comparison
    print(f"\n{'#'*70}")
    print("# GRAND COMPARISON (all benchmarks)")
    print(f"{'#'*70}")
    runner.print_comparison(all_results)
    runner.save_results(all_results, output_dir / "all_results.json")

    # Print industry baselines for context
    _print_baselines()


def _compare(args):
    import json

    all_results = []
    for f in args.files:
        with open(f) as fh:
            data = json.load(fh)
        for item in data:
            from .metrics.scoring import ScoreCard
            sc = ScoreCard(**{k: v for k, v in item["scorecard"].items()})
            from .runners.arena_runner import ExperimentResult
            all_results.append(ExperimentResult(
                benchmark_name=item["benchmark"],
                backend_name=item["backend"],
                hypothesis_name=item["hypothesis"],
                scorecard=sc,
            ))

    ArenaRunner.print_comparison(all_results)


def _print_baselines():
    """Print published industry baselines for reference."""
    print("\n" + "=" * 70)
    print("PUBLISHED INDUSTRY BASELINES (for comparison)")
    print("=" * 70)

    print("\nNatural Questions (21M Wikipedia passages):")
    print(f"  {'Method':<20} {'Top-5':>8} {'Top-20':>8} {'Top-100':>8}")
    print(f"  {'-'*48}")
    from .benchmarks.natural_questions import NQ_BASELINES
    for method, scores in NQ_BASELINES.items():
        print(f"  {method:<20} {scores.get('top_5', '-'):>8} {scores.get('top_20', '-'):>8} {scores.get('top_100', '-'):>8}")

    print("\nHotpotQA (BEIR nDCG@10):")
    from .benchmarks.hotpotqa import HOTPOTQA_BASELINES
    for method, scores in HOTPOTQA_BASELINES.items():
        print(f"  {method:<20} {scores['ndcg_10']:.3f}")

    print("\nBEIR Zero-Shot (nDCG@10):")
    from .benchmarks.beir_subset import BEIR_BASELINES
    print(f"  {'Task':<15} {'BM25':>8} {'DPR':>8} {'E5-large':>8}")
    print(f"  {'-'*43}")
    for task, scores in BEIR_BASELINES.items():
        bm25 = f"{scores.get('BM25', '-'):.3f}" if 'BM25' in scores else '-'
        dpr = f"{scores.get('DPR', '-'):.3f}" if 'DPR' in scores else '-'
        e5 = f"{scores.get('E5_large', '-'):.3f}" if 'E5_large' in scores else '-'
        print(f"  {task:<15} {bm25:>8} {dpr:>8} {e5:>8}")

    print("\nDeepMind Embedding Capacity Ceiling (2025):")
    print(f"  {'Dimension':<15} {'Max Corpus':>15}")
    print(f"  {'-'*30}")
    for dim, max_corpus in [(512, "~500K"), (768, "~1.7M"), (1024, "~4M"), (4096, "~250M")]:
        print(f"  {dim:<15} {max_corpus:>15}")


def _init_config(args):
    config = ArenaConfig()
    config.to_yaml(Path(args.output))
    print(f"Config written to {args.output}")


if __name__ == "__main__":
    main()
