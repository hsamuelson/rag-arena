#!/usr/bin/env python3
"""Upgraded SOTA Benchmark — Stronger Embedder + Fixed Mxbai Reranker.

Runs a focused subset of the SOTA benchmark strategies using an upgraded
embedding model (default: snowflake-arctic-embed-l) instead of nomic-embed-text,
and includes the now-fixed mxbai reranker.

Key differences from run_sota_benchmark.py:
  - Uses HybridSTBackend (sentence-transformers) when a non-Ollama embed model
    is specified, falling back to the standard HybridBackend otherwise.
  - Late interaction reranker uses the SAME model as the embedder (not hardcoded
    nomic-embed-text) for fairness.
  - Prints a comparison section showing deltas vs the original benchmark run.
  - Includes expanded published BEIR baselines for context.

Strategies (10, focused on top performers + fixed mxbai):
  1. Flat baseline (no reranking)
  2. CE L-6 (baseline reference)
  3. CE L-12 (top-50)
  4. BGE-v2-m3 (top-50)
  5. Mxbai-rerank-base-v2 (top-50) — now fixed
  6. MRAM-v1.5 (sentence + CE L-12)
  7. MRAM + BGE
  8. Late interaction (MaxSim) — uses same embedder for fairness
  9. Late interaction + MRAM — same note
  10. Multi-reranker ensemble
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from arena.config import ArenaConfig
from arena.benchmarks.beir_subset import BEIRSubsetBenchmark
from arena.metrics.scoring import ndcg_at_k, compute_scorecard

# ── Published BEIR FiQA baselines (nDCG@10) ──────────────────────────────────
BEIR_BASELINES = {
    "BM25": 0.236,
    "DPR": 0.112,
    "E5-large": 0.370,
    "bge-small-en-v1.5": 0.403,
    "snowflake-arctic-embed-l": 0.447,
    "GTE-ModernColBERT": 0.453,
    "jina-reranker-v3": 0.518,
    "mxbai-rerank-large-v2": 0.528,
}

# ── Original benchmark results (hardcoded from previous run with nomic-embed-text) ──
ORIGINAL_RESULTS = {
    "Flat baseline": None,        # fill with actual value once known
    "CE L-6 (top-10)": None,
    "CE L-12 (top-50)": None,
    "BGE-v2-m3 (top-50)": None,
    "Mxbai-rerank (top-50)": None,
    "MRAM-v1.5": None,
    "MRAM + BGE": None,
    "Late interaction": None,
    "Late interaction + MRAM": None,
    "Multi-reranker ensemble": None,
}

# Try to load actual original results from previous benchmark JSON
_ORIGINAL_RESULTS_PATH = Path(__file__).parent / "results" / "sota_benchmark_raw.json"
if _ORIGINAL_RESULTS_PATH.exists():
    try:
        with open(_ORIGINAL_RESULTS_PATH) as _f:
            _raw = json.load(_f)
        for _name in list(ORIGINAL_RESULTS.keys()):
            if _name in _raw:
                ORIGINAL_RESULTS[_name] = float(np.mean(_raw[_name]))
    except Exception:
        pass


# ── Late Interaction wrapper (uses configurable model) ────────────────────────

class ConfigurableLateInteractionReranker:
    """Thin wrapper around LateInteractionRerankerHypothesis that injects
    a custom sentence-transformers model for MaxSim scoring.

    This avoids modifying the working late_interaction_reranker.py while
    ensuring the LI reranker uses the same embedder as the retrieval backend.
    """

    def __init__(self, model_name, pool_size=50, final_k=10):
        from arena.hypotheses.late_interaction_reranker import LateInteractionRerankerHypothesis
        self._inner = LateInteractionRerankerHypothesis(pool_size=pool_size, final_k=final_k)
        self._model_name = model_name

    def set_backend(self, backend):
        self._inner.set_backend(backend)

    def _inject_model(self):
        """Replace the inner model lazy-loader to use our model name."""
        if self._inner._model is None:
            from sentence_transformers import SentenceTransformer
            self._inner._model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
            )

    @property
    def name(self):
        return self._inner.name

    @property
    def description(self):
        return f"Top-50 reranked by token-level MaxSim ({self._model_name})"

    def apply(self, query, results, embeddings, query_embedding):
        self._inject_model()
        return self._inner.apply(query, results, embeddings, query_embedding)


class ConfigurableLateInteractionMRAM:
    """Thin wrapper around LateInteractionMRAMHypothesis that injects
    a custom sentence-transformers model for MaxSim scoring."""

    def __init__(self, model_name, pool_per_level=50, final_k=10):
        from arena.hypotheses.late_interaction_mram import LateInteractionMRAMHypothesis
        self._inner = LateInteractionMRAMHypothesis(
            pool_per_level=pool_per_level, final_k=final_k
        )
        self._model_name = model_name

    def set_backend(self, backend):
        self._inner.set_backend(backend)

    def _inject_model(self):
        """Replace the inner model lazy-loader to use our model name."""
        if self._inner._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._inner._st_model = SentenceTransformer(
                self._model_name,
                trust_remote_code=True,
            )

    @property
    def name(self):
        return self._inner.name

    @property
    def description(self):
        return f"MRAM + MaxSim token-level rerank ({self._model_name})"

    def apply(self, query, results, embeddings, query_embedding):
        self._inject_model()
        return self._inner.apply(query, results, embeddings, query_embedding)


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_per_query_ndcg(per_question_results, k=10):
    ndcgs = []
    for q in per_question_results:
        retrieved = q["retrieved_ids"][:k]
        relevant = q["relevant_ids"]
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
    return np.array(ndcgs)


def bootstrap_ci(scores, n_bootstrap=2000, ci=0.95):
    rng = np.random.RandomState(42)
    means = sorted(scores[rng.randint(0, len(scores), size=len(scores))].mean()
                   for _ in range(n_bootstrap))
    lo = means[int((1 - ci) / 2 * n_bootstrap)]
    hi = means[int((1 + ci) / 2 * n_bootstrap)]
    return lo, hi


def paired_wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test. Returns p-value."""
    try:
        from scipy.stats import wilcoxon
        diffs = scores_a - scores_b
        diffs = diffs[diffs != 0]
        if len(diffs) < 10:
            return 1.0
        stat, p_val = wilcoxon(diffs)
        return float(p_val)
    except ImportError:
        return paired_permutation_test(scores_a, scores_b)


def paired_permutation_test(scores_a, scores_b, n_perms=5000):
    rng = np.random.RandomState(42)
    observed = scores_a.mean() - scores_b.mean()
    diffs = scores_a - scores_b
    count = sum(1 for _ in range(n_perms)
                if (diffs * rng.choice([-1, 1], size=len(diffs))).mean() >= observed)
    return count / n_perms


def holm_bonferroni(p_values_list):
    """Apply Holm-Bonferroni correction to a list of (name, p_value, ...) tuples.

    Returns list of (name, p_value, threshold, significant, ...) tuples sorted by p-value.
    """
    sorted_pvals = sorted(p_values_list, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    results = []
    for rank, item in enumerate(sorted_pvals):
        name, p_val = item[0], item[1]
        rest = item[2:]
        threshold = 0.05 / (n_tests - rank)
        sig = p_val < threshold
        results.append((name, p_val, threshold, sig, *rest))
    return results


# ── Hypothesis runner ─────────────────────────────────────────────────────────

def run_hypothesis(name, hyp, samples, backend, config):
    from tqdm import tqdm

    per_question = []
    iterator = tqdm(samples, desc=name)

    for sample in iterator:
        t0 = time.time()
        results, embeddings = backend.retrieve_with_embeddings(sample.question, config.top_k)
        query_emb = None
        try:
            query_emb = backend.embed_query(sample.question)
        except Exception:
            pass

        hyp_result = hyp.apply(sample.question, results, embeddings, query_emb)
        latency_ms = (time.time() - t0) * 1000

        per_question.append({
            "question_id": sample.question_id,
            "question": sample.question,
            "prediction": "",
            "ground_truth": sample.ground_truth,
            "category": sample.category,
            "retrieved_ids": [r.doc_id for r in hyp_result.results],
            "relevant_ids": sample.corpus_doc_ids,
            "latency_ms": latency_ms,
            "hypothesis_metadata": hyp_result.metadata,
        })

    scorecard = compute_scorecard(per_question, k=config.top_k)
    return per_question, scorecard


# ── Backend creation ──────────────────────────────────────────────────────────

def create_backend(config, embed_model):
    """Create the appropriate hybrid backend.

    If embed_model looks like a sentence-transformers model (contains '/')
    and HybridSTBackend is available, use that. Otherwise fall back to the
    standard HybridBackend (which uses Ollama).
    """
    if "/" in embed_model:
        try:
            from arena.backends.hybrid_st_backend import HybridSTBackend
            print(f"  Using HybridSTBackend with {embed_model}")
            return HybridSTBackend(config, model_name=embed_model)
        except ImportError:
            print(f"  HybridSTBackend not available, falling back to HybridBackend")
            print(f"  (embed_model '{embed_model}' will use Ollama's configured model)")

    from arena.backends.hybrid_backend import HybridBackend
    print(f"  Using HybridBackend (Ollama: {config.ollama.embed_model})")
    return HybridBackend(config)


# ── Build hypotheses ──────────────────────────────────────────────────────────

def build_hypotheses(backend, embed_model):
    """Build the 10 strategies for the upgraded benchmark."""
    from arena.hypotheses.baseline_flat import FlatBaselineHypothesis
    from arena.hypotheses.cross_encoder_reranker import CrossEncoderRerankerHypothesis
    from arena.hypotheses.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis
    from arena.hypotheses.bge_reranker import BGERerankerHypothesis
    from arena.hypotheses.mxbai_reranker import MxbaiRerankerHypothesis
    from arena.hypotheses.multi_resolution import MultiResolutionHypothesis
    from arena.hypotheses.mram_bge_reranker import MRAMBGERerankerHypothesis
    from arena.hypotheses.multi_reranker_ensemble import MultiRerankerEnsembleHypothesis

    strategies = []

    # 1. Flat baseline
    flat = FlatBaselineHypothesis()
    strategies.append(("Flat baseline", flat))

    # 2. CE L-6 (baseline reference)
    ce_l6 = CrossEncoderRerankerHypothesis()
    strategies.append(("CE L-6 (top-10)", ce_l6))

    # 3. CE L-12 (top-50)
    ce_l12 = DeepPool50CEL12Hypothesis()
    ce_l12.set_backend(backend)
    strategies.append(("CE L-12 (top-50)", ce_l12))

    # 4. BGE-v2-m3 (top-50)
    bge = BGERerankerHypothesis()
    bge.set_backend(backend)
    strategies.append(("BGE-v2-m3 (top-50)", bge))

    # 5. Mxbai-rerank-base-v2 (top-50) -- now fixed
    mxbai = MxbaiRerankerHypothesis()
    mxbai.set_backend(backend)
    strategies.append(("Mxbai-rerank (top-50)", mxbai))

    # 6. MRAM-v1.5
    mram = MultiResolutionHypothesis()
    mram.set_backend(backend)
    strategies.append(("MRAM-v1.5", mram))

    # 7. MRAM + BGE
    mram_bge = MRAMBGERerankerHypothesis()
    mram_bge.set_backend(backend)
    strategies.append(("MRAM + BGE", mram_bge))

    # 8. Late interaction (MaxSim) -- uses same embedder for fairness
    late = ConfigurableLateInteractionReranker(model_name=embed_model)
    late.set_backend(backend)
    strategies.append(("Late interaction", late))

    # 9. Late interaction + MRAM -- uses same embedder for fairness
    late_mram = ConfigurableLateInteractionMRAM(model_name=embed_model)
    late_mram.set_backend(backend)
    strategies.append(("Late interaction + MRAM", late_mram))

    # 10. Multi-reranker ensemble
    ensemble = MultiRerankerEnsembleHypothesis()
    ensemble.set_backend(backend)
    strategies.append(("Multi-reranker ensemble", ensemble))

    return strategies


# ── Display helpers ───────────────────────────────────────────────────────────

def print_tier_results(tier_results, baseline_ndcgs):
    """Print a comparison table for all strategies."""
    print(f"\n{'=' * 90}")
    print(f"  {'Strategy':<30s}  {'nDCG@10':>8s}  {'vs CE-L6':>8s}  "
          f"{'95% CI':>18s}  {'p-val':>8s}")
    print(f"  {'=' * 84}")

    for name, data in tier_results.items():
        ndcg = data["mean_ndcg"]
        ci = bootstrap_ci(data["ndcgs"])
        if baseline_ndcgs is not None and name != "Flat baseline":
            diff_pct = (ndcg - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            if ndcg >= baseline_ndcgs.mean():
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"  {name:<30s}  {ndcg:8.4f}  {diff_pct:+7.1f}%  "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]  {p_val:.4f} {sig}")
        else:
            print(f"  {name:<30s}  {ndcg:8.4f}  {'--':>8s}  "
                  f"[{ci[0]:.4f}, {ci[1]:.4f}]")


def print_comparison(all_results, embed_model):
    """Print comparison of upgraded vs original benchmark results."""
    print(f"\n{'#' * 90}")
    print("  COMPARISON: Upgraded vs Original Benchmark")
    print(f"{'#' * 90}")
    print(f"\n  Upgraded embedder: {embed_model}")
    print(f"  Original embedder: nomic-embed-text (768-dim via Ollama)")
    print()
    print(f"  {'Strategy':<30s}  {'Original':>8s}  {'Upgraded':>8s}  {'Delta':>10s}")
    print(f"  {'-' * 62}")

    for name, data in all_results.items():
        upgraded = data["mean_ndcg"]
        original = ORIGINAL_RESULTS.get(name)
        if original is not None:
            delta = upgraded - original
            delta_pct = delta / original * 100 if original > 0 else 0.0
            print(f"  {name:<30s}  {original:8.4f}  {upgraded:8.4f}  "
                  f"{delta:+8.4f} ({delta_pct:+.1f}%)")
        else:
            print(f"  {name:<30s}  {'N/A':>8s}  {upgraded:8.4f}  {'N/A':>10s}")


def print_beir_context(all_results):
    """Print results in context of published BEIR baselines."""
    print(f"\n{'#' * 90}")
    print("  CONTEXT: Published BEIR FiQA Baselines")
    print(f"{'#' * 90}")
    print()
    print(f"  {'Method':<35s}  {'nDCG@10':>8s}  {'Source':>15s}")
    print(f"  {'-' * 62}")

    for name, score in BEIR_BASELINES.items():
        print(f"  {name:<35s}  {score:8.3f}  {'published':>15s}")

    print(f"  {'-' * 62}")

    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_ndcg"], reverse=True)
    for name, data in sorted_results:
        print(f"  {name:<35s}  {data['mean_ndcg']:8.4f}  {'this run':>15s}")


def print_composability(all_results):
    """Print composability analysis."""
    print(f"\n{'#' * 90}")
    print("  COMPOSABILITY ANALYSIS")
    print(f"{'#' * 90}")
    print()

    mram_score = all_results.get("MRAM-v1.5", {}).get("mean_ndcg", 0)
    bge_score = all_results.get("BGE-v2-m3 (top-50)", {}).get("mean_ndcg", 0)
    mram_bge_score = all_results.get("MRAM + BGE", {}).get("mean_ndcg", 0)
    if mram_score and bge_score and mram_bge_score:
        composable = mram_bge_score > max(mram_score, bge_score)
        label = "COMPOSABLE" if composable else "Not composable"
        print(f"  MRAM + BGE = {mram_bge_score:.4f} vs "
              f"max(MRAM={mram_score:.4f}, BGE={bge_score:.4f}) = {max(mram_score, bge_score):.4f} "
              f"-> {label}")

    late_score = all_results.get("Late interaction", {}).get("mean_ndcg", 0)
    ce_l12_score = all_results.get("CE L-12 (top-50)", {}).get("mean_ndcg", 0)
    if late_score and ce_l12_score:
        winner = "MaxSim wins" if late_score > ce_l12_score else "CE L-12 still better"
        print(f"  Late interaction ({late_score:.4f}) vs CE L-12 ({ce_l12_score:.4f}) -> {winner}")

    late_mram_score = all_results.get("Late interaction + MRAM", {}).get("mean_ndcg", 0)
    late_score = all_results.get("Late interaction", {}).get("mean_ndcg", 0)
    mram_score = all_results.get("MRAM-v1.5", {}).get("mean_ndcg", 0)
    if late_mram_score and late_score and mram_score:
        composable = late_mram_score > max(late_score, mram_score)
        label = "COMPOSABLE" if composable else "Not composable"
        print(f"  LI + MRAM = {late_mram_score:.4f} vs "
              f"max(LI={late_score:.4f}, MRAM={mram_score:.4f}) = {max(late_score, mram_score):.4f} "
              f"-> {label}")

    ensemble_score = all_results.get("Multi-reranker ensemble", {}).get("mean_ndcg", 0)
    if ensemble_score and ce_l12_score and bge_score and late_score:
        best_single = max(ce_l12_score, bge_score, late_score)
        composable = ensemble_score > best_single
        label = "COMPOSABLE" if composable else "Not composable"
        print(f"  Ensemble = {ensemble_score:.4f} vs "
              f"best single reranker = {best_single:.4f} -> {label}")


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(all_results, output_dir, embed_model, max_samples):
    """Generate markdown report with full results."""
    baseline_ndcg = all_results.get("CE L-6 (top-10)", {}).get("mean_ndcg", 0)
    baseline_ndcgs = all_results.get("CE L-6 (top-10)", {}).get("ndcgs")

    lines = [
        "# Upgraded SOTA Benchmark Results",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Benchmark:** FiQA ({max_samples} queries, nDCG@10)",
        f"**Embedder:** {embed_model}",
        f"**Backend:** {embed_model} + BM25 hybrid, RRF fusion, top-50 candidates",
        "",
        "## Published BEIR FiQA Baselines",
        "",
        "| Method | nDCG@10 |",
        "|--------|---------|",
    ]
    for name, score in BEIR_BASELINES.items():
        lines.append(f"| {name} | {score:.3f} |")

    lines.extend(["", "## Results", ""])
    lines.append("| # | Strategy | nDCG@10 | vs CE-L6 | 95% CI | p-value | Sig |")
    lines.append("|---|----------|---------|----------|--------|---------|-----|")

    for i, (name, data) in enumerate(all_results.items(), 1):
        ndcg = data["mean_ndcg"]
        ci = bootstrap_ci(data["ndcgs"])
        if baseline_ndcgs is not None and name not in ("Flat baseline",):
            diff_pct = (ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg else 0
            if ndcg >= baseline_ndcg:
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            lines.append(
                f"| {i} | {name} | {ndcg:.4f} | {diff_pct:+.1f}% | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {p_val:.4f} | {sig} |"
            )
        else:
            lines.append(f"| {i} | {name} | {ndcg:.4f} | -- | [{ci[0]:.4f}, {ci[1]:.4f}] | -- | -- |")

    # Holm-Bonferroni correction
    lines.extend(["", "## Statistical Notes", ""])
    lines.append("- **Test:** Paired Wilcoxon signed-rank test vs CE L-6 baseline")
    lines.append(f"- **Correction:** Holm-Bonferroni for {len(all_results) - 2} comparisons")
    lines.append("- **CI:** Bootstrap 95% confidence interval (2000 resamples, seed=42)")

    if baseline_ndcgs is not None:
        p_values = []
        for name, data in all_results.items():
            if name in ("Flat baseline", "CE L-6 (top-10)"):
                continue
            ndcg = data["mean_ndcg"]
            if ndcg >= baseline_ndcg:
                p = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            p_values.append((name, p, data["mean_ndcg"]))

        corrected = holm_bonferroni(p_values)
        lines.extend(["", "### Holm-Bonferroni Corrected Results", ""])
        lines.append("| Rank | Strategy | p-value | Threshold | Significant? |")
        lines.append("|------|----------|---------|-----------|-------------|")
        for rank, (name, p, threshold, sig, ndcg) in enumerate(corrected, 1):
            sig_str = "Yes" if sig else "No"
            lines.append(f"| {rank} | {name} | {p:.6f} | {threshold:.6f} | {sig_str} |")

    # Comparison vs original
    lines.extend(["", "## Comparison vs Original Benchmark (nomic-embed-text)", ""])
    lines.append("| Strategy | Original | Upgraded | Delta |")
    lines.append("|----------|----------|----------|-------|")
    for name, data in all_results.items():
        upgraded = data["mean_ndcg"]
        original = ORIGINAL_RESULTS.get(name)
        if original is not None:
            delta = upgraded - original
            lines.append(f"| {name} | {original:.4f} | {upgraded:.4f} | {delta:+.4f} |")
        else:
            lines.append(f"| {name} | N/A | {upgraded:.4f} | N/A |")

    # Top 3
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_ndcg"], reverse=True)
    lines.extend(["", "## Top 3 Strategies", ""])
    for i, (name, data) in enumerate(sorted_results[:3], 1):
        lines.append(f"{i}. **{name}**: {data['mean_ndcg']:.4f} nDCG@10")

    report = "\n".join(lines)
    report_path = output_dir / "upgraded_sota_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upgraded SOTA benchmark with stronger embedder + fixed mxbai reranker"
    )
    parser.add_argument(
        "--embed-model",
        default="Snowflake/snowflake-arctic-embed-l",
        help="Sentence-transformers model to use for embeddings (default: Snowflake/snowflake-arctic-embed-l)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Number of queries to evaluate (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory for output files (default: experiments/results)",
    )
    args = parser.parse_args()

    np.random.seed(42)

    config = ArenaConfig.from_yaml(Path(__file__).parent.parent / "arena.yaml")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_model = args.embed_model
    max_samples = args.max_samples

    print("=" * 90)
    print("UPGRADED SOTA BENCHMARK — Stronger Embedder + Fixed Mxbai")
    print("=" * 90)
    print(f"Embedder:  {embed_model}")
    print(f"Benchmark: FiQA, {max_samples} queries, nDCG@10")
    print(f"Seed:      42")
    print()

    # ── Load FiQA ──
    print("### Loading FiQA...")
    benchmark = BEIRSubsetBenchmark(tasks=["fiqa"])
    benchmark.load(str(config.data_dir))
    corpus = benchmark.corpus()
    samples = benchmark.samples()[:max_samples]
    print(f"  Corpus: {len(corpus)} docs, Queries: {len(samples)}")

    # ── Create backend ──
    print(f"\n### Creating backend...")
    backend = create_backend(config, embed_model)

    print(f"\n### Ingesting {len(corpus)} documents...")
    t0 = time.time()
    backend.ingest(corpus)
    ingest_time = time.time() - t0
    print(f"  Ingestion: {ingest_time:.1f}s")

    # ── Build strategies ──
    strategies = build_hypotheses(backend, embed_model)

    all_results = {}
    baseline_ndcgs = None

    # ── Run all strategies ──
    for name, hyp in strategies:
        print(f"\n{'=' * 70}")
        print(f"  Running: {name}")
        print(f"{'=' * 70}")

        start = time.time()
        try:
            per_question, scorecard = run_hypothesis(name, hyp, samples, backend, config)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        elapsed = time.time() - start
        ndcgs = compute_per_query_ndcg(per_question)

        data = {
            "per_question": per_question,
            "scorecard": scorecard,
            "ndcgs": ndcgs,
            "mean_ndcg": float(ndcgs.mean()),
            "elapsed": elapsed,
        }
        all_results[name] = data

        print(f"  nDCG@10={ndcgs.mean():.4f}  R@K={scorecard.recall_at_k:.3f}  "
              f"MRR={scorecard.mrr:.3f}  ({elapsed:.1f}s)")

        # Track baseline
        if name == "CE L-6 (top-10)":
            baseline_ndcgs = ndcgs

        # Save incrementally
        raw = {n: d["ndcgs"].tolist() for n, d in all_results.items()}
        with open(output_dir / "upgraded_sota_raw.json", "w") as f:
            json.dump(raw, f, indent=2)

    # ── Results summary ──
    print(f"\n{'#' * 90}")
    print("FINAL RESULTS")
    print(f"{'#' * 90}")

    print_tier_results(all_results, baseline_ndcgs)

    # ── Statistical tests vs CE L-6 with Holm-Bonferroni ──
    if baseline_ndcgs is not None:
        print(f"\n  Statistical comparisons vs CE L-6 (Holm-Bonferroni corrected):")
        p_values = []
        for name, data in all_results.items():
            if name in ("Flat baseline", "CE L-6 (top-10)"):
                continue
            diff = (data["mean_ndcg"] - baseline_ndcgs.mean()) / baseline_ndcgs.mean() * 100
            if data["mean_ndcg"] >= baseline_ndcgs.mean():
                p_val = paired_wilcoxon_test(data["ndcgs"], baseline_ndcgs)
            else:
                p_val = paired_wilcoxon_test(baseline_ndcgs, data["ndcgs"])
            ci = bootstrap_ci(data["ndcgs"])
            p_values.append((name, p_val, diff, ci))

        corrected = holm_bonferroni(p_values)
        for name, p_val, threshold, sig, diff, ci in corrected:
            sig_str = "***" if p_val < threshold / 10 else "**" if p_val < threshold / 2 else "*" if sig else "ns"
            print(f"    {name:<30s}: {diff:+.1f}% (p={p_val:.4f} {sig_str}), "
                  f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    # ── Comparison vs original ──
    print_comparison(all_results, embed_model)

    # ── BEIR context ──
    print_beir_context(all_results)

    # ── Composability ──
    print_composability(all_results)

    # ── Top 3 ──
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_ndcg"], reverse=True)
    print(f"\n  Top 3 strategies:")
    for i, (name, data) in enumerate(sorted_results[:3], 1):
        print(f"    {i}. {name}: {data['mean_ndcg']:.4f}")

    # ── Save final results ──
    raw = {name: data["ndcgs"].tolist() for name, data in all_results.items()}
    with open(output_dir / "upgraded_sota_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    summary = {
        "date": datetime.now().isoformat(),
        "benchmark": "FiQA",
        "n_queries": max_samples,
        "seed": 42,
        "embed_model": embed_model,
        "backend": f"{embed_model} + BM25 hybrid, RRF fusion, top-50",
        "beir_baselines": BEIR_BASELINES,
        "original_results": {k: v for k, v in ORIGINAL_RESULTS.items() if v is not None},
        "results": {
            name: {
                "ndcg_at_10": data["mean_ndcg"],
                "elapsed_s": data["elapsed"],
                "vs_e5_large_pct": (data["mean_ndcg"] - BEIR_BASELINES["E5-large"])
                                   / BEIR_BASELINES["E5-large"] * 100,
            }
            for name, data in all_results.items()
        },
    }
    with open(output_dir / "upgraded_sota_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate markdown report
    generate_report(all_results, output_dir, embed_model, max_samples)

    print(f"\nAll results saved to {output_dir}/upgraded_sota_*.json")


if __name__ == "__main__":
    main()
