"""ArenaRunner — orchestrates benchmark × backend × hypothesis experiments.

The runner:
1. Loads a benchmark (corpus + questions)
2. Ingests corpus into a backend
3. For each hypothesis:
   a. Retrieves results for each question
   b. Applies the hypothesis transform
   c. Sends context + question to Ollama LLM for answer generation
   d. Scores against ground truth
4. Produces comparative scorecards
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from ..backends.base import Backend, RetrievalResult
from ..benchmarks.base import Benchmark, BenchmarkSample
from ..hypotheses.base import Hypothesis
from ..metrics.scoring import ScoreCard, compute_scorecard
from ..config import ArenaConfig


@dataclass
class ExperimentResult:
    """Full result of one experiment run (one benchmark × backend × hypothesis)."""
    benchmark_name: str
    backend_name: str
    hypothesis_name: str
    scorecard: ScoreCard
    per_question: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark_name,
            "backend": self.backend_name,
            "hypothesis": self.hypothesis_name,
            "scorecard": self.scorecard.to_dict(),
            "per_question": self.per_question,
            "config": self.config,
            "timestamp": self.timestamp,
        }


class ArenaRunner:
    """Runs RAG experiments across benchmarks, backends, and hypotheses."""

    def __init__(self, config: ArenaConfig):
        self.config = config

    def run_experiment(
        self,
        benchmark: Benchmark,
        backend: Backend,
        hypothesis: Hypothesis,
        max_samples: int | None = None,
        skip_llm: bool = False,
        verbose: bool = True,
    ) -> ExperimentResult:
        """Run a single experiment: one benchmark × backend × hypothesis.

        Args:
            benchmark: The benchmark providing corpus and questions.
            backend: The retrieval backend to use.
            hypothesis: The retrieval hypothesis to test.
            max_samples: Limit number of questions (for quick testing).
            skip_llm: If True, skip LLM answer generation (retrieval-only metrics).
            verbose: Show progress bar.
        """
        # 1. Load benchmark
        benchmark.load(str(self.config.data_dir))
        corpus = benchmark.corpus()
        samples = benchmark.samples()
        if max_samples:
            samples = samples[:max_samples]

        # 2. Ingest corpus into backend
        if verbose:
            print(f"Ingesting {len(corpus)} documents into {backend.name}...")
        backend.ingest(corpus)

        # 3. Run each question
        per_question_results = []
        iterator = tqdm(samples, desc=f"{hypothesis.name}", disable=not verbose)

        for sample in iterator:
            result = self._run_single(sample, backend, hypothesis, skip_llm)
            per_question_results.append(result)

        # 4. Compute scorecard
        scorecard = compute_scorecard(per_question_results, k=self.config.top_k)

        return ExperimentResult(
            benchmark_name=benchmark.name,
            backend_name=backend.name,
            hypothesis_name=hypothesis.name,
            scorecard=scorecard,
            per_question=per_question_results,
            config={
                "top_k": self.config.top_k,
                "n_components": self.config.n_components,
                "embed_model": self.config.ollama.embed_model,
                "chat_model": self.config.ollama.chat_model,
                "max_samples": max_samples,
                "skip_llm": skip_llm,
            },
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def _run_single(
        self,
        sample: BenchmarkSample,
        backend: Backend,
        hypothesis: Hypothesis,
        skip_llm: bool,
    ) -> dict:
        """Run one question through the pipeline."""
        t0 = time.time()

        # Retrieve
        results, embeddings = backend.retrieve_with_embeddings(
            sample.question, self.config.top_k
        )

        # Get query embedding for hypotheses that need it
        query_emb = None
        try:
            query_emb = backend.embed_query(sample.question)
        except Exception:
            pass

        # Apply hypothesis
        hyp_result = hypothesis.apply(sample.question, results, embeddings, query_emb)

        # Generate answer via LLM
        if skip_llm:
            prediction = ""
        else:
            prediction = self._generate_answer(
                sample.question, hyp_result.context_prompt
            )

        latency_ms = (time.time() - t0) * 1000

        return {
            "question_id": sample.question_id,
            "question": sample.question,
            "prediction": prediction,
            "ground_truth": sample.ground_truth,
            "category": sample.category,
            "retrieved_ids": [r.doc_id for r in hyp_result.results],
            "relevant_ids": sample.corpus_doc_ids,
            "latency_ms": latency_ms,
            "hypothesis_metadata": hyp_result.metadata,
        }

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using Ollama chat API."""
        system_prompt = (
            "You are a precise question-answering assistant. "
            "Answer the question using ONLY the provided context. "
            "Be concise — give the answer directly without explanation. "
            "If the context doesn't contain the answer, say 'I don't know'."
        )

        user_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        try:
            resp = requests.post(
                f"{self.config.ollama.base_url}/api/chat",
                json={
                    "model": self.config.ollama.chat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=self.config.ollama.timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            return f"[LLM_ERROR: {e}]"

    def run_arena(
        self,
        benchmark: Benchmark,
        backend: Backend,
        hypotheses: list[Hypothesis],
        max_samples: int | None = None,
        skip_llm: bool = False,
        verbose: bool = True,
    ) -> list[ExperimentResult]:
        """Run multiple hypotheses against the same benchmark and backend.

        This is the main entry point for comparative testing.
        """
        results = []

        for hyp in hypotheses:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {hyp.name}")
                print(f"  {hyp.description}")
                print(f"{'='*60}")

            result = self.run_experiment(
                benchmark, backend, hyp,
                max_samples=max_samples,
                skip_llm=skip_llm,
                verbose=verbose,
            )
            results.append(result)

            if verbose:
                sc = result.scorecard
                print(f"\nResults for {hyp.name}:")
                print(f"  EM={sc.exact_match:.3f}  F1={sc.token_f1:.3f}  "
                      f"R@K={sc.recall_at_k:.3f}  nDCG={sc.ndcg_at_k:.3f}  "
                      f"MRR={sc.mrr:.3f}  latency={sc.latency_ms:.1f}ms")

        return results

    @staticmethod
    def save_results(results: list[ExperimentResult], path: Path) -> None:
        """Save experiment results to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]

        def _default(obj):
            """Handle numpy types in JSON serialization."""
            import numpy as np
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_default)

    @staticmethod
    def print_comparison(results: list[ExperimentResult]) -> None:
        """Print a comparison table of experiment results."""
        if not results:
            return

        header = f"{'Hypothesis':<35} {'EM':>6} {'F1':>6} {'R@K':>6} {'nDCG':>6} {'MRR':>6} {'ms':>8}"
        print(f"\n{header}")
        print("-" * len(header))

        for r in results:
            sc = r.scorecard
            print(
                f"{r.hypothesis_name:<35} "
                f"{sc.exact_match:>6.3f} "
                f"{sc.token_f1:>6.3f} "
                f"{sc.recall_at_k:>6.3f} "
                f"{sc.ndcg_at_k:>6.3f} "
                f"{sc.mrr:>6.3f} "
                f"{sc.latency_ms:>8.1f}"
            )

        # Per-category breakdown
        all_categories = set()
        for r in results:
            all_categories.update(r.scorecard.category_scores.keys())

        if len(all_categories) > 1:
            for cat in sorted(all_categories):
                print(f"\n  Category: {cat}")
                cat_header = f"  {'Hypothesis':<33} {'EM':>6} {'F1':>6} {'R@K':>6} {'N':>4}"
                print(f"  {cat_header}")
                print(f"  {'-' * len(cat_header)}")

                for r in results:
                    cs = r.scorecard.category_scores.get(cat, {})
                    print(
                        f"  {r.hypothesis_name:<33} "
                        f"{cs.get('exact_match', 0):>6.3f} "
                        f"{cs.get('token_f1', 0):>6.3f} "
                        f"{cs.get('recall_at_k', 0):>6.3f} "
                        f"{cs.get('num_samples', 0):>4d}"
                    )
