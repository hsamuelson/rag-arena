"""BEIR benchmark subset — zero-shot transfer evaluation.

Tests how well retrieval generalises to unseen domains. We include
a curated subset of BEIR that covers diverse domain types and is
practical to run locally.

Selected tasks (6 diverse domains):
  - SciFact:    Scientific claim verification (small, precise)
  - NFCorpus:   Biomedical (nutrition/fitness)
  - FiQA:       Financial opinion QA
  - TREC-COVID: COVID-19 biomedical search
  - ArguAna:    Argument retrieval
  - HotpotQA:   Multi-hop (via BEIR format)

Standard BM25 nDCG@10 baselines:
  SciFact:     0.665
  NFCorpus:    0.325
  FiQA:        0.236
  TREC-COVID:  0.656
  ArguAna:     0.315

Source: BeIR/* on HuggingFace
"""

from pathlib import Path

from .base import Benchmark, BenchmarkSample


BEIR_BASELINES = {
    "scifact": {"BM25": 0.665, "DPR": 0.318, "E5_large": 0.720},
    "nfcorpus": {"BM25": 0.325, "DPR": 0.189, "E5_large": 0.360},
    "fiqa": {"BM25": 0.236, "DPR": 0.112, "E5_large": 0.370},
    "trec-covid": {"BM25": 0.656, "DPR": 0.332},
    "arguana": {"BM25": 0.315, "DPR": 0.175},
}

# These are small enough to run locally quickly
DEFAULT_TASKS = ["scifact", "nfcorpus", "fiqa"]


class BEIRSubsetBenchmark(Benchmark):
    """BEIR zero-shot transfer benchmark (curated subset)."""

    def __init__(self, tasks: list[str] | None = None):
        self._tasks = tasks or DEFAULT_TASKS
        self._corpus: list[dict] = []
        self._samples: list[BenchmarkSample] = []
        self._loaded = False

    @property
    def name(self) -> str:
        return f"beir-{'+'.join(self._tasks)}"

    @property
    def description(self) -> str:
        return f"BEIR zero-shot transfer: {', '.join(self._tasks)}"

    def categories(self) -> list[str]:
        return list(self._tasks)

    def load(self, data_dir: str | None = None) -> None:
        if self._loaded:
            return

        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Install 'datasets': pip install datasets")

        for task in self._tasks:
            self._load_beir_task(task)

        self._loaded = True

    def _load_beir_task(self, task: str) -> None:
        from datasets import load_dataset

        # BEIR datasets on HuggingFace (mteb/ org has script-free versions)
        hf_name = f"mteb/{task}"

        try:
            # Load corpus
            corpus_ds = load_dataset(hf_name, "corpus", split="corpus")
            for row in corpus_ds:
                doc_id = f"{task}_{row['_id']}"
                title = row.get("title", "")
                text = row.get("text", "")
                full_text = f"{title}\n{text}" if title else text

                self._corpus.append({
                    "id": doc_id,
                    "text": full_text,
                    "metadata": {"task": task, "title": title},
                })

            # Load queries and qrels
            queries_ds = load_dataset(hf_name, "queries", split="queries")
            query_map = {}
            for row in queries_ds:
                query_map[row["_id"]] = row["text"]

            # Load test qrels
            try:
                qrels_ds = load_dataset(hf_name, "default", split="test")
            except Exception:
                qrels_ds = load_dataset(hf_name, split="test")

            # Group by query
            qrel_groups: dict[str, list[str]] = {}
            for row in qrels_ds:
                qid = str(row.get("query-id", row.get("query_id", "")))
                did = str(row.get("corpus-id", row.get("corpus_id", "")))
                score = row.get("score", 1)
                if score > 0:
                    qrel_groups.setdefault(qid, []).append(f"{task}_{did}")

            for qid, relevant_ids in qrel_groups.items():
                if qid in query_map:
                    self._samples.append(BenchmarkSample(
                        question_id=f"{task}_{qid}",
                        question=query_map[qid],
                        ground_truth="",  # BEIR uses retrieval metrics, not answer generation
                        category=task,
                        corpus_doc_ids=relevant_ids,
                    ))

        except Exception as e:
            print(f"Warning: failed to load BEIR task '{task}': {e}")

    def corpus(self) -> list[dict]:
        return self._corpus

    def samples(self) -> list[BenchmarkSample]:
        return self._samples
