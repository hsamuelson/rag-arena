"""HotpotQA benchmark — multi-hop question answering.

Specifically tests retrieval that requires connecting multiple documents.
113K questions in the fullwiki setting over ~5.2M Wikipedia abstracts.

Standard baselines (BEIR nDCG@10):
  BM25:  0.603
  DPR:   0.391

Source: hotpotqa/hotpot_qa (HuggingFace), BeIR/hotpotqa
"""

from pathlib import Path

from .base import Benchmark, BenchmarkSample


HOTPOTQA_BASELINES = {
    "BM25": {"ndcg_10": 0.603},
    "DPR": {"ndcg_10": 0.391},
}


class HotpotQABenchmark(Benchmark):
    """HotpotQA multi-hop QA benchmark."""

    def __init__(self, split: str = "validation", max_samples: int | None = None):
        self._corpus: list[dict] = []
        self._samples: list[BenchmarkSample] = []
        self._loaded = False
        self._split = split
        self._max_samples = max_samples

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def description(self) -> str:
        return "HotpotQA — multi-hop questions requiring 2+ supporting documents"

    def categories(self) -> list[str]:
        return ["bridge", "comparison"]

    def load(self, data_dir: str | None = None) -> None:
        if self._loaded:
            return
        self._load_from_huggingface()
        self._loaded = True

    def _load_from_huggingface(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Install 'datasets': pip install datasets")

        ds = load_dataset(
            "hotpotqa/hotpot_qa", "distractor",
            split=self._split,
        )

        seen_docs: set[str] = set()

        limit = self._max_samples or len(ds)
        for idx, row in enumerate(ds):
            if idx >= limit:
                break

            question = row["question"]
            answer = row["answer"]
            q_type = row.get("type", "bridge")
            level = row.get("level", "")

            # Build corpus from context paragraphs
            supporting_ids = []
            context_titles = row.get("context", {}).get("title", [])
            context_sentences = row.get("context", {}).get("sentences", [])

            for title, sentences in zip(context_titles, context_sentences):
                doc_id = f"hotpot_{title}"
                text = " ".join(sentences)

                if doc_id not in seen_docs:
                    self._corpus.append({
                        "id": doc_id,
                        "text": text,
                        "metadata": {"title": title},
                    })
                    seen_docs.add(doc_id)

            # Supporting facts identify which paragraphs are needed
            sf_titles = row.get("supporting_facts", {}).get("title", [])
            for sf_title in sf_titles:
                sf_id = f"hotpot_{sf_title}"
                if sf_id not in supporting_ids:
                    supporting_ids.append(sf_id)

            self._samples.append(BenchmarkSample(
                question_id=f"hotpot_{idx}",
                question=question,
                ground_truth=answer,
                category=q_type,
                corpus_doc_ids=supporting_ids,
                metadata={"level": level, "type": q_type},
            ))

    def corpus(self) -> list[dict]:
        return self._corpus

    def samples(self) -> list[BenchmarkSample]:
        return self._samples
