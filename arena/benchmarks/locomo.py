"""LoCoMo benchmark — Long-Context Conversational Memory.

Tests retrieval across 5 categories:
  1. single-hop:   Direct fact lookup
  2. temporal:     Time-aware retrieval
  3. open-domain:  Broad knowledge questions
  4. multi-hop:    Requires connecting multiple pieces
  5. adversarial:  Designed to trick retrieval systems

Source: https://huggingface.co/datasets/LoCoMo/LoCoMo
"""

import json
from pathlib import Path

from .base import Benchmark, BenchmarkSample

CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "open-domain",
    4: "multi-hop",
    5: "adversarial",
}


class LoCoMoBenchmark(Benchmark):
    """LoCoMo benchmark for conversational memory evaluation."""

    def __init__(self):
        self._corpus: list[dict] = []
        self._samples: list[BenchmarkSample] = []
        self._loaded = False

    @property
    def name(self) -> str:
        return "locomo"

    @property
    def description(self) -> str:
        return "Long-Context Conversational Memory — 5 categories testing single-hop through adversarial retrieval"

    def categories(self) -> list[str]:
        return list(CATEGORY_NAMES.values())

    def load(self, data_dir: str | None = None) -> None:
        """Load LoCoMo from local data dir or download via HuggingFace datasets."""
        if self._loaded:
            return

        local_path = Path(data_dir) / "locomo" if data_dir else None

        if local_path and (local_path / "locomo.json").exists():
            self._load_from_local(local_path / "locomo.json")
        else:
            self._load_from_huggingface(local_path)

        self._loaded = True

    def _load_from_local(self, path: Path) -> None:
        """Load from a local JSON file (same format as Caldera's experiments)."""
        with open(path) as f:
            data = json.load(f)

        for session in data:
            session_id = str(session.get("session_id", ""))

            # Build corpus from conversation
            conversation = session.get("conversation", [])
            for i, turn in enumerate(conversation):
                text = turn.get("text", turn.get("content", ""))
                doc_id = f"{session_id}_turn_{i}"
                self._corpus.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "session_id": session_id,
                        "turn_index": i,
                        "speaker": turn.get("speaker", turn.get("role", "")),
                    },
                })

            # Build samples from QA pairs
            for qa in session.get("qa_pairs", []):
                cat_id = qa.get("category", 1)
                category = CATEGORY_NAMES.get(cat_id, f"category-{cat_id}")
                question_id = f"{session_id}_q{qa.get('id', '')}"

                # Relevant doc IDs from evidence indices
                relevant_ids = []
                for ev in qa.get("evidence", []):
                    if isinstance(ev, int):
                        relevant_ids.append(f"{session_id}_turn_{ev}")
                    elif isinstance(ev, dict) and "turn_index" in ev:
                        relevant_ids.append(f"{session_id}_turn_{ev['turn_index']}")

                self._samples.append(BenchmarkSample(
                    question_id=question_id,
                    question=qa.get("question", ""),
                    ground_truth=qa.get("answer", ""),
                    category=category,
                    corpus_doc_ids=relevant_ids,
                    metadata={"session_id": session_id},
                ))

    def _load_from_huggingface(self, cache_dir: Path | None = None) -> None:
        """Load from HuggingFace datasets library."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "Install 'datasets' package to download LoCoMo: pip install datasets"
            )

        ds = load_dataset("LoCoMo/LoCoMo", split="test")

        for idx, row in enumerate(ds):
            session_id = str(row.get("session_id", idx))

            # Parse conversation
            conversation = row.get("conversation", [])
            if isinstance(conversation, str):
                conversation = json.loads(conversation)

            for i, turn in enumerate(conversation):
                text = turn.get("text", turn.get("content", ""))
                doc_id = f"{session_id}_turn_{i}"
                self._corpus.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {
                        "session_id": session_id,
                        "turn_index": i,
                    },
                })

            # Parse QA pairs
            qa_pairs = row.get("qa_pairs", [])
            if isinstance(qa_pairs, str):
                qa_pairs = json.loads(qa_pairs)

            for qa in qa_pairs:
                cat_id = qa.get("category", 1)
                category = CATEGORY_NAMES.get(cat_id, f"category-{cat_id}")
                question_id = f"{session_id}_q{qa.get('id', '')}"

                relevant_ids = []
                for ev in qa.get("evidence", []):
                    if isinstance(ev, int):
                        relevant_ids.append(f"{session_id}_turn_{ev}")

                self._samples.append(BenchmarkSample(
                    question_id=question_id,
                    question=qa["question"],
                    ground_truth=qa["answer"],
                    category=category,
                    corpus_doc_ids=relevant_ids,
                    metadata={"session_id": session_id},
                ))

        # Cache locally
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / "locomo.json", "w") as f:
                json.dump(
                    [{"corpus": self._corpus, "samples": [s.__dict__ for s in self._samples]}],
                    f,
                )

    def corpus(self) -> list[dict]:
        return self._corpus

    def samples(self) -> list[BenchmarkSample]:
        return self._samples
