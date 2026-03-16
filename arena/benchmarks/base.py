"""Abstract benchmark interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BenchmarkSample:
    """A single benchmark question with context."""
    question_id: str
    question: str
    ground_truth: str
    category: str                          # e.g. "single-hop", "multi-hop", "temporal"
    corpus_doc_ids: list[str] = field(default_factory=list)  # IDs of relevant docs
    metadata: dict = field(default_factory=dict)


class Benchmark(ABC):
    """Abstract benchmark that provides corpus + questions."""

    @abstractmethod
    def load(self, data_dir: str | None = None) -> None:
        """Load or download the benchmark data."""

    @abstractmethod
    def corpus(self) -> list[dict]:
        """Return corpus documents as [{"id": str, "text": str, "metadata": dict}]."""

    @abstractmethod
    def samples(self) -> list[BenchmarkSample]:
        """Return evaluation samples (questions + ground truth)."""

    @abstractmethod
    def categories(self) -> list[str]:
        """Return the list of question categories in this benchmark."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""

    @property
    def description(self) -> str:
        """Short description of what this benchmark tests."""
        return ""
