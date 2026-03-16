"""Abstract backend interface for retrieval systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieved document/chunk."""
    doc_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract retrieval backend.

    Backends are responsible for:
    1. Ingesting a corpus (documents with IDs)
    2. Retrieving top-K results for a query
    3. Optionally returning raw embeddings for hypothesis post-processing
    """

    @abstractmethod
    def ingest(self, documents: list[dict]) -> None:
        """Ingest documents into the backend.

        Each document: {"id": str, "text": str, "metadata": dict}
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve top-K documents for a query."""

    @abstractmethod
    def retrieve_with_embeddings(
        self, query: str, top_k: int = 10
    ) -> tuple[list[RetrievalResult], "np.ndarray | None"]:
        """Retrieve top-K documents and their embedding vectors.

        Returns (results, embeddings_matrix) where embeddings_matrix is
        shape (K, D) or None if the backend doesn't support raw embeddings.
        """

    @abstractmethod
    def embed_query(self, query: str) -> "np.ndarray":
        """Embed a single query text. Returns (D,) array."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> "np.ndarray":
        """Embed a batch of texts. Returns (N, D) array."""

    def clear(self) -> None:
        """Clear all ingested data. Optional."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
