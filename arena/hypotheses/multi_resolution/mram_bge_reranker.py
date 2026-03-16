"""MRAM + BGE reranker: sentence-level retrieval with stronger reranker.

Tests whether MRAM's sentence-level candidate expansion composes with
a stronger reranker (BGE-v2-m3, 568M params) vs the default CE L-12.
If MRAM+BGE > max(MRAM, BGE alone), the architecture and reranker
are complementary.
"""

from .multi_resolution import MultiResolutionHypothesis


class MRAMBGERerankerHypothesis(MultiResolutionHypothesis):
    """MRAM Phase 1.5 with BGE-v2-m3 reranker instead of CE L-12."""

    def __init__(self):
        super().__init__(model_name="BAAI/bge-reranker-v2-m3")

    @property
    def name(self):
        return "mram-bge-reranker"

    @property
    def description(self):
        return (
            f"MRAM sentence+passage retrieval, "
            f"merge {self._pool_per_level}×2 candidates, "
            f"BGE-v2-m3 rerank to {self._final_k}"
        )
