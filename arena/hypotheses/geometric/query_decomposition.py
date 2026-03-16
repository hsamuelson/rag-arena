"""Query decomposition for multi-hop retrieval.

Hypothesis: Complex questions that require multiple pieces of information
fail with single-query retrieval because the query embedding averages
across all required facts, landing in a "no man's land" between the
relevant document clusters. Decomposing into atomic sub-queries and
retrieving for each produces better coverage.

At scale this is critical: the larger the corpus, the more the single
query embedding gets lost in the crowd. Sub-queries are more specific
and penetrate deeper into relevant clusters.

Algorithm:
1. Use LLM to decompose query into 2-4 atomic sub-queries
2. Retrieve top-K/n for each sub-query (where n = number of sub-queries)
3. Merge results via weighted RRF
4. Present with sub-query provenance

References:
  - A-RAG (Feb 2026): Hierarchical retrieval interfaces
  - IRCoT (2023): Interleaving retrieval with chain-of-thought
  - DecompRC (2019): Multi-hop reading via decomposition
"""

import numpy as np
import requests

from ..base import Hypothesis, HypothesisResult
from ...backends.base import Backend, RetrievalResult


class QueryDecompositionHypothesis(Hypothesis):
    """Decompose complex queries into atomic sub-queries for broader retrieval."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        chat_model: str = "qwen3.5:122b",
        backend: Backend | None = None,
        top_k: int = 10,
    ):
        self._ollama_url = ollama_url
        self._chat_model = chat_model
        self._backend = backend
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "query-decomposition"

    @property
    def description(self) -> str:
        return (
            "Query decomposition — break complex queries into atomic sub-queries, "
            "retrieve for each, merge via RRF for multi-hop coverage"
        )

    def apply(
        self,
        query: str,
        results: list[RetrievalResult],
        embeddings: np.ndarray | None,
        query_embedding: np.ndarray | None,
    ) -> HypothesisResult:
        if self._backend is None:
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results, []),
                metadata={"fallback": True, "reason": "no backend for sub-query retrieval"},
            )

        # Decompose query
        sub_queries = self._decompose(query)

        if len(sub_queries) <= 1:
            # Query was already atomic
            return HypothesisResult(
                results=results,
                context_prompt=self._format(results, [query]),
                metadata={"sub_queries": [query], "decomposed": False},
            )

        # Retrieve for each sub-query
        per_k = max(3, self._top_k // len(sub_queries))
        all_sub_results: list[tuple[str, list[RetrievalResult]]] = []

        for sq in sub_queries:
            sub_results = self._backend.retrieve(sq, per_k)
            all_sub_results.append((sq, sub_results))

        # Merge via weighted RRF
        merged = self._merge_rrf(all_sub_results)

        return HypothesisResult(
            results=merged,
            context_prompt=self._format(merged, sub_queries),
            metadata={
                "sub_queries": sub_queries,
                "decomposed": True,
                "results_per_subquery": [len(sr) for _, sr in all_sub_results],
                "merged_count": len(merged),
            },
        )

    def _decompose(self, query: str) -> list[str]:
        """Use LLM to decompose a query into atomic sub-queries."""
        prompt = (
            "Break this question into 2-4 simpler, independent sub-questions "
            "that together answer the original. Each sub-question should be "
            "self-contained and answerable from a single document.\n\n"
            "Return ONLY the sub-questions, one per line, with no numbering "
            "or extra text.\n\n"
            f"Question: {query}"
        )

        try:
            resp = requests.post(
                f"{self._ollama_url}/api/chat",
                json={
                    "model": self._chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"].strip()
            sub_queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in text.split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return sub_queries if sub_queries else [query]
        except Exception:
            return [query]

    def _merge_rrf(
        self,
        sub_results: list[tuple[str, list[RetrievalResult]]],
    ) -> list[RetrievalResult]:
        """Merge sub-query results via Reciprocal Rank Fusion."""
        k = 60
        score_map: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}
        n_subqueries = len(sub_results)
        weight = 1.0 / n_subqueries

        for _, results in sub_results:
            for rank, r in enumerate(results, 1):
                score_map[r.doc_id] = score_map.get(r.doc_id, 0) + weight / (rank + k)
                if r.doc_id not in result_map:
                    result_map[r.doc_id] = r

        sorted_ids = sorted(score_map, key=lambda x: score_map[x], reverse=True)
        return [
            RetrievalResult(
                doc_id=did,
                text=result_map[did].text,
                score=score_map[did],
                metadata=result_map[did].metadata,
            )
            for did in sorted_ids[:self._top_k]
        ]

    def _format(self, results: list[RetrievalResult], sub_queries: list[str]) -> str:
        lines = []
        if len(sub_queries) > 1:
            lines.append("Query decomposed into:")
            for i, sq in enumerate(sub_queries, 1):
                lines.append(f"  {i}. {sq}")
            lines.append("")
        lines.append("Retrieved context (multi-query merged):")
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] (score: {r.score:.3f})")
            lines.append(r.text)
        return "\n".join(lines)
