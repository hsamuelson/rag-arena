"""LLM Query Expansion + Deep Pool CE.

Use LLM to generate a search-optimized reformulation of the query,
then retrieve with BOTH original and expanded queries, merge, CE rerank.

Different from decomposition: this generates a SINGLE better query,
not sub-questions. Focuses on adding keywords and context.
"""

import json
import urllib.request
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 150) -> str:
    data = json.dumps({
        "model": "qwen3.5:122b",
        "prompt": f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
        "stream": False,
        "raw": True,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
    }).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    output = result.get("response", "")
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


class LLMQueryExpansionCEHypothesis(Hypothesis):
    """LLM generates expanded query, retrieve with both, CE rerank merged pool."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", final_k=10):
        self._model_name = model_name
        self._final_k = final_k
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return "llm-query-expansion-ce"

    @property
    def description(self):
        return "LLM expands query with keywords, retrieve with both queries, CE rerank"

    def _expand_query(self, query: str) -> str:
        prompt = (
            "Rewrite this question as a search-optimized query. Add relevant keywords "
            "and related terms that would help find the answer. Return ONLY the expanded "
            "query, nothing else.\n\n"
            f"Question: {query}\n\n"
            "Expanded search query:"
        )
        try:
            expanded = _call_llm(prompt, max_tokens=100).strip().strip('"\'')
            if len(expanded) < 5:
                return query
            return expanded
        except Exception:
            return query

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        # Expand query
        expanded = self._expand_query(query)

        # Retrieve with both queries
        all_results = {}
        for q in [query, expanded]:
            try:
                q_results, _ = self._backend.retrieve_with_embeddings(q, 20)
                for r in q_results:
                    if r.doc_id not in all_results:
                        all_results[r.doc_id] = r
            except Exception:
                pass

        # Also include original results
        for r in results:
            if r.doc_id not in all_results:
                all_results[r.doc_id] = r

        merged = list(all_results.values())
        return self._ce_rerank(query, merged, expanded_query=expanded)

    def _ce_rerank(self, query, results, expanded_query=None):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (LLM expanded, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "expanded_query": expanded_query or ""},
        )
