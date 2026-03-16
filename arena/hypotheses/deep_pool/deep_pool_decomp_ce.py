"""Deep Pool + LLM Decomposition: combine both proven approaches.

Deep pool (+6.1%) is our best finding. LLM decomposition targets multi-hop.
Combine: decompose → retrieve per sub-question with deep pools → merge → CE.
"""

import json
import urllib.request
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
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


class DeepPoolDecompCEHypothesis(Hypothesis):
    """LLM decompose + deep pool per sub-question + CE rerank."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 pool_per_query=20, final_k=10):
        self._model_name = model_name
        self._pool_per_query = pool_per_query
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
        return "deep-pool-decomp-ce"

    @property
    def description(self):
        return f"LLM decompose + deep pool {self._pool_per_query}/sub-Q + CE rerank"

    def _decompose(self, query: str) -> list[str]:
        prompt = (
            "Break this question into 2-3 simple sub-questions. "
            "Return ONLY the sub-questions, one per line.\n\n"
            f"Question: {query}"
        )
        try:
            response = _call_llm(prompt, max_tokens=300)
            sub_qs = [q.strip().lstrip("0123456789.-) ") for q in response.strip().split("\n") if q.strip()]
            sub_qs = [q for q in sub_qs if len(q) > 10]
            return sub_qs if sub_qs else [query]
        except Exception:
            return [query]

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        # Decompose
        sub_questions = self._decompose(query)

        # Deep retrieve for each sub-question + original
        all_results = {}
        for q in [query] + sub_questions:
            try:
                q_results, _ = self._backend.retrieve_with_embeddings(q, self._pool_per_query)
                for r in q_results:
                    if r.doc_id not in all_results:
                        all_results[r.doc_id] = r
            except Exception:
                pass

        merged = list(all_results.values())
        return self._ce_rerank(query, merged)

    def _ce_rerank(self, query, results):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (decomp+deep, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results)},
        )
