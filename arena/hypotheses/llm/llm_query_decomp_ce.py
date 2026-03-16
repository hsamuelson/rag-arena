"""LLM Query Decomposition + CE reranking.

Uses Qwen 3.5 122B to decompose multi-hop questions into sub-questions,
retrieves for each sub-question independently, merges candidates, then
CE reranks the combined pool.

Research basis: DecompRC (Min et al., 2019), Self-Ask (Press et al., 2022).
Published: +3-11 F1 on multi-hop. Our heuristic attempt HURT (-4.5% to -6.0%)
because regex parsing is too poor. LLM decomposition should fix this.
"""

import json
import urllib.request
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call Qwen 3.5 via Ollama, handling <think> tags."""
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

    # Strip thinking tokens
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


class LLMQueryDecompCEHypothesis(Hypothesis):
    """Decompose multi-hop questions via LLM, retrieve per sub-question, CE rerank."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        final_k: int = 10,
    ):
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
    def name(self) -> str:
        return "llm-query-decomp-ce"

    @property
    def description(self) -> str:
        return "LLM decomposes multi-hop Q into sub-Qs, retrieve for each, CE rerank merged pool"

    def _decompose_question(self, query: str) -> list[str]:
        """Use LLM to decompose a multi-hop question into sub-questions."""
        prompt = (
            "Break this question into 2-3 simple sub-questions that can be answered "
            "independently. Return ONLY the sub-questions, one per line. No numbering, "
            "no explanation.\n\n"
            f"Question: {query}"
        )
        try:
            response = _call_llm(prompt, max_tokens=300)
            sub_qs = [q.strip().lstrip("0123456789.-) ") for q in response.strip().split("\n") if q.strip()]
            # Filter out empty or too-short sub-questions
            sub_qs = [q for q in sub_qs if len(q) > 10]
            if not sub_qs:
                return [query]
            return sub_qs
        except Exception:
            return [query]

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            # Fall back to plain CE if no backend
            return self._ce_rerank(query, results)

        # Decompose question
        sub_questions = self._decompose_question(query)

        # Retrieve for each sub-question
        all_results = {}
        for sq in sub_questions:
            try:
                sq_results, _ = self._backend.retrieve_with_embeddings(sq, 15)
                for r in sq_results:
                    if r.doc_id not in all_results:
                        all_results[r.doc_id] = r
            except Exception:
                pass

        # Also include original query results
        for r in results:
            if r.doc_id not in all_results:
                all_results[r.doc_id] = r

        merged = list(all_results.values())
        return self._ce_rerank(query, merged)

    def _ce_rerank(self, query, results):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[: self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (LLM decomp, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "n_sub_questions": len(self._decompose_question.__code__.co_varnames)},
        )
