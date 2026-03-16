"""LLM pointwise reranker: score each document independently with LLM.

CE pre-filters top-50 to top-20, then qwen3.5:122b scores each
document independently on a 3-level relevance scale. Unlike listwise,
each document is judged without seeing the others.

20 LLM calls per query — slower but avoids position bias in listwise.
"""

import json
import re
import urllib.request

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"

# Relevance scale
_RELEVANCE_SCORES = {
    "highly relevant": 3.0,
    "somewhat relevant": 2.0,
    "not relevant": 1.0,
}


def _call_llm(prompt: str, max_tokens: int = 100) -> str:
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


class LLMPointwiseRerankerHypothesis(Hypothesis):
    """CE top-50 → top-20, then LLM pointwise scoring per document."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 pre_filter_k=20, final_k=10):
        self._model_name = model_name
        self._pool_size = 50
        self._pre_filter_k = pre_filter_k
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
        return "llm-pointwise-reranker"

    @property
    def description(self):
        return f"CE top-50 → top-{self._pre_filter_k}, LLM pointwise scoring to top-{self._final_k}"

    def _score_document(self, query: str, doc_text: str) -> float:
        prompt = (
            "Rate how relevant this document is to the question.\n"
            "Answer with EXACTLY one of: Highly Relevant, Somewhat Relevant, Not Relevant\n\n"
            f"Question: {query}\n\n"
            f"Document: {doc_text[:500]}\n\n"
            "Relevance:"
        )

        try:
            response = _call_llm(prompt, max_tokens=20)
            response_lower = response.strip().lower()
            for label, score in _RELEVANCE_SCORES.items():
                if label in response_lower:
                    return score
            # Try to extract any relevance signal
            if "highly" in response_lower or "very" in response_lower:
                return 3.0
            elif "somewhat" in response_lower or "partial" in response_lower:
                return 2.0
            return 1.0
        except Exception:
            return 0.0  # LLM failure → lowest score

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, self._pool_size)
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Stage 1: CE pre-filter to top-20
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()
        ce_ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        ce_top = [results[i] for i in ce_ranked[:self._pre_filter_k]]
        ce_top_scores = [ce_scores[i] for i in ce_ranked[:self._pre_filter_k]]

        # Stage 2: LLM pointwise scoring (20 calls)
        llm_scores = []
        for doc in ce_top:
            score = self._score_document(query, doc.text)
            llm_scores.append(score)

        # Combine: use LLM score as primary, CE score as tiebreaker
        combined = []
        for i in range(len(ce_top)):
            combined.append((llm_scores[i], ce_top_scores[i], i))

        combined.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_indices = [c[2] for c in combined[:self._final_k]]
        reranked = [ce_top[i] for i in top_indices]

        lines = [f"Retrieved context (CE + LLM pointwise, {len(results)} → {self._pre_filter_k} → {len(reranked)}):"]
        for rank, i in enumerate(top_indices, 1):
            label = "Highly" if llm_scores[i] >= 3 else "Somewhat" if llm_scores[i] >= 2 else "Not"
            lines.append(f"\n[{rank}] (llm={label}, ce={ce_top_scores[i]:.4f})")
            lines.append(ce_top[i].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "llm_scores": [llm_scores[i] for i in top_indices],
                "ce_scores": [ce_top_scores[i] for i in top_indices],
                "pool_size": len(results),
                "pre_filter_k": self._pre_filter_k,
            },
        )
