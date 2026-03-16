"""LLM listwise reranker (RankGPT-style).

CE pre-filters top-50 to top-20, then qwen3.5:122b ranks them in a
single prompt. The LLM sees all 20 documents at once and outputs
a permutation. Falls back to CE ranking on LLM failure.

Reference: Sun et al., "Is ChatGPT Good at Search?" (RankGPT, 2023).
"""

import json
import re
import urllib.request

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 500) -> str:
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
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    output = result.get("response", "")
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


class LLMListwiseRerankerHypothesis(Hypothesis):
    """CE top-50 → top-20, then LLM listwise reranking."""

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
        return "llm-listwise-reranker"

    @property
    def description(self):
        return f"CE top-50 → top-{self._pre_filter_k}, LLM listwise rerank to top-{self._final_k}"

    def _llm_rerank(self, query: str, docs: list[RetrievalResult]) -> list[int]:
        doc_list = []
        for i, d in enumerate(docs):
            doc_list.append(f"[{i+1}] {d.text[:400]}")

        prompt = (
            "Rank these documents by relevance to the question. "
            "Return ONLY the document numbers in order from most to least relevant, "
            "separated by commas. Example: 3, 1, 5, 2, 4\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n" + "\n".join(doc_list) + "\n\n"
            "Ranking (most relevant first):"
        )

        try:
            response = _call_llm(prompt, max_tokens=200)
            numbers = re.findall(r'\d+', response)
            ranking = []
            seen = set()
            for n in numbers:
                idx = int(n) - 1
                if 0 <= idx < len(docs) and idx not in seen:
                    ranking.append(idx)
                    seen.add(idx)
            for i in range(len(docs)):
                if i not in seen:
                    ranking.append(i)
            return ranking
        except Exception:
            return list(range(len(docs)))

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

        # Stage 2: LLM listwise rerank
        llm_ranking = self._llm_rerank(query, ce_top)
        reranked = [ce_top[i] for i in llm_ranking[:self._final_k]]

        lines = [f"Retrieved context (CE + LLM listwise, {len(results)} → {self._pre_filter_k} → {len(reranked)}):"]
        for i, doc in enumerate(reranked, 1):
            lines.append(f"\n[{i}]")
            lines.append(doc.text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "llm_ranking": llm_ranking[:self._final_k],
                "pool_size": len(results),
                "pre_filter_k": self._pre_filter_k,
            },
        )
