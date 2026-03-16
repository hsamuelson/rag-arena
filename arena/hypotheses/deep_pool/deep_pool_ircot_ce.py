"""Deep Pool + IRCoT: the kitchen sink approach.

Combines our two best strategies:
1. Deep pool (proven +6.1%)
2. IRCoT reasoning (published +8-15%)

Deep pool 30 for original query + IRCoT follow-up retrieval → merged → CE.
"""

import json
import urllib.request
import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 200) -> str:
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


class DeepPoolIRCoTCEHypothesis(Hypothesis):
    """Deep pool + IRCoT: deep retrieve, reason, deep retrieve again, CE rerank."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 pool_size=30, final_k=10):
        self._model_name = model_name
        self._pool_size = pool_size
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
        return "deep-pool-ircot-ce"

    @property
    def description(self):
        return f"Deep pool {self._pool_size} + IRCoT reasoning + CE rerank"

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        model = self._get_model()
        all_results = {}

        # Step 1: Deep retrieval for original query
        try:
            deep1, _ = self._backend.retrieve_with_embeddings(query, self._pool_size)
            for r in deep1:
                all_results[r.doc_id] = r
        except Exception:
            deep1 = results

        # Step 2: CE rank to find top-1 for reasoning
        if deep1:
            pairs = [(query, r.text[:_MAX_CHARS]) for r in deep1]
            scores = model.predict(pairs).tolist()
            top_idx = max(range(len(scores)), key=lambda i: scores[i])
            top_doc = deep1[top_idx]
        else:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Step 3: LLM reasoning to generate follow-up
        prompt = (
            "You are answering a multi-hop question. Based on the question and first "
            "document, generate a SHORT search query to find the missing information. "
            "Return ONLY the search query.\n\n"
            f"Question: {query}\n"
            f"Document: {top_doc.text[:800]}\n\n"
            "Follow-up search query:"
        )
        try:
            followup = _call_llm(prompt, max_tokens=100).strip().strip('"\'')
        except Exception:
            followup = query

        # Step 4: Deep retrieval for follow-up
        try:
            deep2, _ = self._backend.retrieve_with_embeddings(followup, self._pool_size)
            for r in deep2:
                if r.doc_id not in all_results:
                    all_results[r.doc_id] = r
        except Exception:
            pass

        # Step 5: CE rerank merged pool
        merged = list(all_results.values())
        return self._ce_rerank(query, merged, followup=followup)

    def _ce_rerank(self, query, results, followup=None):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (deep+IRCoT, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "followup": followup or ""},
        )
