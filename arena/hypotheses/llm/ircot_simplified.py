"""Simplified IRCoT: Interleaving Retrieval with Chain-of-Thought.

Based on Trivedi et al., ACL 2023. Published: +21 recall points, +7.1 F1 on HotpotQA.

Simplified version (no full CoT loop):
1. Retrieve top-15 for original query
2. CE rerank → take top-1
3. LLM reads top-1 doc and generates a follow-up retrieval query
4. Retrieve top-15 for follow-up query
5. Merge pools → CE rerank to top-10

This captures the core insight: use intermediate reasoning to find bridge documents.
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


class IRCoTSimplifiedHypothesis(Hypothesis):
    """Simplified IRCoT: retrieve, reason with LLM, retrieve again, CE rerank."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        first_pool: int = 15,
        second_pool: int = 15,
        final_k: int = 10,
    ):
        self._model_name = model_name
        self._first_pool = first_pool
        self._second_pool = second_pool
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
        return "ircot-simplified"

    @property
    def description(self) -> str:
        return "Retrieve → CE top-1 → LLM reasoning → retrieve again → CE rerank merged pool"

    def _generate_followup_query(self, question: str, top_doc_text: str) -> str:
        """Use LLM to generate a follow-up retrieval query based on initial findings."""
        prompt = (
            "You are helping answer a multi-hop question. Based on the question and "
            "the first document found, generate a SHORT follow-up search query to find "
            "the missing information. Return ONLY the search query, nothing else.\n\n"
            f"Question: {question}\n\n"
            f"First document found:\n{top_doc_text[:1000]}\n\n"
            "Follow-up search query:"
        )
        try:
            return _call_llm(prompt, max_tokens=100).strip().strip('"\'')
        except Exception:
            return question  # fall back to original

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        # Step 1: First retrieval (may already be done, but we want deeper pool)
        try:
            first_results, _ = self._backend.retrieve_with_embeddings(query, self._first_pool)
        except Exception:
            first_results = results

        # Step 2: Quick CE to find top-1
        model = self._get_model()
        if first_results:
            pairs = [(query, r.text[:_MAX_CHARS]) for r in first_results]
            scores = model.predict(pairs).tolist()
            top_idx = max(range(len(scores)), key=lambda i: scores[i])
            top_doc = first_results[top_idx]
        else:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Step 3: LLM generates follow-up query
        followup_query = self._generate_followup_query(query, top_doc.text)

        # Step 4: Second retrieval with follow-up query
        try:
            second_results, _ = self._backend.retrieve_with_embeddings(followup_query, self._second_pool)
        except Exception:
            second_results = []

        # Step 5: Merge pools (deduplicate by doc_id)
        all_results = {}
        for r in first_results:
            all_results[r.doc_id] = r
        for r in second_results:
            if r.doc_id not in all_results:
                all_results[r.doc_id] = r

        merged = list(all_results.values())

        # Step 6: Final CE rerank on merged pool using ORIGINAL query
        return self._ce_rerank(query, merged, followup_query=followup_query)

    def _ce_rerank(self, query, results, followup_query=None):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[: self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (IRCoT, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "pool_size": len(results),
                "followup_query": followup_query or "",
            },
        )
