"""Full IRCoT: 2-hop interleaved retrieval with chain-of-thought.

Based on Trivedi et al., ACL 2023.
Retrieve → LLM CoT → Retrieve again → LLM CoT → Final CE rerank.

This is the full 2-hop version (vs simplified 1-hop in ircot_simplified.py).
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
    resp = urllib.request.urlopen(req, timeout=180)
    result = json.loads(resp.read())
    output = result.get("response", "")
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


class IRCoTFullHypothesis(Hypothesis):
    """Full 2-hop IRCoT: retrieve → reason → retrieve → reason → CE rerank."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 retrieve_k=15, final_k=10):
        self._model_name = model_name
        self._retrieve_k = retrieve_k
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
        return "ircot-full-2hop"

    @property
    def description(self):
        return "Full IRCoT: retrieve → CoT → retrieve → CoT → CE rerank"

    def _cot_step(self, question: str, docs: list[RetrievalResult], hop: int) -> str:
        """Generate chain-of-thought reasoning and a follow-up query."""
        doc_texts = "\n\n".join(
            f"Document {i+1}: {d.text[:500]}" for i, d in enumerate(docs[:3])
        )
        prompt = (
            f"You are answering this question step by step: {question}\n\n"
            f"Here are documents found so far:\n{doc_texts}\n\n"
            f"This is hop {hop} of reasoning. Based on what you've found, "
            "what information is still missing? Generate a SHORT search query "
            "to find the missing information. Return ONLY the search query."
        )
        try:
            return _call_llm(prompt, max_tokens=100).strip().strip('"\'')
        except Exception:
            return question

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        all_results = {}
        model = self._get_model()

        # Hop 1: Initial retrieval
        try:
            hop1_results, _ = self._backend.retrieve_with_embeddings(query, self._retrieve_k)
        except Exception:
            hop1_results = results

        for r in hop1_results:
            all_results[r.doc_id] = r

        # CE rank hop 1 to get best docs for reasoning
        if hop1_results:
            pairs = [(query, r.text[:_MAX_CHARS]) for r in hop1_results]
            scores = model.predict(pairs).tolist()
            best_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            best_docs = [hop1_results[i] for i in best_indices]
        else:
            best_docs = []

        # CoT step 1: Generate follow-up query
        followup1 = self._cot_step(query, best_docs, hop=1)

        # Hop 2: Retrieve with follow-up
        try:
            hop2_results, _ = self._backend.retrieve_with_embeddings(followup1, self._retrieve_k)
            for r in hop2_results:
                if r.doc_id not in all_results:
                    all_results[r.doc_id] = r
        except Exception:
            pass

        # CE rank hop 2
        hop2_list = [r for r in all_results.values() if r.doc_id not in {r.doc_id for r in hop1_results}]
        all_best = list(best_docs)
        if hop2_list:
            pairs2 = [(query, r.text[:_MAX_CHARS]) for r in hop2_list]
            scores2 = model.predict(pairs2).tolist()
            best2 = sorted(range(len(scores2)), key=lambda i: scores2[i], reverse=True)[:3]
            all_best.extend([hop2_list[i] for i in best2])

        # CoT step 2: Generate another follow-up
        followup2 = self._cot_step(query, all_best, hop=2)

        # Hop 3: Final retrieval with second follow-up
        try:
            hop3_results, _ = self._backend.retrieve_with_embeddings(followup2, self._retrieve_k)
            for r in hop3_results:
                if r.doc_id not in all_results:
                    all_results[r.doc_id] = r
        except Exception:
            pass

        # Final CE rerank on full merged pool
        merged = list(all_results.values())
        return self._ce_rerank(query, merged, hops=[followup1, followup2])

    def _ce_rerank(self, query, results, hops=None):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (IRCoT 2-hop, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "hop_queries": hops or []},
        )
