"""LLM Query Fusion: generate multiple query perspectives, retrieve for each.

Based on RAG-Fusion (Raudaschl, 2023). LLM generates 3 different query
formulations, retrieve for each, merge via RRF, then CE rerank.
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
        "options": {"num_predict": max_tokens, "temperature": 0.3},
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


class LLMQueryFusionCEHypothesis(Hypothesis):
    """LLM generates 3 query variants, retrieve for each, RRF merge, CE rerank."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 n_variants=3, retrieve_k=15, final_k=10):
        self._model_name = model_name
        self._n_variants = n_variants
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
        return "llm-query-fusion-ce"

    @property
    def description(self):
        return f"LLM generates {self._n_variants} query variants, retrieve+RRF+CE"

    def _generate_variants(self, query: str) -> list[str]:
        prompt = (
            f"Generate {self._n_variants} different search queries that would help answer "
            "this question. Each query should approach the topic from a different angle. "
            "Return one query per line, no numbering.\n\n"
            f"Question: {query}"
        )
        try:
            response = _call_llm(prompt, max_tokens=300)
            variants = [q.strip().lstrip("0123456789.-) ") for q in response.strip().split("\n") if q.strip()]
            variants = [q for q in variants if len(q) > 10]
            return variants[:self._n_variants] if variants else [query]
        except Exception:
            return [query]

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        variants = self._generate_variants(query)

        # Retrieve for each variant and build RRF scores
        doc_rrf = {}  # doc_id -> rrf_score
        doc_map = {}  # doc_id -> RetrievalResult

        for q in [query] + variants:
            try:
                q_results, _ = self._backend.retrieve_with_embeddings(q, self._retrieve_k)
                for rank, r in enumerate(q_results):
                    if r.doc_id not in doc_map:
                        doc_map[r.doc_id] = r
                    rrf_score = 1.0 / (60 + rank + 1)
                    doc_rrf[r.doc_id] = doc_rrf.get(r.doc_id, 0) + rrf_score
            except Exception:
                pass

        # Also include original results
        for rank, r in enumerate(results):
            if r.doc_id not in doc_map:
                doc_map[r.doc_id] = r
            rrf_score = 1.0 / (60 + rank + 1)
            doc_rrf[r.doc_id] = doc_rrf.get(r.doc_id, 0) + rrf_score

        # Sort by RRF and take top candidates for CE
        sorted_ids = sorted(doc_rrf.keys(), key=lambda d: doc_rrf[d], reverse=True)
        candidates = [doc_map[d] for d in sorted_ids[:30]]  # CE on top 30 RRF

        return self._ce_rerank(query, candidates, n_variants=len(variants))

    def _ce_rerank(self, query, results, n_variants=0):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (query fusion, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "n_variants": n_variants},
        )
