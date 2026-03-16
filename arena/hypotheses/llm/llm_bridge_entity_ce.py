"""LLM Bridge Entity Extraction + targeted retrieval.

For multi-hop questions, the key insight is that the "bridge entity" connects
the two hops. Use LLM to extract this entity from the top-1 CE document,
then retrieve specifically for that entity.

Example: "What is the capital of the country where the Eiffel Tower is?"
Top-1 doc: "The Eiffel Tower is in Paris, France"
Bridge entity: "France"
Follow-up retrieval: "France capital"
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


class LLMBridgeEntityCEHypothesis(Hypothesis):
    """Extract bridge entity via LLM from top-1, retrieve for it, CE rerank merged pool."""

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
        return "llm-bridge-entity-ce"

    @property
    def description(self):
        return "LLM extracts bridge entity from top-1 doc, retrieve for it, CE rerank merged"

    def _extract_bridge_entity(self, question: str, doc_text: str) -> str | None:
        prompt = (
            "Given this question and document, identify the KEY ENTITY mentioned in the "
            "document that we need to look up to fully answer the question. "
            "Return ONLY the entity name, nothing else. If no bridge entity is needed, "
            "return 'NONE'.\n\n"
            f"Question: {question}\n"
            f"Document: {doc_text[:800]}\n\n"
            "Bridge entity:"
        )
        try:
            entity = _call_llm(prompt, max_tokens=50).strip().strip('"\'')
            if entity.upper() == "NONE" or len(entity) < 2:
                return None
            return entity
        except Exception:
            return None

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is None:
            return self._ce_rerank(query, results)

        # Step 1: Deep first retrieval
        try:
            first_results, _ = self._backend.retrieve_with_embeddings(query, 15)
        except Exception:
            first_results = results

        # Step 2: Quick CE to find top-1
        model = self._get_model()
        if not first_results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        pairs = [(query, r.text[:_MAX_CHARS]) for r in first_results]
        scores = model.predict(pairs).tolist()
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        top_doc = first_results[top_idx]

        # Step 3: Extract bridge entity
        bridge_entity = self._extract_bridge_entity(query, top_doc.text)

        # Step 4: Retrieve for bridge entity if found
        all_results = {r.doc_id: r for r in first_results}
        if bridge_entity:
            try:
                bridge_results, _ = self._backend.retrieve_with_embeddings(bridge_entity, 15)
                for r in bridge_results:
                    if r.doc_id not in all_results:
                        all_results[r.doc_id] = r
            except Exception:
                pass

        merged = list(all_results.values())
        return self._ce_rerank(query, merged, bridge_entity=bridge_entity)

    def _ce_rerank(self, query, results, bridge_entity=None):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (bridge entity, pool={len(results)}, CE reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"pool_size": len(results), "bridge_entity": bridge_entity or ""},
        )
