"""LLM Relevance Judge: use Qwen as a listwise reranker.

Research basis: RankGPT (Sun et al., 2023), RankZephyr (2024).
LLM-based listwise reranking gives +1-4% over CE on BEIR.

Instead of replacing CE, we use LLM as a second-stage judge:
CE picks top-10, LLM reranks them. Best of both worlds.
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


class LLMRelevanceJudgeHypothesis(Hypothesis):
    """CE picks top-10, then LLM reranks them as a listwise judge."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", final_k=10):
        self._model_name = model_name
        self._final_k = final_k
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return "llm-relevance-judge"

    @property
    def description(self):
        return "CE picks top-10, LLM reranks as listwise judge"

    def _llm_rerank(self, query: str, docs: list[RetrievalResult]) -> list[int]:
        """Use LLM to rerank documents by relevance."""
        # Build document list for LLM
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
            # Parse ranking from response
            import re
            numbers = re.findall(r'\d+', response)
            ranking = []
            seen = set()
            for n in numbers:
                idx = int(n) - 1  # convert 1-indexed to 0-indexed
                if 0 <= idx < len(docs) and idx not in seen:
                    ranking.append(idx)
                    seen.add(idx)
            # Add any missing indices at the end
            for i in range(len(docs)):
                if i not in seen:
                    ranking.append(i)
            return ranking
        except Exception:
            return list(range(len(docs)))

    def apply(self, query, results, embeddings, query_embedding):
        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        model = self._get_model()

        # Stage 1: CE rerank
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()
        ce_ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        ce_top = [results[i] for i in ce_ranked[:self._final_k]]

        # Stage 2: LLM rerank the CE top-K
        llm_ranking = self._llm_rerank(query, ce_top)
        reranked = [ce_top[i] for i in llm_ranking]

        lines = ["Retrieved context (CE + LLM judge reranked):"]
        for i, doc in enumerate(reranked, 1):
            lines.append(f"\n[{i}]")
            lines.append(doc.text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={"llm_ranking": llm_ranking},
        )
