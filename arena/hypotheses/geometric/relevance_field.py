"""Relevance Field Estimation — LLM as a sparse relevance oracle.

Uses a deep candidate pool + CE scoring, then selects a diverse sample of
documents for LLM relevance judgement. The LLM scores are propagated to
unscored documents via k-NN weighted interpolation in embedding space,
creating a "relevance field" over the full candidate set.

The final ranking blends CE scores with the estimated relevance field.

Note: LLM calls are slow (~17s each × 15 docs ≈ 4 min per query).
Each call is wrapped in try/except so a single failure won't kill the query.
"""

import json
import random
import urllib.request

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 100) -> str:
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


def _parse_relevance(response: str) -> float:
    """Parse LLM response into a relevance score."""
    text = response.lower().strip()
    if "partially relevant" in text:
        return 0.5
    if "irrelevant" in text or "not relevant" in text:
        return 0.0
    if "relevant" in text:
        return 1.0
    # Fallback: assume irrelevant if we can't parse
    return 0.0


class RelevanceFieldHypothesis(Hypothesis):
    """Estimate a relevance field over embedding space using sparse LLM judgements."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        n_top: int = 5,
        n_bottom: int = 5,
        n_middle: int = 5,
        knn_k: int = 5,
        llm_weight: float = 0.4,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._n_top = n_top
        self._n_bottom = n_bottom
        self._n_middle = n_middle
        self._knn_k = knn_k
        self._llm_weight = llm_weight
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        """Inject the backend for deep retrieval."""
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "relevance-field"

    @property
    def description(self) -> str:
        return (
            "Sparse LLM relevance judgements propagated via k-NN interpolation "
            "in embedding space, blended with CE scores"
        )

    def _select_diverse_sample(self, ce_ranked_indices: list[int]) -> list[int]:
        """Select a diverse sample of indices for LLM scoring.

        Picks top-N by CE (high confidence relevant), bottom-N (likely
        irrelevant), and N random from the middle (uncertain region).
        """
        n = len(ce_ranked_indices)
        top = ce_ranked_indices[: self._n_top]
        bottom = ce_ranked_indices[max(0, n - self._n_bottom) :]
        # Middle region: everything not in top or bottom
        top_set = set(top)
        bottom_set = set(bottom)
        middle_pool = [i for i in ce_ranked_indices if i not in top_set and i not in bottom_set]
        n_mid = min(self._n_middle, len(middle_pool))
        middle = random.sample(middle_pool, n_mid) if middle_pool else []
        # Combine, preserving uniqueness
        sample_set = set()
        sample = []
        for idx in top + middle + bottom:
            if idx not in sample_set:
                sample_set.add(idx)
                sample.append(idx)
        return sample

    def _score_with_llm(self, query: str, doc_text: str) -> float | None:
        """Score a single document with the LLM. Returns None on failure."""
        prompt = (
            f"Given the question: {query}\n"
            f"Document: {doc_text[:500]}\n"
            "Is this document relevant to answering the question? "
            "Answer: relevant, partially relevant, or irrelevant."
        )
        try:
            response = _call_llm(prompt, max_tokens=100)
            return _parse_relevance(response)
        except Exception:
            return None

    def _knn_predict(
        self,
        scored_indices: list[int],
        llm_scores: dict[int, float],
        embeddings: np.ndarray,
        target_idx: int,
    ) -> float:
        """Predict relevance for an unscored doc using k-NN interpolation.

        Uses inverse-distance weighted average of LLM scores from the
        k nearest scored documents in embedding space.
        """
        target_emb = embeddings[target_idx]
        scored_embs = embeddings[scored_indices]

        # Compute distances
        diffs = scored_embs - target_emb[np.newaxis, :]
        distances = np.linalg.norm(diffs, axis=1)

        # Get k nearest
        k = min(self._knn_k, len(scored_indices))
        nearest_local = np.argsort(distances)[:k]

        # Inverse-distance weighted average
        weights = []
        values = []
        for local_i in nearest_local:
            global_i = scored_indices[local_i]
            d = distances[local_i]
            # Avoid division by zero — if distance is 0, this doc is identical
            w = 1.0 / (d + 1e-10)
            weights.append(w)
            values.append(llm_scores[global_i])

        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return sum(w * v for w, v in zip(weights, values)) / total_w

    def apply(self, query, results, embeddings, query_embedding):
        # Step 1: Deep retrieval from backend
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to normal results

        if not results:
            return HypothesisResult(
                results=[],
                context_prompt="Retrieved context:\n(no results)",
                metadata={},
            )

        # Step 2: CE score all candidates
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Sort indices by CE score descending
        ce_ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)

        # Step 3: Get embeddings (already available from deep retrieval)
        has_embeddings = embeddings is not None and len(embeddings) == len(results)

        # Step 4: Select diverse sample for LLM scoring
        sample_indices = self._select_diverse_sample(ce_ranked)

        # Step 5: LLM scores each sampled doc
        llm_scores = {}  # index -> score
        for idx in sample_indices:
            score = self._score_with_llm(query, results[idx].text)
            if score is not None:
                llm_scores[idx] = score

        # Step 6 & 7: Compute final scores
        scored_indices = list(llm_scores.keys())
        final_scores = []

        for i in range(len(results)):
            ce_s = ce_scores[i]

            if i in llm_scores:
                # Directly scored by LLM
                llm_s = llm_scores[i]
            elif scored_indices and has_embeddings:
                # Predict via k-NN interpolation in embedding space
                llm_s = self._knn_predict(scored_indices, llm_scores, embeddings, i)
            else:
                # No LLM signal available — use CE only
                llm_s = 0.0

            final = ce_s + self._llm_weight * llm_s
            final_scores.append(final)

        # Step 8: Return top-K by final score
        ranked = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top = ranked[: self._final_k]
        reranked = [results[i] for i in top]

        # Build context prompt
        lines = [
            f"Retrieved context (relevance field, pool={len(results)}, "
            f"llm_scored={len(llm_scores)}, λ={self._llm_weight}):"
        ]
        for rank, idx in enumerate(top, 1):
            ce_s = ce_scores[idx]
            llm_tag = ""
            if idx in llm_scores:
                llm_tag = f", llm={llm_scores[idx]:.1f}"
            lines.append(f"\n[{rank}] (final: {final_scores[idx]:.4f}, ce: {ce_s:.4f}{llm_tag})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "pool_size": len(results),
                "n_llm_scored": len(llm_scores),
                "llm_scores": llm_scores,
                "ce_scores": [ce_scores[i] for i in top],
                "final_scores": [final_scores[i] for i in top],
                "sample_indices": sample_indices,
                "llm_weight": self._llm_weight,
            },
        )
