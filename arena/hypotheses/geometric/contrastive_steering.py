"""Contrastive Embedding Steering.

Uses an LLM to generate positive and negative document descriptions for a query,
computes a steering vector in embedding space (positive - negative, normalised),
then shifts the query embedding toward the positive direction. Documents from a
deep pool are scored by a weighted combination of CE score and cosine similarity
to the steered query.

If the LLM is unavailable (Ollama not reachable), falls back to pure deep-pool
CE reranking.
"""

import json
import urllib.request

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_OLLAMA_URL = "http://host.docker.internal:11434"


def _call_llm(prompt: str, max_tokens: int = 200) -> str:
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
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


class ContrastiveSteeringHypothesis(Hypothesis):
    """Retrieve a deep pool, steer query embedding via contrastive LLM descriptions, CE+cosine rerank."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        pool_size: int = 50,
        final_k: int = 10,
        beta: float = 0.3,
        lam: float = 0.3,
    ):
        self._model_name = model_name
        self._pool_size = pool_size
        self._final_k = final_k
        self._beta = beta
        self._lam = lam
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        """Inject the backend for deep retrieval and embedding."""
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self) -> str:
        return "contrastive-steering"

    @property
    def description(self) -> str:
        return (
            "LLM generates positive/negative descriptions, builds a steering vector "
            "in embedding space, combines CE scores with steered cosine similarity"
        )

    def _generate_descriptions(self, query: str) -> tuple[str, str]:
        """Ask the LLM for a positive and negative document description."""
        pos_prompt = (
            f"Given the question: '{query}'\n"
            "Describe in 1-2 sentences what a highly relevant document would discuss. "
            "Be specific about the key topics, entities, and facts it would contain. "
            "Just give the description, nothing else."
        )
        neg_prompt = (
            f"Given the question: '{query}'\n"
            "Describe in 1-2 sentences what a misleading or irrelevant document might "
            "discuss — something that shares surface-level keywords but doesn't actually "
            "help answer the question. Just give the description, nothing else."
        )
        positive_text = _call_llm(pos_prompt)
        negative_text = _call_llm(neg_prompt)
        return positive_text, negative_text

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm

    def apply(self, query, results, embeddings, query_embedding):
        # Deep retrieval
        if self._backend is not None:
            try:
                deep_results, deep_embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
                results = deep_results
                embeddings = deep_embeddings
            except Exception:
                pass  # fall back to provided results

        if not results:
            return HypothesisResult(
                results=[], context_prompt="Retrieved context:\n(no results)", metadata={}
            )

        # CE scores for the pool
        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        ce_scores = model.predict(pairs).tolist()

        # Attempt contrastive steering via LLM
        used_steering = False
        steered_sims = [0.0] * len(results)

        try:
            positive_text, negative_text = self._generate_descriptions(query)

            if self._backend is not None and query_embedding is not None:
                # Embed the positive and negative descriptions
                pos_emb = self._backend.embed_query(positive_text)
                neg_emb = self._backend.embed_query(negative_text)

                # Steering vector: positive - negative, normalised
                steer = self._normalise(pos_emb - neg_emb)

                # Steered query embedding
                query_steered = self._normalise(query_embedding + self._beta * steer)

                # Cosine similarities of each doc to the steered query
                if embeddings is not None and len(embeddings) == len(results):
                    # Normalise doc embeddings row-wise
                    doc_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    doc_norms = np.where(doc_norms < 1e-12, 1.0, doc_norms)
                    doc_normed = embeddings / doc_norms
                    steered_sims = (doc_normed @ query_steered).tolist()
                    used_steering = True
        except Exception:
            pass  # LLM or embedding failed — fall back to pure CE

        # Final score = ce_score + lambda * steered_sim
        final_scores = [
            ce_scores[i] + self._lam * steered_sims[i]
            for i in range(len(results))
        ]

        # Rank by final score, take top-k
        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_indices = ranked_indices[: self._final_k]
        reranked = [results[i] for i in top_indices]

        # Format context prompt
        tag = "contrastive-steered" if used_steering else "CE-only fallback"
        lines = [f"Retrieved context (pool={len(results)}, {tag}):"]
        for i, idx in enumerate(top_indices, 1):
            lines.append(
                f"\n[{i}] (final: {final_scores[idx]:.4f}, ce: {ce_scores[idx]:.4f}, "
                f"steer_sim: {steered_sims[idx]:.4f})"
            )
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top_indices],
                "steered_sims": [steered_sims[i] for i in top_indices],
                "final_scores": [final_scores[i] for i in top_indices],
                "pool_size": len(results),
                "final_k": len(top_indices),
                "used_steering": used_steering,
                "beta": self._beta,
                "lambda": self._lam,
            },
        )
