"""Late interaction (ColBERT-style) reranker using nomic-embed-text.

Computes MaxSim between query and document token embeddings from the
same nomic-embed-text model used for retrieval. This is maximally fair:
no new model weights, just a different scoring function (token-level
MaxSim vs pooled cosine similarity).

Reference: Khattab & Zaharia, "ColBERT" (SIGIR 2020).
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class LateInteractionRerankerHypothesis(Hypothesis):
    """Top-50 candidates reranked by token-level MaxSim using nomic-embed-text."""

    def __init__(self, pool_size=50, final_k=10):
        self._pool_size = pool_size
        self._final_k = final_k
        self._model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
            )
        return self._model

    @property
    def name(self):
        return "late-interaction-reranker"

    @property
    def description(self):
        return f"Top-{self._pool_size} reranked by token-level MaxSim (nomic-embed-text)"

    def _token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get per-token embeddings for a list of texts."""
        model = self._get_model()
        # Encode with token-level output
        token_embs = model.encode(
            texts,
            output_value="token_embeddings",
            batch_size=16,
            show_progress_bar=False,
        )
        # Normalize each token embedding
        result = []
        for emb in token_embs:
            if isinstance(emb, np.ndarray):
                arr = emb
            else:
                arr = emb.cpu().numpy()
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            result.append(arr / norms)
        return result

    def _maxsim_score(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """Compute MaxSim: sum of max cosine similarities per query token."""
        # query_tokens: (Q, D), doc_tokens: (T, D)
        # For each query token, find max similarity with any doc token
        sim_matrix = query_tokens @ doc_tokens.T  # (Q, T)
        max_sims = sim_matrix.max(axis=1)  # (Q,)
        return float(max_sims.sum())

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, self._pool_size)
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Get token embeddings for query and docs, using model-specific prefixes
        model = self._get_model()
        prompts = getattr(model, 'prompts', {})
        q_prefix = prompts.get('query', 'search_query: ')
        d_prefix = prompts.get('document', 'search_document: ')

        query_token_embs = self._token_embeddings([f"{q_prefix}{query}"])[0]

        doc_texts = [f"{d_prefix}{r.text[:_MAX_CHARS]}" for r in results]
        doc_token_embs = self._token_embeddings(doc_texts)

        # Compute MaxSim scores
        maxsim_scores = []
        for doc_emb in doc_token_embs:
            score = self._maxsim_score(query_token_embs, doc_emb)
            maxsim_scores.append(score)

        # Rank by MaxSim
        ranked = sorted(range(len(maxsim_scores)), key=lambda i: maxsim_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (deep pool {len(results)} → {len(top)}, MaxSim reranked):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (maxsim: {maxsim_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "maxsim_scores": [maxsim_scores[i] for i in top],
                "pool_size": len(results),
            },
        )
