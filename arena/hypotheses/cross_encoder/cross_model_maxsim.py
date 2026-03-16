"""Cross-Model MaxSim: heterogeneous retrieval stages.

Uses the primary embedder (e.g. snowflake) for first-stage retrieval, but
ALWAYS uses nomic-embed-text for MaxSim token-level reranking. Decouples
the retrieval embedder from the reranking model.

Why robust: nomic's token embeddings are trained for MaxSim-style interaction
(via Matryoshka + ColBERT-inspired training). Snowflake's token embeddings
are not optimised for this. By always using nomic for token-level scoring,
we get consistent reranking quality regardless of the retrieval embedder.

Why novel: Heterogeneous retrieval — deliberately using different models for
different stages by design, rather than the usual homogeneous pipeline.

References:
  - Khattab & Zaharia, "ColBERT" (SIGIR 2020)
  - nomic-embed-text v1.5 (Matryoshka + ColBERT training)
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class CrossModelMaxSimHypothesis(Hypothesis):
    """First-stage: any embedder. Rerank: always nomic MaxSim."""

    def __init__(self, pool_size=50, final_k=10):
        self._pool_size = pool_size
        self._final_k = final_k
        self._nomic_model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_nomic_model(self):
        """Always use nomic for token-level reranking."""
        if self._nomic_model is None:
            from sentence_transformers import SentenceTransformer
            self._nomic_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
            )
        return self._nomic_model

    @property
    def name(self):
        return "cross-model-maxsim"

    @property
    def description(self):
        return (
            f"Cross-model: any embedder for retrieval, "
            f"nomic MaxSim token-level rerank top-{self._pool_size} → {self._final_k}"
        )

    def _token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Get per-token embeddings using nomic model."""
        model = self._get_nomic_model()
        token_embs = model.encode(
            texts,
            output_value="token_embeddings",
            batch_size=16,
            show_progress_bar=False,
        )
        result = []
        for emb in token_embs:
            arr = emb.cpu().numpy() if not isinstance(emb, np.ndarray) else emb
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            result.append(arr / norms)
        return result

    def _maxsim_score(self, query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
        """MaxSim: sum of max cosine similarities per query token."""
        sim_matrix = query_tokens @ doc_tokens.T
        return float(sim_matrix.max(axis=1).sum())

    def apply(self, query, results, embeddings, query_embedding):
        # First stage: use whatever backend embedder for retrieval
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Second stage: ALWAYS use nomic for MaxSim token-level reranking
        model = self._get_nomic_model()
        prompts = getattr(model, 'prompts', {})
        q_prefix = prompts.get('query', 'search_query: ')
        d_prefix = prompts.get('document', 'search_document: ')

        query_token_embs = self._token_embeddings([f"{q_prefix}{query}"])[0]
        doc_texts = [f"{d_prefix}{r.text[:_MAX_CHARS]}" for r in results]
        doc_token_embs = self._token_embeddings(doc_texts)

        maxsim_scores = [self._maxsim_score(query_token_embs, d) for d in doc_token_embs]

        ranked = sorted(range(len(maxsim_scores)), key=lambda i: maxsim_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (cross-model MaxSim, pool {len(results)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (maxsim: {maxsim_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "pool_size": len(results),
                "maxsim_scores": [maxsim_scores[i] for i in top],
                "reranker_model": "nomic-ai/nomic-embed-text-v1.5",
            },
        )
