"""Multi-reranker ensemble: RRF fusion of CE-L12 + BGE + MaxSim scores.

Scores top-50 candidates with three independent rerankers and fuses
their rankings via Reciprocal Rank Fusion. Rank-based fusion avoids
score normalization issues across heterogeneous scoring functions.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_RRF_K = 60  # standard RRF constant


class MultiRerankerEnsembleHypothesis(Hypothesis):
    """Top-50 candidates scored by 3 rerankers, fused via RRF."""

    def __init__(self, final_k=10):
        self._pool_size = 50
        self._final_k = final_k
        self._ce_model = None
        self._bge_model = None
        self._st_model = None
        self._backend = None

    def set_backend(self, backend):
        self._backend = backend

    def _get_ce_model(self):
        if self._ce_model is None:
            from sentence_transformers import CrossEncoder
            self._ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        return self._ce_model

    def _get_bge_model(self):
        if self._bge_model is None:
            from sentence_transformers import CrossEncoder
            self._bge_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        return self._bge_model

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
            )
        return self._st_model

    @property
    def name(self):
        return "multi-reranker-ensemble"

    @property
    def description(self):
        return f"Top-{self._pool_size} scored by CE-L12 + BGE + MaxSim, RRF fused to top-{self._final_k}"

    def _maxsim_scores(self, query: str, results: list[RetrievalResult]) -> list[float]:
        model = self._get_st_model()
        query_embs = model.encode(
            [f"search_query: {query}"],
            output_value="token_embeddings",
            batch_size=1,
            show_progress_bar=False,
        )
        q_tokens = query_embs[0]
        if not isinstance(q_tokens, np.ndarray):
            q_tokens = q_tokens.cpu().numpy()
        q_norms = np.linalg.norm(q_tokens, axis=1, keepdims=True)
        q_tokens = q_tokens / np.maximum(q_norms, 1e-12)

        doc_texts = [f"search_document: {r.text[:_MAX_CHARS]}" for r in results]
        doc_embs = model.encode(
            doc_texts,
            output_value="token_embeddings",
            batch_size=16,
            show_progress_bar=False,
        )

        scores = []
        for d_emb in doc_embs:
            if not isinstance(d_emb, np.ndarray):
                d_emb = d_emb.cpu().numpy()
            d_norms = np.linalg.norm(d_emb, axis=1, keepdims=True)
            d_emb = d_emb / np.maximum(d_norms, 1e-12)
            sim = q_tokens @ d_emb.T
            scores.append(float(sim.max(axis=1).sum()))
        return scores

    def _rrf_fuse(self, rankings: list[list[int]], n_items: int) -> list[int]:
        """Reciprocal Rank Fusion across multiple rankings."""
        scores = np.zeros(n_items)
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[idx] += 1.0 / (_RRF_K + rank + 1)
        return sorted(range(n_items), key=lambda i: scores[i], reverse=True)

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(query, self._pool_size)
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        n = len(results)

        # Score with CE-L12
        ce_scores = self._get_ce_model().predict(pairs).tolist()
        ce_ranking = sorted(range(n), key=lambda i: ce_scores[i], reverse=True)

        # Score with BGE
        bge_scores = self._get_bge_model().predict(pairs).tolist()
        bge_ranking = sorted(range(n), key=lambda i: bge_scores[i], reverse=True)

        # Score with MaxSim
        ms_scores = self._maxsim_scores(query, results)
        ms_ranking = sorted(range(n), key=lambda i: ms_scores[i], reverse=True)

        # RRF fusion
        fused = self._rrf_fuse([ce_ranking, bge_ranking, ms_ranking], n)
        top = fused[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (ensemble RRF, {len(results)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce={ce_scores[idx]:.3f}, bge={bge_scores[idx]:.3f}, ms={ms_scores[idx]:.1f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "ce_scores": [ce_scores[i] for i in top],
                "bge_scores": [bge_scores[i] for i in top],
                "maxsim_scores": [ms_scores[i] for i in top],
                "pool_size": n,
            },
        )
