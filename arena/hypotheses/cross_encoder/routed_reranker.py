"""Query-Adaptive Reranker Routing.

Uses score distribution features (CV, gap, entropy, query length) to route
each query to the optimal reranker: CE L-12, BGE-v2-m3, or MaxSim.

Why robust: Routing adapts to embedder strengths automatically. On snowflake
where BGE dominates, the router mostly picks BGE. On nomic where different
queries benefit from different rerankers, the router exploits that diversity.

Highest complexity, highest overfitting risk — uses a simple decision tree
(no learned parameters) to minimise overfitting.
"""

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000


class RoutedRerankerHypothesis(Hypothesis):
    """Route queries to optimal reranker based on score distribution features."""

    def __init__(self, pool_size=50, final_k=10):
        self._pool_size = pool_size
        self._final_k = final_k
        self._ce_model = None
        self._bge_model = None
        self._nomic_model = None
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

    def _get_nomic_model(self):
        if self._nomic_model is None:
            from sentence_transformers import SentenceTransformer
            self._nomic_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True,
            )
        return self._nomic_model

    @property
    def name(self):
        return "routed-reranker"

    @property
    def description(self):
        return (
            f"Query-adaptive routing: route to CE/BGE/MaxSim based on "
            f"score distribution features, pool {self._pool_size} → {self._final_k}"
        )

    def _compute_features(self, query, scores):
        """Compute routing features from query and score distribution."""
        scores = np.array(scores)

        # Score distribution features
        cv = scores.std() / max(abs(scores.mean()), 1e-12)

        # Gap between top-1 and top-2
        sorted_scores = np.sort(scores)[::-1]
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

        # Entropy of normalised scores
        probs = scores - scores.min()
        prob_sum = probs.sum()
        if prob_sum > 1e-12:
            probs = probs / prob_sum
            entropy = -np.sum(probs * np.log(probs + 1e-12))
        else:
            entropy = 0.0

        # Query length (word count)
        query_len = len(query.split())

        return {
            "cv": float(cv),
            "gap": float(gap),
            "entropy": float(entropy),
            "query_len": query_len,
        }

    def _route(self, features):
        """Simple decision tree for routing. No learned parameters.

        Routing logic:
        - High CV (peaked distribution) + short query → BGE (strong at focused queries)
        - Low CV (flat distribution) → CE (better at discrimination on hard queries)
        - Long queries (>10 words) → MaxSim (token-level matching helps)
        """
        if features["query_len"] > 10:
            return "maxsim"
        if features["cv"] > 0.15:
            return "bge"
        return "ce"

    def _rerank_ce(self, query, results):
        model = self._get_ce_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        return model.predict(pairs).tolist()

    def _rerank_bge(self, query, results):
        model = self._get_bge_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in results]
        return model.predict(pairs).tolist()

    def _rerank_maxsim(self, query, results):
        model = self._get_nomic_model()
        prompts = getattr(model, 'prompts', {})
        q_prefix = prompts.get('query', 'search_query: ')
        d_prefix = prompts.get('document', 'search_document: ')

        # Get token embeddings
        query_embs = model.encode(
            [f"{q_prefix}{query}"],
            output_value="token_embeddings",
            batch_size=1,
            show_progress_bar=False,
        )
        q_tokens = query_embs[0]
        if not isinstance(q_tokens, np.ndarray):
            q_tokens = q_tokens.cpu().numpy()
        q_norms = np.linalg.norm(q_tokens, axis=1, keepdims=True)
        q_tokens = q_tokens / np.maximum(q_norms, 1e-12)

        doc_texts = [f"{d_prefix}{r.text[:_MAX_CHARS]}" for r in results]
        doc_embs = model.encode(
            doc_texts,
            output_value="token_embeddings",
            batch_size=16,
            show_progress_bar=False,
        )

        scores = []
        for d_emb in doc_embs:
            arr = d_emb.cpu().numpy() if not isinstance(d_emb, np.ndarray) else d_emb
            d_norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(d_norms, 1e-12)
            sim = q_tokens @ arr.T
            scores.append(float(sim.max(axis=1).sum()))

        return scores

    def apply(self, query, results, embeddings, query_embedding):
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Compute routing features
        scores = [r.score for r in results]
        features = self._compute_features(query, scores)
        route = self._route(features)

        # Apply chosen reranker
        if route == "bge":
            rerank_scores = self._rerank_bge(query, results)
        elif route == "maxsim":
            rerank_scores = self._rerank_maxsim(query, results)
        else:
            rerank_scores = self._rerank_ce(query, results)

        ranked = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [results[i] for i in top]

        lines = [f"Retrieved context (routed → {route}, pool {len(results)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (score: {rerank_scores[idx]:.4f})")
            lines.append(results[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "route": route,
                "features": features,
                "pool_size": len(results),
                "rerank_scores": [rerank_scores[i] for i in top],
            },
        )
