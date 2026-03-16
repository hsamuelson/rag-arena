"""Score-Distribution-Gated MRAM + CE reranking.

Activates MRAM (sentence-level expansion) only when the passage embedding
score distribution shows signs of dilution (flat scores, low CV). On strong
embedders like snowflake, the score distribution is more peaked (high CV)
→ the gate rarely triggers → avoids the composability failure. On weaker
embedders like nomic, scores are flatter → gate triggers → gets MRAM benefit.

Why robust: Automatically adapts to embedder quality by measuring output
characteristics rather than assuming anything about the model. Addresses
the root cause of MRAM's composability failure.
"""

import re

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class GatedMRAMCEHypothesis(Hypothesis):
    """Score-distribution-gated MRAM: activate sentence expansion only when needed."""

    def __init__(
        self,
        pool_size=50,
        cv_threshold=0.15,
        ce_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        final_k=10,
    ):
        self._pool_size = pool_size
        self._cv_threshold = cv_threshold
        self._ce_model_name = ce_model
        self._final_k = final_k
        self._ce_model = None
        self._backend = None

        # Sentence index (built on first use, same as MRAM)
        self._sentence_embeddings = None
        self._sentence_texts = []
        self._sentence_parent_idx = []
        self._sentence_parent_id = []
        self._id_to_idx = {}
        self._index_built = False

    def set_backend(self, backend):
        self._backend = backend

    def _get_ce_model(self):
        if self._ce_model is None:
            from sentence_transformers import CrossEncoder
            self._ce_model = CrossEncoder(self._ce_model_name)
        return self._ce_model

    @property
    def name(self):
        return f"gated-mram-ce-cv{self._cv_threshold}"

    @property
    def description(self):
        return (
            f"Gated MRAM: activate sentence expansion when score CV < {self._cv_threshold}, "
            f"CE rerank to {self._final_k}"
        )

    def _build_sentence_index(self):
        """Build sentence-level index from backend corpus (same as MRAM)."""
        if self._backend is None or not hasattr(self._backend, '_corpus_texts'):
            return

        corpus_texts = self._backend._corpus_texts
        corpus_ids = self._backend._corpus_ids
        n_docs = len(corpus_texts)
        if n_docs == 0:
            return

        self._id_to_idx = {did: i for i, did in enumerate(corpus_ids)}

        print(f"  [Gated-MRAM] Building sentence index over {n_docs} docs...")

        sentences = []
        parent_indices = []
        parent_ids = []

        for doc_idx, text in enumerate(corpus_texts):
            sents = _SENT_SPLIT.split(text.strip())
            sents = [s.strip() for s in sents if len(s.strip()) >= 20]
            if not sents:
                sents = [text.strip()[:500]]
            for s in sents:
                sentences.append(s[:500])
                parent_indices.append(doc_idx)
                parent_ids.append(corpus_ids[doc_idx])

        self._sentence_texts = sentences
        self._sentence_parent_idx = parent_indices
        self._sentence_parent_id = parent_ids

        # Embed sentences in batches
        print(f"  [Gated-MRAM] Embedding {len(sentences)} sentences...")
        batch_size = 512
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embs = self._backend.embed_batch(batch)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = (embs / np.maximum(norms, 1e-12)).astype(np.float32)
            all_embs.append(embs)
            if (i // batch_size) % 20 == 0:
                print(f"    [{i}/{len(sentences)}] embedded...")

        self._sentence_embeddings = np.vstack(all_embs)
        del all_embs
        self._index_built = True
        print(f"  [Gated-MRAM] Sentence index ready: {self._sentence_embeddings.shape}")

    def _retrieve_sentence_level(self, query_emb, top_k):
        """Retrieve parent docs via sentence-level similarity."""
        if self._sentence_embeddings is None:
            return {}

        q = query_emb / max(np.linalg.norm(query_emb), 1e-12)
        sims = self._sentence_embeddings @ q.astype(np.float32)

        n_select = min(top_k * 3, len(sims))
        top_sent_indices = np.argpartition(sims, -n_select)[-n_select:]
        top_sent_indices = top_sent_indices[np.argsort(sims[top_sent_indices])[::-1]]

        seen_docs = set()
        doc_scores = {}
        for sent_idx in top_sent_indices:
            parent_id = self._sentence_parent_id[sent_idx]
            if parent_id not in seen_docs:
                seen_docs.add(parent_id)
                doc_scores[parent_id] = float(sims[sent_idx])
                if len(seen_docs) >= top_k:
                    break
        return doc_scores

    def apply(self, query, results, embeddings, query_embedding):
        if not self._index_built:
            self._build_sentence_index()

        # Passage-level retrieval
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_size
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        if query_embedding is None and self._backend is not None:
            try:
                query_embedding = self._backend.embed_query(query)
            except Exception:
                pass

        # Measure score distribution CV (coefficient of variation)
        scores = np.array([r.score for r in results])
        score_mean = scores.mean()
        score_std = scores.std()
        cv = score_std / max(abs(score_mean), 1e-12)

        # Gate: activate MRAM only if CV is below threshold (flat = noisy)
        mram_activated = cv < self._cv_threshold

        candidates = {r.doc_id: r for r in results}
        passage_doc_ids = set(candidates.keys())
        sentence_doc_ids = set()

        if mram_activated and query_embedding is not None:
            sentence_scores = self._retrieve_sentence_level(
                query_embedding, self._pool_size
            )
            for doc_id, score in sentence_scores.items():
                sentence_doc_ids.add(doc_id)
                if doc_id not in candidates:
                    idx = self._id_to_idx.get(doc_id)
                    if idx is not None:
                        candidates[doc_id] = RetrievalResult(
                            doc_id=doc_id,
                            text=self._backend._corpus_texts[idx],
                            score=score,
                            metadata=self._backend._corpus_metadata[idx],
                        )

        # CE rerank merged candidates
        candidate_list = list(candidates.values())

        model = self._get_ce_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in candidate_list]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [candidate_list[i] for i in top]

        unique_from_sentence = sum(
            1 for d in reranked
            if d.doc_id not in passage_doc_ids and d.doc_id in sentence_doc_ids
        )

        lines = [f"Retrieved context (gated-MRAM, CV={cv:.3f}, gate={'ON' if mram_activated else 'OFF'}, "
                 f"{len(candidate_list)} → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (ce_score: {ce_scores[idx]:.4f})")
            lines.append(candidate_list[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "score_cv": float(cv),
                "cv_threshold": self._cv_threshold,
                "mram_activated": mram_activated,
                "total_candidates": len(candidate_list),
                "from_passage": len(passage_doc_ids),
                "from_sentence": len(sentence_doc_ids),
                "unique_from_sentence": unique_from_sentence,
                "ce_scores": [ce_scores[i] for i in top],
            },
        )
