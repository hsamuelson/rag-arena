"""Multi-Resolution Associative Memory (MRAM) — Phase 1.5.

Upgraded from Phase 1 based on findings:
- float32 sentence embeddings (was float16, now have 32GB RAM)
- Upgraded CE reranker (ms-marco-MiniLM-L-12-v2, 2x params)
- Dropped topic-level index (only helped 25% of queries, marginal gain)
- Kept sentence + passage as the two resolution levels

Algorithm:
1. Pre-build sentence-level index on first use:
   - Split each corpus doc into sentences
   - Embed all sentences (float32), maintain sentence→parent_doc mapping
2. At query time, retrieve from 2 levels:
   - Passage level: standard top-50 from backend (BM25 + dense hybrid)
   - Sentence level: top-50 sentences by cosine sim → map to parent docs
3. Merge candidates (deduplicate by doc_id)
4. CE rerank merged pool to top-10
"""

import re

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class MultiResolutionHypothesis(Hypothesis):
    """MRAM Phase 1.5: sentence + passage retrieval with upgraded CE reranker."""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        pool_per_level=50,
        final_k=10,
    ):
        self._model_name = model_name
        self._pool_per_level = pool_per_level
        self._final_k = final_k
        self._model = None
        self._backend = None

        # Sentence-level index (built on first use)
        self._sentence_embeddings = None  # (N_sentences, dim) float32
        self._sentence_texts = []         # raw sentence text
        self._sentence_parent_idx = []    # index into backend corpus
        self._sentence_parent_id = []     # doc_id of parent

        # Fast doc_id → corpus index lookup
        self._id_to_idx = {}

        self._index_built = False

    def set_backend(self, backend):
        self._backend = backend

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    @property
    def name(self):
        return "multi-resolution"

    @property
    def description(self):
        return (
            f"MRAM Phase 1.5: sentence+passage retrieval, "
            f"merge {self._pool_per_level}×2 candidates, "
            f"CE rerank ({self._model_name.split('/')[-1]}) to {self._final_k}"
        )

    def _build_index(self):
        """Build sentence-level index from backend corpus."""
        if self._backend is None:
            return
        if not hasattr(self._backend, '_corpus_texts'):
            return

        corpus_texts = self._backend._corpus_texts
        corpus_ids = self._backend._corpus_ids
        n_docs = len(corpus_texts)

        if n_docs == 0:
            return

        # Build fast lookup
        self._id_to_idx = {did: i for i, did in enumerate(corpus_ids)}

        print(f"  [MRAM] Building sentence-level index over {n_docs} docs...")

        # ─── Sentence splitting ───
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
        print(f"  [MRAM] {len(sentences)} sentences from {n_docs} docs")

        # ─── Embed in batches, normalize, keep float32 ───
        print(f"  [MRAM] Embedding sentences (float32, batched)...")
        batch_size = 512
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embs = self._backend.embed_batch(batch)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embs = (embs / norms).astype(np.float32)
            all_embs.append(embs)
            if (i // batch_size) % 20 == 0:
                print(f"    [{i}/{len(sentences)}] embedded...")

        self._sentence_embeddings = np.vstack(all_embs)
        del all_embs
        mem_mb = self._sentence_embeddings.nbytes / 1024 / 1024
        print(f"  [MRAM] Sentence embeddings: {self._sentence_embeddings.shape} "
              f"(float32, {mem_mb:.0f} MB)")

        self._index_built = True
        print(f"  [MRAM] Sentence index ready")

    def _retrieve_sentence_level(self, query_emb, top_k):
        """Retrieve top-K parent documents by sentence similarity."""
        if self._sentence_embeddings is None or len(self._sentence_texts) == 0:
            return {}

        q = query_emb / max(np.linalg.norm(query_emb), 1e-12)
        sims = self._sentence_embeddings @ q.astype(np.float32)

        top_sent_indices = np.argpartition(sims, -min(top_k * 3, len(sims)))[-top_k * 3:]
        top_sent_indices = top_sent_indices[np.argsort(sims[top_sent_indices])[::-1]]

        # Deduplicate by parent doc, keep best sentence per doc
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
        # Build index on first use
        if not self._index_built:
            self._build_index()

        # ─── Level 1: Passage-level retrieval (deep pool) ───
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_per_level
                )
            except Exception:
                pass

        if not results:
            return HypothesisResult(results=[], context_prompt="(no results)", metadata={})

        # Get query embedding
        if query_embedding is None and self._backend is not None:
            try:
                query_embedding = self._backend.embed_query(query)
            except Exception:
                pass

        # ─── Collect candidates from both levels ───
        candidates = {}  # doc_id → RetrievalResult
        for r in results:
            candidates[r.doc_id] = r

        passage_doc_ids = set(candidates.keys())
        sentence_doc_ids = set()

        # ─── Level 2: Sentence-level candidates ───
        if query_embedding is not None:
            sentence_scores = self._retrieve_sentence_level(
                query_embedding, self._pool_per_level
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

        # ─── CE rerank merged pool ───
        candidate_list = list(candidates.values())

        model = self._get_model()
        pairs = [(query, r.text[:_MAX_CHARS]) for r in candidate_list]
        ce_scores = model.predict(pairs).tolist()

        ranked = sorted(range(len(ce_scores)), key=lambda i: ce_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [candidate_list[i] for i in top]

        # Track provenance
        source_info = {}
        for idx in top:
            doc_id = candidate_list[idx].doc_id
            sources = []
            if doc_id in passage_doc_ids:
                sources.append("passage")
            if doc_id in sentence_doc_ids:
                sources.append("sentence")
            source_info[doc_id] = sources

        unique_from_sentence = sum(
            1 for d in reranked
            if d.doc_id not in passage_doc_ids and d.doc_id in sentence_doc_ids
        )

        lines = [f"Retrieved context (multi-resolution, {len(candidate_list)} candidates → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            doc_id = candidate_list[idx].doc_id
            lines.append(f"\n[{i}] (ce={ce_scores[idx]:.4f}, from={source_info.get(doc_id, [])})")
            lines.append(candidate_list[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "total_candidates": len(candidate_list),
                "from_passage": len(passage_doc_ids),
                "from_sentence": len(sentence_doc_ids),
                "unique_from_sentence": unique_from_sentence,
                "ce_scores": [ce_scores[i] for i in top],
                "source_info": source_info,
            },
        )
