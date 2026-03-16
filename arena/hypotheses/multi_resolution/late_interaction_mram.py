"""MRAM + Late Interaction: sentence-level candidates with MaxSim reranking.

Combines MRAM's sentence-level candidate expansion with ColBERT-style
token-level MaxSim reranking. Tests whether the two innovations compose:
better candidates (MRAM) + better scoring (MaxSim).
"""

import re

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class LateInteractionMRAMHypothesis(Hypothesis):
    """MRAM sentence+passage candidates, MaxSim token-level reranking."""

    def __init__(self, pool_per_level=50, final_k=10):
        self._pool_per_level = pool_per_level
        self._final_k = final_k
        self._st_model = None
        self._backend = None

        # Sentence-level index (shared with MRAM)
        self._sentence_embeddings = None
        self._sentence_texts = []
        self._sentence_parent_idx = []
        self._sentence_parent_id = []
        self._id_to_idx = {}
        self._index_built = False

    def set_backend(self, backend):
        self._backend = backend

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
        return "late-interaction-mram"

    @property
    def description(self):
        return (
            f"MRAM sentence+passage retrieval, "
            f"merge {self._pool_per_level}×2 candidates, "
            f"MaxSim token-level rerank to {self._final_k}"
        )

    def _build_index(self):
        """Build sentence-level index from backend corpus."""
        if self._backend is None or not hasattr(self._backend, '_corpus_texts'):
            return

        corpus_texts = self._backend._corpus_texts
        corpus_ids = self._backend._corpus_ids
        n_docs = len(corpus_texts)
        if n_docs == 0:
            return

        self._id_to_idx = {did: i for i, did in enumerate(corpus_ids)}

        print(f"  [LI-MRAM] Building sentence-level index over {n_docs} docs...")

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
        print(f"  [LI-MRAM] {len(sentences)} sentences from {n_docs} docs")

        # Embed using backend (same as MRAM)
        print(f"  [LI-MRAM] Embedding sentences...")
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
        self._index_built = True
        print(f"  [LI-MRAM] Sentence index ready: {self._sentence_embeddings.shape}")

    def _retrieve_sentence_level(self, query_emb, top_k):
        if self._sentence_embeddings is None or len(self._sentence_texts) == 0:
            return {}

        q = query_emb / max(np.linalg.norm(query_emb), 1e-12)
        sims = self._sentence_embeddings @ q.astype(np.float32)

        top_sent_indices = np.argpartition(sims, -min(top_k * 3, len(sims)))[-top_k * 3:]
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

    def _token_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        model = self._get_st_model()
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
        sim_matrix = query_tokens @ doc_tokens.T
        return float(sim_matrix.max(axis=1).sum())

    def apply(self, query, results, embeddings, query_embedding):
        if not self._index_built:
            self._build_index()

        # Level 1: Passage-level retrieval
        if self._backend is not None:
            try:
                results, embeddings = self._backend.retrieve_with_embeddings(
                    query, self._pool_per_level
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

        # Collect candidates from both levels
        candidates = {}
        for r in results:
            candidates[r.doc_id] = r
        passage_doc_ids = set(candidates.keys())
        sentence_doc_ids = set()

        # Level 2: Sentence-level candidates
        if query_embedding is not None:
            sentence_scores = self._retrieve_sentence_level(query_embedding, self._pool_per_level)
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

        candidate_list = list(candidates.values())

        # MaxSim token-level reranking (use model-specific prefixes)
        model = self._get_st_model()
        prompts = getattr(model, 'prompts', {})
        q_prefix = prompts.get('query', 'search_query: ')
        d_prefix = prompts.get('document', 'search_document: ')

        query_token_embs = self._token_embeddings([f"{q_prefix}{query}"])[0]
        doc_texts = [f"{d_prefix}{r.text[:_MAX_CHARS]}" for r in candidate_list]
        doc_token_embs = self._token_embeddings(doc_texts)

        maxsim_scores = [self._maxsim_score(query_token_embs, d) for d in doc_token_embs]

        ranked = sorted(range(len(maxsim_scores)), key=lambda i: maxsim_scores[i], reverse=True)
        top = ranked[:self._final_k]
        reranked = [candidate_list[i] for i in top]

        unique_from_sentence = sum(
            1 for d in reranked
            if d.doc_id not in passage_doc_ids and d.doc_id in sentence_doc_ids
        )

        lines = [f"Retrieved context (MRAM + MaxSim, {len(candidate_list)} candidates → {len(top)}):"]
        for i, idx in enumerate(top, 1):
            lines.append(f"\n[{i}] (maxsim: {maxsim_scores[idx]:.4f})")
            lines.append(candidate_list[idx].text)

        return HypothesisResult(
            results=reranked,
            context_prompt="\n".join(lines),
            metadata={
                "total_candidates": len(candidate_list),
                "from_passage": len(passage_doc_ids),
                "from_sentence": len(sentence_doc_ids),
                "unique_from_sentence": unique_from_sentence,
                "maxsim_scores": [maxsim_scores[i] for i in top],
            },
        )
