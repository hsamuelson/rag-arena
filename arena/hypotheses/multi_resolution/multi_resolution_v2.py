"""Multi-Resolution Associative Memory (MRAM) — Phase 2.

Builds on Phase 1.5 (sentence + passage + upgraded CE) by adding:
- Entity association layer: NER extraction + co-occurrence inverted index
- At query time: extract entities from query → find docs containing those
  entities → add to candidate pool alongside passage + sentence results
- This catches multi-hop connections that vector search misses

Algorithm:
1. Pre-build sentence-level index (same as Phase 1.5)
2. Pre-build entity index:
   - Run NER (spaCy) over all corpus documents
   - Build entity → [doc_ids] inverted index
   - Build entity co-occurrence map (entities in same doc are associated)
3. At query time, retrieve from 3 sources:
   - Passage level: standard top-50 from backend (BM25 + dense hybrid)
   - Sentence level: top-50 sentences by cosine sim → map to parent docs
   - Entity level: extract entities from query → find docs containing them
     + docs containing co-occurring entities (1-hop expansion)
4. Merge candidates (deduplicate by doc_id)
5. CE rerank merged pool to top-10
"""

import re
from collections import defaultdict

import numpy as np

from ..base import Hypothesis, HypothesisResult
from ...backends.base import RetrievalResult

_MAX_CHARS = 2000
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class MultiResolutionV2Hypothesis(Hypothesis):
    """MRAM Phase 2: multi-resolution + entity association."""

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        pool_per_level=50,
        entity_pool=30,
        final_k=10,
    ):
        self._model_name = model_name
        self._pool_per_level = pool_per_level
        self._entity_pool = entity_pool
        self._final_k = final_k
        self._model = None
        self._backend = None
        self._nlp = None

        # Sentence-level index
        self._sentence_embeddings = None  # (N_sentences, dim) float32
        self._sentence_texts = []
        self._sentence_parent_idx = []
        self._sentence_parent_id = []

        # Entity index
        self._entity_to_docs = defaultdict(set)    # entity_text → {doc_id, ...}
        self._doc_to_entities = defaultdict(set)    # doc_id → {entity_text, ...}
        self._entity_cooccurrence = defaultdict(set)  # entity → {co-occurring entities}

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

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError:
                print("  [MRAM] Downloading spaCy model en_core_web_sm...")
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        return self._nlp

    @property
    def name(self):
        return "multi-resolution-v2"

    @property
    def description(self):
        return (
            f"MRAM Phase 2: sentence+passage+entity retrieval, "
            f"CE rerank ({self._model_name.split('/')[-1]}) to {self._final_k}"
        )

    def _extract_entities(self, text, max_chars=5000):
        """Extract named entities from text using spaCy."""
        nlp = self._get_nlp()
        doc = nlp(text[:max_chars])
        entities = set()
        for ent in doc.ents:
            # Normalize: lowercase, strip whitespace
            normalized = ent.text.strip().lower()
            if len(normalized) >= 2:
                entities.add(normalized)
        return entities

    def _build_index(self):
        """Build sentence-level and entity indices from backend corpus."""
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

        # ─── Sentence-level index ───
        print(f"  [MRAM] Building sentence-level index over {n_docs} docs...")
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

        # Embed in batches, normalize, float32
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

        # ─── Entity association index ───
        print(f"  [MRAM] Building entity index (NER over {n_docs} docs)...")
        nlp = self._get_nlp()

        # Process in batches using spaCy pipe for speed
        batch_size_ner = 256
        doc_entities_list = []

        for batch_start in range(0, n_docs, batch_size_ner):
            batch_end = min(batch_start + batch_size_ner, n_docs)
            batch_texts = [t[:5000] for t in corpus_texts[batch_start:batch_end]]
            batch_docs = list(nlp.pipe(batch_texts, batch_size=batch_size_ner))

            for local_idx, spacy_doc in enumerate(batch_docs):
                doc_idx = batch_start + local_idx
                doc_id = corpus_ids[doc_idx]

                entities = set()
                for ent in spacy_doc.ents:
                    normalized = ent.text.strip().lower()
                    if len(normalized) >= 2:
                        entities.add(normalized)

                doc_entities_list.append((doc_id, entities))

                # Build inverted index
                for entity in entities:
                    self._entity_to_docs[entity].add(doc_id)
                self._doc_to_entities[doc_id] = entities

            if (batch_start // batch_size_ner) % 10 == 0:
                print(f"    [{batch_start}/{n_docs}] NER processed...")

        # Build co-occurrence map: entities that appear in the same doc are associated
        for doc_id, entities in doc_entities_list:
            entity_list = list(entities)
            for i, e1 in enumerate(entity_list):
                for e2 in entity_list[i + 1:]:
                    self._entity_cooccurrence[e1].add(e2)
                    self._entity_cooccurrence[e2].add(e1)

        total_entities = len(self._entity_to_docs)
        avg_per_doc = np.mean([len(ents) for _, ents in doc_entities_list]) if doc_entities_list else 0
        print(f"  [MRAM] Entity index: {total_entities} unique entities, "
              f"avg {avg_per_doc:.1f} per doc")

        self._index_built = True
        print(f"  [MRAM] Phase 2 index ready (sentence + entity)")

    def _retrieve_sentence_level(self, query_emb, top_k):
        """Retrieve top-K parent documents by sentence similarity."""
        if self._sentence_embeddings is None or len(self._sentence_texts) == 0:
            return {}

        q = query_emb / max(np.linalg.norm(query_emb), 1e-12)
        sims = self._sentence_embeddings @ q.astype(np.float32)

        n_candidates = min(top_k * 3, len(sims))
        top_sent_indices = np.argpartition(sims, -n_candidates)[-n_candidates:]
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

    def _retrieve_entity_level(self, query, top_k):
        """Retrieve documents via entity matching + 1-hop co-occurrence expansion."""
        if not self._entity_to_docs:
            return {}, set()

        # Extract entities from query
        query_entities = self._extract_entities(query)
        if not query_entities:
            return {}, set()

        # Direct entity matches: docs containing query entities
        matched_docs = defaultdict(float)
        for entity in query_entities:
            if entity in self._entity_to_docs:
                for doc_id in self._entity_to_docs[entity]:
                    matched_docs[doc_id] += 1.0  # count matching entities

        # 1-hop expansion: find co-occurring entities and their docs
        expanded_entities = set()
        for entity in query_entities:
            if entity in self._entity_cooccurrence:
                # Take top co-occurring entities (limit to avoid explosion)
                cooccurring = self._entity_cooccurrence[entity]
                # Score by how many query entities they co-occur with
                for co_ent in cooccurring:
                    expanded_entities.add(co_ent)

        # Add docs from expanded entities with lower weight
        for entity in expanded_entities - query_entities:
            if entity in self._entity_to_docs:
                for doc_id in self._entity_to_docs[entity]:
                    matched_docs[doc_id] += 0.3  # lower weight for 1-hop

        # Sort by match count, return top_k
        sorted_docs = sorted(matched_docs.items(), key=lambda x: x[1], reverse=True)
        doc_scores = dict(sorted_docs[:top_k])

        return doc_scores, query_entities

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

        # ─── Collect candidates from all levels ───
        candidates = {}  # doc_id → RetrievalResult
        for r in results:
            candidates[r.doc_id] = r

        passage_doc_ids = set(candidates.keys())
        sentence_doc_ids = set()
        entity_doc_ids = set()

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

        # ─── Level 3: Entity-level candidates ───
        entity_scores, query_entities = self._retrieve_entity_level(
            query, self._entity_pool
        )
        for doc_id, score in entity_scores.items():
            entity_doc_ids.add(doc_id)
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
            if doc_id in entity_doc_ids:
                sources.append("entity")
            source_info[doc_id] = sources

        unique_from_sentence = sum(
            1 for d in reranked
            if d.doc_id not in passage_doc_ids and d.doc_id in sentence_doc_ids
        )
        unique_from_entity = sum(
            1 for d in reranked
            if d.doc_id not in passage_doc_ids
            and d.doc_id not in sentence_doc_ids
            and d.doc_id in entity_doc_ids
        )

        lines = [f"Retrieved context (MRAM-v2, {len(candidate_list)} candidates → {len(top)}):"]
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
                "from_entity": len(entity_doc_ids),
                "unique_from_sentence": unique_from_sentence,
                "unique_from_entity": unique_from_entity,
                "query_entities": list(query_entities) if query_entities else [],
                "total_unique_entities": len(self._entity_to_docs),
                "ce_scores": [ce_scores[i] for i in top],
                "source_info": source_info,
            },
        )
