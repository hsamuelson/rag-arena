"""MS MARCO passage retrieval benchmark — the standard large-scale test.

8.8M passages, 7K dev queries with explicit passage-level relevance judgments.
This is the gold standard for passage retrieval evaluation at scale.

Published baselines (nDCG@10):
  BM25:        0.228
  DPR:         0.177
  ANCE:        0.330
  ColBERTv2:   0.397

Source: mteb/msmarco on HuggingFace

Since the full 8.8M corpus doesn't fit in memory with embeddings, this loader
supports smart sampling: load N queries, include all their gold passages,
then pad with random passages to reach max_corpus size. This preserves
evaluation validity while keeping memory manageable.
"""

import random

from .base import Benchmark, BenchmarkSample


MSMARCO_BASELINES = {
    "BM25": 0.228,
    "DPR": 0.177,
    "ANCE": 0.330,
    "ColBERTv2": 0.397,
}


class MSMarcoBenchmark(Benchmark):
    """MS MARCO passage retrieval benchmark with corpus sampling."""

    def __init__(
        self,
        max_corpus: int = 100_000,
        max_queries: int = 200,
        seed: int = 42,
    ):
        self._corpus: list[dict] = []
        self._samples: list[BenchmarkSample] = []
        self._loaded = False
        self._max_corpus = max_corpus
        self._max_queries = max_queries
        self._seed = seed

    @property
    def name(self) -> str:
        return "msmarco"

    @property
    def description(self) -> str:
        return f"MS MARCO passage retrieval ({self._max_corpus // 1000}K corpus, {self._max_queries} queries)"

    def categories(self) -> list[str]:
        return ["passage-retrieval"]

    def load(self, data_dir: str | None = None) -> None:
        if self._loaded:
            return
        self._load_from_huggingface()
        self._loaded = True

    def _load_from_huggingface(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Install 'datasets': pip install datasets")

        rng = random.Random(self._seed)

        # 1. Load qrels (dev split) to find queries with relevance judgments
        print("  [MS MARCO] Loading qrels (dev split)...")
        qrels_ds = load_dataset("mteb/msmarco", "default", split="dev")

        qrel_groups: dict[str, list[str]] = {}
        for row in qrels_ds:
            qid = str(row.get("query-id", row.get("query_id", "")))
            did = str(row.get("corpus-id", row.get("corpus_id", "")))
            score = row.get("score", 1)
            if score > 0:
                qrel_groups.setdefault(qid, []).append(did)

        # 2. Sample queries
        all_qids = list(qrel_groups.keys())
        rng.shuffle(all_qids)
        selected_qids = set(all_qids[: self._max_queries])
        print(f"  [MS MARCO] Selected {len(selected_qids)} queries from {len(all_qids)} available")

        # 3. Collect all gold passage IDs for selected queries
        gold_doc_ids: set[str] = set()
        for qid in selected_qids:
            gold_doc_ids.update(qrel_groups[qid])
        print(f"  [MS MARCO] Gold passages needed: {len(gold_doc_ids)}")

        # 4. Load queries
        print("  [MS MARCO] Loading queries...")
        queries_ds = load_dataset("mteb/msmarco", "queries", split="queries")
        query_map = {}
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in selected_qids:
                query_map[qid] = row["text"]

        # 5. Load corpus — stream to avoid loading all 8.8M into memory
        print(f"  [MS MARCO] Loading corpus (target: {self._max_corpus} docs)...")
        corpus_ds = load_dataset("mteb/msmarco", "corpus", split="corpus", streaming=True)

        corpus_map: dict[str, dict] = {}
        non_gold_ids: list[str] = []
        scanned = 0

        for row in corpus_ds:
            doc_id = str(row["_id"])
            title = row.get("title", "")
            text = row.get("text", "")
            full_text = f"{title}\n{text}" if title else text

            doc = {
                "id": doc_id,
                "text": full_text,
                "metadata": {"title": title},
            }

            if doc_id in gold_doc_ids:
                corpus_map[doc_id] = doc
            elif len(non_gold_ids) < self._max_corpus:
                corpus_map[doc_id] = doc
                non_gold_ids.append(doc_id)

            scanned += 1
            if scanned % 100_000 == 0:
                print(f"    Scanned {scanned} docs, collected {len(corpus_map)}...")

            # Stop once we have enough non-gold AND all gold docs
            if len(non_gold_ids) >= self._max_corpus and not (gold_doc_ids - set(corpus_map.keys())):
                break

        # If we still need more gold docs (shouldn't happen with streaming),
        # warn about it
        missing_gold = gold_doc_ids - set(corpus_map.keys())
        if missing_gold:
            print(f"  [MS MARCO] WARNING: {len(missing_gold)} gold docs not found in corpus stream")

        # 6. Trim non-gold docs if corpus is too large
        target_non_gold = self._max_corpus - len(gold_doc_ids & set(corpus_map.keys()))
        if len(non_gold_ids) > target_non_gold:
            rng.shuffle(non_gold_ids)
            remove_ids = set(non_gold_ids[target_non_gold:])
            for rid in remove_ids:
                corpus_map.pop(rid, None)

        self._corpus = list(corpus_map.values())
        print(f"  [MS MARCO] Final corpus: {len(self._corpus)} docs "
              f"({len(gold_doc_ids & set(corpus_map.keys()))} gold + "
              f"{len(self._corpus) - len(gold_doc_ids & set(corpus_map.keys()))} filler)")

        # 7. Build samples
        for qid in selected_qids:
            if qid in query_map:
                relevant_ids = qrel_groups[qid]
                # Only include queries whose gold docs are in the corpus
                if any(did in corpus_map for did in relevant_ids):
                    self._samples.append(BenchmarkSample(
                        question_id=f"msmarco_{qid}",
                        question=query_map[qid],
                        ground_truth="",
                        category="passage-retrieval",
                        corpus_doc_ids=relevant_ids,
                    ))

        print(f"  [MS MARCO] Final queries: {len(self._samples)}")

    def corpus(self) -> list[dict]:
        return self._corpus

    def samples(self) -> list[BenchmarkSample]:
        return self._samples
