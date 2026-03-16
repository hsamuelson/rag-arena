"""Core scoring functions for RAG evaluation.

Provides both answer-quality metrics (EM, F1) and retrieval metrics
(recall@K, precision@K, nDCG@K, MRR).
"""

import re
import string
from collections import Counter
from dataclasses import dataclass, field


# ── Text normalisation (for answer comparison) ────────────────


def _normalise_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _get_tokens(text: str) -> list[str]:
    return _normalise_answer(text).split()


# ── Answer-quality metrics ────────────────────────────────────


def exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalisation. Returns 0.0 or 1.0."""
    return 1.0 if _normalise_answer(prediction) == _normalise_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(ground_truth)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Retrieval metrics ─────────────────────────────────────────


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant documents found in top-K retrieved."""
    if not relevant_ids:
        return 1.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-K retrieved documents that are relevant."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Binary relevance: 1 if doc is in relevant set, 0 otherwise.
    """
    import math

    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 1.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG (all relevant docs at top)
    ideal_k = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant document."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# ── Aggregate scorecard ──────────────────────────────────────


@dataclass
class ScoreCard:
    """Aggregate scores for an experiment run."""
    exact_match: float = 0.0
    token_f1: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    num_samples: int = 0
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "exact_match": round(self.exact_match, 4),
            "token_f1": round(self.token_f1, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "mrr": round(self.mrr, 4),
            "num_samples": self.num_samples,
            "latency_ms": round(self.latency_ms, 2),
            "category_scores": self.category_scores,
        }


def compute_scorecard(results: list[dict], k: int = 10) -> ScoreCard:
    """Compute aggregate scores from a list of per-question results.

    Each result dict should contain:
    - "prediction": str (LLM answer)
    - "ground_truth": str (expected answer)
    - "retrieved_ids": list[str] (IDs of retrieved docs)
    - "relevant_ids": list[str] (IDs of actually relevant docs)
    - "category": str (optional, for per-category breakdown)
    - "latency_ms": float (optional)
    """
    if not results:
        return ScoreCard()

    em_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    ndcg_scores = []
    mrr_scores = []
    latencies = []
    by_category: dict[str, list[dict]] = {}

    for r in results:
        em = exact_match(r["prediction"], r["ground_truth"])
        f1 = token_f1(r["prediction"], r["ground_truth"])
        rec = recall_at_k(r.get("retrieved_ids", []), r.get("relevant_ids", []), k)
        prec = precision_at_k(r.get("retrieved_ids", []), r.get("relevant_ids", []), k)
        ndcg = ndcg_at_k(r.get("retrieved_ids", []), r.get("relevant_ids", []), k)
        m = mrr(r.get("retrieved_ids", []), r.get("relevant_ids", []))

        em_scores.append(em)
        f1_scores.append(f1)
        recall_scores.append(rec)
        precision_scores.append(prec)
        ndcg_scores.append(ndcg)
        mrr_scores.append(m)

        if "latency_ms" in r:
            latencies.append(r["latency_ms"])

        cat = r.get("category", "all")
        by_category.setdefault(cat, []).append({
            "em": em, "f1": f1, "recall": rec, "precision": prec,
        })

    # Per-category aggregation
    category_scores = {}
    for cat, cat_results in by_category.items():
        n = len(cat_results)
        category_scores[cat] = {
            "exact_match": round(sum(r["em"] for r in cat_results) / n, 4),
            "token_f1": round(sum(r["f1"] for r in cat_results) / n, 4),
            "recall_at_k": round(sum(r["recall"] for r in cat_results) / n, 4),
            "num_samples": n,
        }

    n = len(results)
    return ScoreCard(
        exact_match=sum(em_scores) / n,
        token_f1=sum(f1_scores) / n,
        recall_at_k=sum(recall_scores) / n,
        precision_at_k=sum(precision_scores) / n,
        ndcg_at_k=sum(ndcg_scores) / n,
        mrr=sum(mrr_scores) / n,
        num_samples=n,
        category_scores=category_scores,
        latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )
