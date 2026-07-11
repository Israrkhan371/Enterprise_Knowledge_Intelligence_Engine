import json
from pathlib import Path

import mlflow
from sqlalchemy.orm import Session

from app.core.config import settings
from app.search.hybrid import hybrid_search

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    """
    Each entry: {"query": str, "relevant_document_ids": [str, ...]}
    Build this from real intern/mentor questions plus their known-correct
    source documents.
    """
    if not EVAL_SET_PATH.exists():
        return []
    return json.loads(EVAL_SET_PATH.read_text())


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def run_evaluation(db: Session, k: int = 10) -> dict:
    eval_set = load_eval_set()
    if not eval_set:
        return {"error": "no eval set found — add app/evaluation/eval_set.json"}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ekie-retrieval-eval")

    precisions, recalls, rr_scores = [], [], []

    with mlflow.start_run():
        for item in eval_set:
            hits = hybrid_search(db, item["query"], top_k=k)
            retrieved_ids = [h.get("document_id", h.get("id")) for h in hits]
            relevant_ids = set(item["relevant_document_ids"])

            precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
            recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
            rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))

        results = {
            "precision_at_k": round(sum(precisions) / len(precisions), 3),
            "recall_at_k": round(sum(recalls) / len(recalls), 3),
            "mrr": round(sum(rr_scores) / len(rr_scores), 3),
            "num_queries": len(eval_set),
            "k": k,
        }
        mlflow.log_metrics({k_: v for k_, v in results.items() if isinstance(v, (int, float))})

    return results
