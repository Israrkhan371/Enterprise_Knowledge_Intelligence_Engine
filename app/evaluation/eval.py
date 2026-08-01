import json
import logging
from pathlib import Path

import mlflow
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.models import Document
from app.search.hybrid import hybrid_search

logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set() -> list[dict]:
    """
    Each entry: {"query": str, "relevant_document_titles": [str, ...]}
    (an optional "relevant_document_ids": [str, ...] is also supported and
    merged in, for the rare case a real document id is already known).

    Titles, not raw document ids, are the stable identifier here on purpose:
    Document.id is a random uuid4 assigned at ingest time (see gen_uuid() in
    app/core/models.py), so it can't be known ahead of ingestion and changes
    every time the corpus is wiped/re-seeded — an eval_set.json hand-filled
    with ids from one ingestion run silently stops matching anything after
    the next reset. Titles are resolved to whatever the *current* document
    ids are at evaluation time via resolve_relevant_ids() below, so this
    file stays valid across reseeds as long as the referenced documents
    exist under the same titles.
    """
    if not EVAL_SET_PATH.exists():
        return []
    return json.loads(EVAL_SET_PATH.read_text())


def resolve_relevant_ids(db: Session, item: dict) -> set[str]:
    """
    Combine any directly-supplied relevant_document_ids with ids looked up
    by relevant_document_titles against the live documents table. Missing
    titles are logged (not raised) so one typo/un-ingested doc doesn't
    crash the whole evaluation run — it just makes that query score lower,
    which is visible in the results rather than hidden by a crash.
    """
    relevant_ids = set(item.get("relevant_document_ids", []))
    titles = item.get("relevant_document_titles", [])
    if titles:
        rows = db.query(Document.id, Document.title).filter(Document.title.in_(titles)).all()
        found_titles = {title for _id, title in rows}
        relevant_ids.update(doc_id for doc_id, _title in rows)
        missing = set(titles) - found_titles
        if missing:
            logger.warning(
                "Eval query %r references %d document title(s) not found in the "
                "database (not yet ingested, or title drifted): %s",
                item.get("query"), len(missing), sorted(missing),
            )
    return relevant_ids


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
    skipped = 0

    with mlflow.start_run():
        for item in eval_set:
            relevant_ids = resolve_relevant_ids(db, item)
            if not relevant_ids:
                # No known-correct document for this query yet (typo'd
                # title, or the source document hasn't been ingested).
                # Silently scoring this as a 0 would drag precision/recall
                # down for a reason that has nothing to do with retrieval
                # quality, and would hide the real problem (a bad fixture
                # entry) inside what looks like a retrieval metric. Skip it
                # and surface the count instead.
                skipped += 1
                continue

            hits = hybrid_search(db, item["query"], top_k=k)
            retrieved_ids = [h.get("document_id", h.get("id")) for h in hits]

            precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
            recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
            rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))

        if not precisions:
            return {
                "error": "no eval queries had resolvable relevant documents — "
                         "check that eval_set.json titles match ingested Document.title values",
                "num_queries": len(eval_set),
                "skipped": skipped,
            }

        results = {
            "precision_at_k": round(sum(precisions) / len(precisions), 3),
            "recall_at_k": round(sum(recalls) / len(recalls), 3),
            "mrr": round(sum(rr_scores) / len(rr_scores), 3),
            "num_queries": len(precisions),
            "skipped": skipped,
            "k": k,
        }
        mlflow.log_metrics({k_: v for k_, v in results.items() if isinstance(v, (int, float))})

    return results
