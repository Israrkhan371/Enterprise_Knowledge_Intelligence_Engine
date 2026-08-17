import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.deps import ensure_valid_uuid
from app.core.models import Document, UsageLog
from app.search.semantic import semantic_search
from app.search.keyword import keyword_search
from app.search.hybrid import hybrid_search
from app.search.metadata import metadata_search
from app.search.context_aware import rewrite_query
from app.rag.generate import generate_answer
from app.rag.citation_check import verify_citations
from app.rag.intelligence import compare_documents, compare_documents_full, summarize_document_full, suggest_document_updates
from app.graph.queries import (
    explain_relationship,
    get_skill_dependencies,
    get_technology_map,
    recommend_learning_path,
)
from app.evaluation.eval import run_evaluation

router = APIRouter(tags=["knowledge"])

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    query: str
    history: list[dict] = []
    user_id: str | None = None


class FeedbackRequest(BaseModel):
    was_helpful: bool


class FeedbackResponse(BaseModel):
    usage_log_id: str
    was_helpful: bool


class CompareRequest(BaseModel):
    document_id_a: str = Field(min_length=1)
    document_id_b: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_must_differ(self):
        if self.document_id_a == self.document_id_b:
            raise ValueError("document_id_a and document_id_b must be different documents")
        return self


class CompareResponse(BaseModel):
    similarity: float | None
    diff: list[str]
    summary: str


class SummaryResponse(BaseModel):
    summary: str


@router.get("/search/semantic")
def search_semantic(q: str, top_k: int = 10, category_filter: str | None = None, db: Session = Depends(get_db)):
    return semantic_search(q, top_k=top_k, category_filter=category_filter, db=db)


@router.get("/search/keyword")
def search_keyword(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    return keyword_search(db, q, top_k=top_k)


@router.get("/search/context-aware")
def search_context_aware(
    q: str,
    history: list[str] = Query(
        default=[],
        description=(
            "Prior turns as alternating role:content strings, oldest first, "
            "e.g. ['user: what is FastAPI', 'assistant: a Python web framework']. "
            "Only the last 4 are used (same window POST /ask uses)."
        ),
    ),
    top_k: int = 10,
    db: Session = Depends(get_db),
):
    """
    AI Search: Context-Aware Search. The other four search types
    (semantic/keyword/hybrid/metadata) are directly callable and stateless;
    this one is inherently stateful - it needs conversation history to do
    anything a stateless search doesn't already do. POST /ask uses this same
    rewrite_query() + hybrid_search() combination internally, but that
    endpoint bundles it into full answer generation. This is the standalone
    equivalent: rewrite the follow-up in isolation and run the resulting
    query through hybrid search, so context-aware retrieval can be exercised
    and tested on its own rather than only as a side effect of asking a
    question.
    """
    turns = []
    for entry in history:
        role, _, content = entry.partition(":")
        turns.append({"role": role.strip() or "user", "content": content.strip() or entry})

    rewritten_query = rewrite_query(q, turns)
    results = hybrid_search(db, rewritten_query, top_k=top_k)
    return {"original_query": q, "rewritten_query": rewritten_query, "results": results}


@router.get("/search/hybrid")
def search_hybrid(
    q: str,
    top_k: int = 10,
    category_filter: str | None = None,
    use_reranker: bool = True,
    db: Session = Depends(get_db),
):
    return hybrid_search(db, q, top_k=top_k, category_filter=category_filter, use_reranker=use_reranker)


@router.get("/search/metadata")
def search_metadata(
    category: str | None = None,
    source_type: str | None = None,
    status: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    top_k: int = 10,
    db: Session = Depends(get_db),
):
    return metadata_search(
        db,
        category=category,
        source_type=source_type,
        status=status,
        date_from=date_from,
        date_to=date_to,
        top_k=top_k,
    )


@router.post("/ask")
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    if payload.user_id is not None:
        ensure_valid_uuid(payload.user_id, status_code=400, detail="user_id must be a valid UUID")

    query = rewrite_query(payload.query, payload.history)
    try:
        result = generate_answer(db, query)
    except TimeoutError:
        logger.error("Answer generation timed out for query=%r.", payload.query)
        raise HTTPException(status_code=504, detail="LLM request timed out")

    verification = verify_citations(result["answer"], result["sources"])

    usage_log = UsageLog(
        user_id=payload.user_id,
        query=payload.query,
        answer=result["answer"],
        sources=result["sources"],
        retrieval_score=1.0 if verification["verified"] else 0.5,
        citation_verified=verification["verified"],
        citation_flags=verification["flags"],
        # Auto-flag unverified answers for the admin review queue; admins
        # can still clear or re-flag this via POST /admin/answers/{id}/review.
        flagged_for_review=not verification["verified"],
    )
    db.add(usage_log)
    db.commit()
    db.refresh(usage_log)

    return {**result, "citation_check": verification, "usage_log_id": usage_log.id}


@router.post("/ask/{usage_log_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(usage_log_id: str, payload: FeedbackRequest, db: Session = Depends(get_db)):
    """Lets a caller mark a previously generated answer (by the id returned
    from POST /ask) as helpful/unhelpful. This is what populates
    UsageLog.was_helpful, which GET /admin/analytics/usage reports on."""
    ensure_valid_uuid(usage_log_id, detail=f"usage log not found: {usage_log_id}")
    usage_log = db.get(UsageLog, usage_log_id)
    if not usage_log:
        raise HTTPException(status_code=404, detail=f"usage log not found: {usage_log_id}")

    usage_log.was_helpful = payload.was_helpful
    db.add(usage_log)
    db.commit()
    return FeedbackResponse(usage_log_id=usage_log_id, was_helpful=payload.was_helpful)


@router.post("/documents/compare", response_model=CompareResponse)
def compare(payload: CompareRequest, db: Session = Depends(get_db)):
    doc_a = db.get(Document, payload.document_id_a)
    doc_b = db.get(Document, payload.document_id_b)

    missing_ids = [
        doc_id for doc_id, doc in (
            (payload.document_id_a, doc_a),
            (payload.document_id_b, doc_b),
        ) if doc is None
    ]
    if missing_ids:
        raise HTTPException(status_code=404, detail=f"document(s) not found: {', '.join(missing_ids)}")

    try:
        return compare_documents_full(doc_a.raw_text or "", doc_b.raw_text or "")
    except TimeoutError:
        logger.error(
            "Document comparison timed out for document_id_a=%s, document_id_b=%s.",
            payload.document_id_a, payload.document_id_b,
        )
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception:
        logger.exception(
            "Document comparison failed unexpectedly for document_id_a=%s, document_id_b=%s.",
            payload.document_id_a, payload.document_id_b,
        )
        raise HTTPException(status_code=502, detail="Document comparison failed. Please try again later.")


@router.get("/documents/{document_id}/summary", response_model=SummaryResponse)
def summarize(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    try:
        return {"summary": summarize_document_full(doc.raw_text or "")}
    except TimeoutError:
        logger.error("Document summary timed out for document_id=%s.", document_id)
        raise HTTPException(status_code=504, detail="LLM request timed out")

@router.get("/documents/{document_id}/suggest-updates")
def suggest_updates(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    try:
        return suggest_document_updates(db, document_id)
    except TimeoutError:
        logger.error("Suggest-updates timed out for document_id=%s.", document_id)
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception:
        logger.exception("Suggest-updates failed unexpectedly for document_id=%s.", document_id)
        raise HTTPException(status_code=502, detail="Suggest-updates failed. Please try again later.")

@router.get("/graph/technology-map")
def technology_map(entity_label: str = "TECH"):
    return get_technology_map(entity_label)


@router.get("/graph/skill-dependencies")
def skill_dependencies(skill: str | None = None):
    return get_skill_dependencies(skill)


@router.get("/graph/relationships/explain")
def relationship_explanation(source: str, target: str):
    """Traceable-source explanation for one edge: relation type, confidence,
    reasoning and evidence — see app/graph/relationships.py."""
    return explain_relationship(source, target)


@router.get("/graph/learning-recommendations")
def learning_recommendations(user_query_history: list[str] = Query(default=[])):
    return recommend_learning_path(user_query_history)


@router.post("/evaluation/run")
def evaluation_run(k: int = 10, db: Session = Depends(get_db)):
    return run_evaluation(db, k=k)
