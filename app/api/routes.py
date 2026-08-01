import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import Document, UsageLog
from app.search.semantic import semantic_search
from app.search.keyword import keyword_search
from app.search.hybrid import hybrid_search
from app.search.context_aware import rewrite_query
from app.rag.generate import generate_answer
from app.rag.citation_check import verify_citations
from app.rag.intelligence import compare_documents, compare_documents_full, summarize_document_full
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
def search_semantic(q: str, top_k: int = 10):
    return semantic_search(q, top_k=top_k)


@router.get("/search/keyword")
def search_keyword(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    return keyword_search(db, q, top_k=top_k)


@router.get("/search/hybrid")
def search_hybrid(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    return hybrid_search(db, q, top_k=top_k)


@router.post("/ask")
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    query = rewrite_query(payload.query, payload.history)
    try:
        result = generate_answer(db, query)
    except TimeoutError:
        logger.error("Answer generation timed out for query=%r.", payload.query)
        raise HTTPException(status_code=504, detail="LLM request timed out")

    verification = verify_citations(result["answer"], result["sources"])

    db.add(UsageLog(
        user_id=payload.user_id,
        query=payload.query,
        answer=result["answer"],
        retrieval_score=1.0 if verification["verified"] else 0.5,
    ))
    db.commit()

    return {**result, "citation_check": verification}


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
def learning_recommendations(user_query_history: list[str] = None):
    return recommend_learning_path(user_query_history or [])


@router.post("/evaluation/run")
def evaluation_run(k: int = 10, db: Session = Depends(get_db)):
    return run_evaluation(db, k=k)
