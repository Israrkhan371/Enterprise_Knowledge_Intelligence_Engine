from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.models import Document, UsageLog
from app.search.semantic import semantic_search
from app.search.keyword import keyword_search
from app.search.hybrid import hybrid_search
from app.search.metadata import metadata_search
from app.search.context_aware import rewrite_query
from app.rag.generate import generate_answer
from app.rag.citation_check import verify_citations
from app.rag.intelligence import compare_documents, summarize_document
from app.graph.queries import get_technology_map, get_skill_dependencies, recommend_learning_path
from app.evaluation.eval import run_evaluation

router = APIRouter(tags=["knowledge"])


class AskRequest(BaseModel):
    query: str
    history: list[dict] = []
    user_id: str | None = None


class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str


@router.get("/search/semantic")
def search_semantic(q: str, top_k: int = 10, category_filter: str | None = None):
    return semantic_search(q, top_k=top_k, category_filter=category_filter)


@router.get("/search/keyword")
def search_keyword(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    return keyword_search(db, q, top_k=top_k)


@router.get("/search/hybrid")
def search_hybrid(q: str, top_k: int = 10, category_filter: str | None = None, db: Session = Depends(get_db)):
    return hybrid_search(db, q, top_k=top_k, category_filter=category_filter)


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
    query = rewrite_query(payload.query, payload.history)
    result = generate_answer(db, query)
    verification = verify_citations(result["answer"], result["sources"])

    db.add(UsageLog(
        user_id=payload.user_id,
        query=payload.query,
        answer=result["answer"],
        retrieval_score=1.0 if verification["verified"] else 0.5,
    ))
    db.commit()

    return {**result, "citation_check": verification}


@router.post("/documents/compare")
def compare(payload: CompareRequest, db: Session = Depends(get_db)):
    doc_a = db.get(Document, payload.document_id_a)
    doc_b = db.get(Document, payload.document_id_b)
    if not doc_a or not doc_b:
        return {"error": "one or both documents not found"}
    return {"comparison": compare_documents(doc_a.raw_text or "", doc_b.raw_text or "")}


@router.get("/documents/{document_id}/summary")
def summarize(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        return {"error": "document not found"}
    return {"summary": summarize_document(doc.raw_text or "")}


@router.get("/graph/technology-map")
def technology_map(entity_label: str = "TECH"):
    return get_technology_map(entity_label)


@router.get("/graph/skill-dependencies")
def skill_dependencies(skill: str):
    return get_skill_dependencies(skill)


@router.get("/graph/learning-recommendations")
def learning_recommendations(user_query_history: list[str] = None):
    return recommend_learning_path(user_query_history or [])


@router.post("/evaluation/run")
def evaluation_run(k: int = 10, db: Session = Depends(get_db)):
    return run_evaluation(db, k=k)
