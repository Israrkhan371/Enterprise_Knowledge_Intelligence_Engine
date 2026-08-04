import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import Document, DocumentChunk, Category, ApprovalLog
from app.ingestion.pipeline import ingest_document
from app.rag.intelligence import detect_duplicates, detect_outdated, detect_knowledge_gaps

router = APIRouter(prefix="/admin", tags=["admin"])

UPLOAD_DIR = Path("/tmp/ekie_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    status: str
    chunk_count: int


class DocumentSummaryResponse(BaseModel):
    """One row of GET /admin/documents — deliberately lighter than
    DocumentDetailResponse (no raw_text) so listing a large corpus stays
    cheap."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    source_type: str
    category_id: str | None
    status: str
    uploaded_by: str | None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentSummaryResponse]


class DocumentDetailResponse(DocumentSummaryResponse):
    quality_score: float | None
    chunk_count: int
    source_uri: str | None


class ApprovalResponse(BaseModel):
    document_id: str
    status: str


class CategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None


class UsageAnalyticsResponse(BaseModel):
    total_queries: int
    avg_retrieval_score: float | None
    helpful_count: int


def _chunk_count(db: Session, document_id: str) -> int:
    return db.query(func.count(DocumentChunk.id)).filter(
        DocumentChunk.document_id == document_id
    ).scalar() or 0


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    summary="Upload and ingest a document",
    description=(
        "Ingestion (loading, chunking, embedding, vector-store upsert, and "
        "knowledge-graph population) runs synchronously in this request — "
        "by the time this responds, the document is already chunked, "
        "embedded, and fully searchable via /search and /ask. "
        "`status` is set to 'pending' regardless: that field tracks "
        "*admin review*, not ingestion/processing state, and search/RAG "
        "currently do not filter by it — an uploaded document is queryable "
        "immediately, before any admin calls POST /documents/{id}/approve. "
        "Call POST /documents/{id}/approve to record a review decision."
    ),
)
def upload_document(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    category_id: str | None = Form(None),
    uploaded_by: str | None = Form(None),
    title: str | None = Form(None, description="Defaults to the uploaded filename if omitted."),
    db: Session = Depends(get_db),
):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document = Document(
        title=title or file.filename,
        source_type=source_type,
        category_id=category_id,
        uploaded_by=uploaded_by,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    document = ingest_document(db, document, str(dest))
    return DocumentUploadResponse(
        document_id=document.id,
        title=document.title,
        status=document.status,
        chunk_count=_chunk_count(db, document.id),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List uploaded documents",
    description=(
        "Primary way to discover document ids for POST /documents/{id}/approve "
        "or GET /documents/{id}, and to check ingestion status without "
        "querying Postgres or reading container logs directly."
    ),
)
def list_documents(
    status: str | None = None,
    source_type: str | None = None,
    category_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    if not (1 <= limit <= 200):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    query = db.query(Document)
    if status:
        query = query.filter(Document.status == status)
    if source_type:
        query = query.filter(Document.source_type == source_type)
    if category_id:
        query = query.filter(Document.category_id == category_id)

    total = query.count()
    rows = (
        query.order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return DocumentListResponse(total=total, limit=limit, offset=offset, documents=rows)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get one document's metadata and ingestion status",
)
def get_document(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return DocumentDetailResponse(
        id=document.id,
        title=document.title,
        source_type=document.source_type,
        category_id=document.category_id,
        status=document.status,
        uploaded_by=document.uploaded_by,
        created_at=document.created_at,
        updated_at=document.updated_at,
        quality_score=document.quality_score,
        chunk_count=_chunk_count(db, document.id),
        source_uri=document.source_uri,
    )


@router.post(
    "/documents/{document_id}/approve",
    response_model=ApprovalResponse,
    summary="Record an admin review decision for a document",
    description=(
        "Records the decision in approval_logs and sets Document.status to "
        "'approved' or 'rejected'. This does not currently gate search/RAG "
        "visibility (see the upload endpoint's description) — it's a review "
        "record, not a publish step."
    ),
)
def approve_document(document_id: str, reviewer: str, decision: str, comment: str = "", db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")

    document.status = "approved" if decision == "approved" else "rejected"
    db.add(document)
    db.add(ApprovalLog(document_id=document_id, reviewer=reviewer, decision=decision, comment=comment))
    db.commit()
    return ApprovalResponse(document_id=document_id, status=document.status)


@router.get("/categories", response_model=list[CategoryResponse])
def list_categories(db: Session = Depends(get_db)):
    return db.query(Category).all()


@router.post("/categories", response_model=CategoryResponse)
def create_category(name: str, description: str = "", db: Session = Depends(get_db)):
    category = Category(name=name, description=description)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


@router.get("/analytics/usage", response_model=UsageAnalyticsResponse)
def usage_analytics(db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT COUNT(*) AS total_queries,
               AVG(retrieval_score) AS avg_retrieval_score,
               SUM(CASE WHEN was_helpful THEN 1 ELSE 0 END) AS helpful_count
        FROM usage_logs
    """)).fetchone()
    return UsageAnalyticsResponse(
        total_queries=row.total_queries,
        avg_retrieval_score=round(row.avg_retrieval_score, 3) if row.avg_retrieval_score else None,
        helpful_count=row.helpful_count or 0,
    )


@router.get("/quality/duplicates")
def quality_duplicates(db: Session = Depends(get_db)):
    return detect_duplicates(db)


@router.get("/quality/outdated")
def quality_outdated(
    staleness_days: int = 180,
    llm_cross_check: bool = False,
    db: Session = Depends(get_db),
):
    return detect_outdated(db, staleness_days=staleness_days, llm_cross_check=llm_cross_check)


@router.get("/quality/gaps")
def quality_gaps(db: Session = Depends(get_db)):
    return detect_knowledge_gaps(db)