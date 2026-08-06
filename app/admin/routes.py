import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import require_admin, ensure_valid_uuid
from app.core.models import Document, DocumentChunk, Category, ApprovalLog, UsageLog, AnswerReviewLog, User
from app.graph.coverage import detect_missing_knowledge
from app.ingestion.pipeline import ingest_document, ingest_github_repo
from app.rag.intelligence import detect_duplicates, detect_outdated, detect_knowledge_gaps, suggest_document_updates
from app.rag.quality import score_all_documents, score_document_quality
from app.rag.version_intelligence import (
    VersionLinkError,
    detect_version_candidates,
    get_version_history,
    link_version,
)

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

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
    version: int
    supersedes_id: str | None


class VersionCandidateResponse(BaseModel):
    document_id: str
    title: str
    similarity: float
    status: str
    version: int


class VersionLinkResponse(BaseModel):
    document_id: str
    supersedes_id: str
    version: int
    status: str


class VersionHistoryEntry(BaseModel):
    document_id: str
    title: str
    version: int
    status: str
    created_at: str | None
    is_current: bool


class QualityScoreResponse(BaseModel):
    document_id: str
    title: str
    overall_score: float
    completeness_score: float
    freshness_score: float
    originality_score: float
    word_count: int


class MissingKnowledgeAlert(BaseModel):
    entity: str
    label: str
    mentioned_in_document_count: int
    mentioning_documents: list[str]


class ApprovalResponse(BaseModel):
    document_id: str
    status: str


class CategoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None


class AnswerSummaryResponse(BaseModel):
    """One row of GET /admin/answers — no `answer`/`sources`/`citation_flags`
    body, mirroring why DocumentSummaryResponse leaves out raw_text."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str | None
    query: str
    retrieval_score: float | None
    citation_verified: bool | None
    was_helpful: bool | None
    flagged_for_review: bool
    reviewed: bool
    created_at: datetime


class AnswerListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    answers: list[AnswerSummaryResponse]


class AnswerDetailResponse(AnswerSummaryResponse):
    answer: str | None
    sources: list[dict] | None
    citation_flags: list[dict] | None


class AnswerReviewLogEntry(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    reviewer: str | None
    decision: str
    comment: str | None
    created_at: datetime


class AnswerReviewResponse(BaseModel):
    usage_log_id: str
    reviewed: bool
    flagged_for_review: bool
    decision: str


class UsageAnalyticsResponse(BaseModel):
    total_queries: int
    avg_retrieval_score: float | None
    helpful_count: int
    unhelpful_count: int
    no_feedback_count: int
    verified_count: int
    flagged_for_review_count: int
    reviewed_count: int
    pending_review_count: int


class UsageTimeseriesPoint(BaseModel):
    date: str
    query_count: int
    helpful_count: int
    flagged_count: int


class TopQueryEntry(BaseModel):
    query: str
    occurrences: int


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
    title: str | None = Form(None, description="Defaults to the uploaded filename if omitted."),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    # Use a generated filename, not the client-supplied one: file.filename
    # is attacker-controlled input, and writing straight to
    # UPLOAD_DIR / file.filename both allows path traversal (e.g. a
    # filename containing "../..") and lets two concurrent uploads with
    # the same name silently clobber each other. The extension is kept
    # because some loaders (transcripts, zip archives) branch on
    # Path(path).suffix.
    original_suffix = Path(file.filename or "").suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{original_suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document = Document(
        title=title or file.filename,
        source_type=source_type,
        category_id=category_id,
        uploaded_by=admin.email,
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


class GithubIngestResponse(BaseModel):
    document_ids: list[str]
    file_count: int


@router.post(
    "/documents/ingest-github",
    response_model=GithubIngestResponse,
    summary="Ingest a GitHub repository as a Knowledge Source",
    description=(
        "Knowledge Source: GitHub Repositories. Fans the repo out into one "
        "Document row per file (source_type='github'), each chunked, "
        "embedded, and graph-populated the same as any other upload - "
        "distinct from POST /documents/upload, which takes a single file "
        "and can't represent a whole repo as one Document. github_token is "
        "only needed for private repositories; omit it for public ones."
    ),
)
def ingest_github(
    repo_url: str = Form(...),
    category_id: str | None = Form(None),
    github_token: str | None = Form(None),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    documents = ingest_github_repo(
        db, repo_url, category_id=category_id, uploaded_by=admin.email, github_token=github_token,
    )
    return GithubIngestResponse(document_ids=[d.id for d in documents], file_count=len(documents))


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
        version=document.version,
        supersedes_id=document.supersedes_id,
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
def approve_document(document_id: str, decision: str, comment: str = "", admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")

    document.status = "approved" if decision == "approved" else "rejected"
    db.add(document)
    db.add(ApprovalLog(document_id=document_id, reviewer=admin.email, decision=decision, comment=comment))
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


@router.get(
    "/answers",
    response_model=AnswerListResponse,
    summary="List AI-generated answers for review",
    description=(
        "Primary discovery endpoint for the 'review AI answers' admin queue. "
        "Backed by usage_logs (one row per POST /ask call). Use "
        "flagged_for_review=true to see answers that failed citation "
        "verification or were flagged by an admin, and reviewed=false to "
        "see what's still awaiting a decision."
    ),
)
def list_answers(
    flagged_for_review: bool | None = None,
    reviewed: bool | None = None,
    was_helpful: bool | None = None,
    min_score: float | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    if not (1 <= limit <= 200):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")

    query = db.query(UsageLog)
    if flagged_for_review is not None:
        query = query.filter(UsageLog.flagged_for_review == flagged_for_review)
    if reviewed is not None:
        query = query.filter(UsageLog.reviewed == reviewed)
    if was_helpful is not None:
        query = query.filter(UsageLog.was_helpful == was_helpful)
    if min_score is not None:
        query = query.filter(UsageLog.retrieval_score >= min_score)
    if search:
        query = query.filter(UsageLog.query.ilike(f"%{search}%"))

    total = query.count()
    rows = (
        query.order_by(UsageLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return AnswerListResponse(total=total, limit=limit, offset=offset, answers=rows)


@router.get(
    "/answers/{usage_log_id}",
    response_model=AnswerDetailResponse,
    summary="Get one AI answer's full text, sources, and citation verification",
)
def get_answer(usage_log_id: str, db: Session = Depends(get_db)):
    ensure_valid_uuid(usage_log_id, detail=f"answer not found: {usage_log_id}")
    usage_log = db.get(UsageLog, usage_log_id)
    if not usage_log:
        raise HTTPException(status_code=404, detail=f"answer not found: {usage_log_id}")
    return usage_log


@router.get(
    "/answers/{usage_log_id}/review-history",
    response_model=list[AnswerReviewLogEntry],
    summary="Get the audit trail of admin review decisions for one answer",
)
def get_answer_review_history(usage_log_id: str, db: Session = Depends(get_db)):
    ensure_valid_uuid(usage_log_id, detail=f"answer not found: {usage_log_id}")
    usage_log = db.get(UsageLog, usage_log_id)
    if not usage_log:
        raise HTTPException(status_code=404, detail=f"answer not found: {usage_log_id}")
    return (
        db.query(AnswerReviewLog)
        .filter(AnswerReviewLog.usage_log_id == usage_log_id)
        .order_by(AnswerReviewLog.created_at.desc())
        .all()
    )


@router.post(
    "/answers/{usage_log_id}/review",
    response_model=AnswerReviewResponse,
    summary="Record an admin review decision for an AI answer",
    description=(
        "Records the decision in answer_review_logs and updates the "
        "answer's `reviewed`/`flagged_for_review` state. decision must be "
        "one of: approved (looks correct, clears any flag), flagged "
        "(needs a fix or follow-up — e.g. bad citation, wrong answer), "
        "or dismissed (reviewed, no action needed, leave flag as-is)."
    ),
)
def review_answer(usage_log_id: str, decision: str, comment: str = "", admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    ensure_valid_uuid(usage_log_id, detail=f"answer not found: {usage_log_id}")
    usage_log = db.get(UsageLog, usage_log_id)
    if not usage_log:
        raise HTTPException(status_code=404, detail=f"answer not found: {usage_log_id}")
    if decision not in ("approved", "flagged", "dismissed"):
        raise HTTPException(status_code=400, detail="decision must be one of: approved, flagged, dismissed")

    usage_log.reviewed = True
    if decision == "approved":
        usage_log.flagged_for_review = False
    elif decision == "flagged":
        usage_log.flagged_for_review = True
    # "dismissed" leaves flagged_for_review as-is.

    db.add(usage_log)
    db.add(AnswerReviewLog(usage_log_id=usage_log_id, reviewer=admin.email, decision=decision, comment=comment))
    db.commit()
    return AnswerReviewResponse(
        usage_log_id=usage_log_id,
        reviewed=True,
        flagged_for_review=usage_log.flagged_for_review,
        decision=decision,
    )


@router.get("/analytics/usage", response_model=UsageAnalyticsResponse)
def usage_analytics(db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT COUNT(*) AS total_queries,
               AVG(retrieval_score) AS avg_retrieval_score,
               SUM(CASE WHEN was_helpful IS TRUE THEN 1 ELSE 0 END) AS helpful_count,
               SUM(CASE WHEN was_helpful IS FALSE THEN 1 ELSE 0 END) AS unhelpful_count,
               SUM(CASE WHEN was_helpful IS NULL THEN 1 ELSE 0 END) AS no_feedback_count,
               SUM(CASE WHEN citation_verified IS TRUE THEN 1 ELSE 0 END) AS verified_count,
               SUM(CASE WHEN flagged_for_review IS TRUE THEN 1 ELSE 0 END) AS flagged_for_review_count,
               SUM(CASE WHEN reviewed IS TRUE THEN 1 ELSE 0 END) AS reviewed_count,
               SUM(CASE WHEN reviewed IS FALSE THEN 1 ELSE 0 END) AS pending_review_count
        FROM usage_logs
    """)).fetchone()
    return UsageAnalyticsResponse(
        total_queries=row.total_queries,
        avg_retrieval_score=round(row.avg_retrieval_score, 3) if row.avg_retrieval_score is not None else None,
        helpful_count=row.helpful_count or 0,
        unhelpful_count=row.unhelpful_count or 0,
        no_feedback_count=row.no_feedback_count or 0,
        verified_count=row.verified_count or 0,
        flagged_for_review_count=row.flagged_for_review_count or 0,
        reviewed_count=row.reviewed_count or 0,
        pending_review_count=row.pending_review_count or 0,
    )


@router.get(
    "/analytics/usage/timeseries",
    response_model=list[UsageTimeseriesPoint],
    summary="Daily query volume for usage-analytics charts",
)
def usage_analytics_timeseries(days: int = 14, db: Session = Depends(get_db)):
    if not (1 <= days <= 365):
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")

    rows = db.execute(
        text("""
            SELECT DATE(created_at) AS day,
                   COUNT(*) AS query_count,
                   SUM(CASE WHEN was_helpful IS TRUE THEN 1 ELSE 0 END) AS helpful_count,
                   SUM(CASE WHEN flagged_for_review IS TRUE THEN 1 ELSE 0 END) AS flagged_count
            FROM usage_logs
            WHERE created_at >= NOW() - (:days || ' days')::interval
            GROUP BY DATE(created_at)
            ORDER BY day
        """),
        {"days": days},
    ).fetchall()
    return [
        UsageTimeseriesPoint(
            date=str(r.day),
            query_count=r.query_count,
            helpful_count=r.helpful_count or 0,
            flagged_count=r.flagged_count or 0,
        )
        for r in rows
    ]


@router.get(
    "/analytics/usage/top-queries",
    response_model=list[TopQueryEntry],
    summary="Most frequently asked queries, for surfacing FAQ candidates",
)
def usage_top_queries(limit: int = 10, db: Session = Depends(get_db)):
    if not (1 <= limit <= 100):
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    rows = db.execute(
        text("""
            SELECT MAX(query) AS query, COUNT(*) AS occurrences
            FROM usage_logs
            GROUP BY lower(trim(query))
            ORDER BY occurrences DESC, MAX(created_at) DESC
            LIMIT :limit
        """),
        {"limit": limit},
    ).fetchall()
    return [TopQueryEntry(query=r.query, occurrences=r.occurrences) for r in rows]


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


class RelatedDocumentEntry(BaseModel):
    document_id: str
    title: str
    updated_at: str


class SuggestUpdatesResponse(BaseModel):
    document_id: str
    title: str
    is_outdated: bool
    related_documents: list[RelatedDocumentEntry]
    suggested_updates: str


@router.get(
    "/documents/{document_id}/suggest-updates",
    response_model=SuggestUpdatesResponse,
    summary="AI-suggested updates for one document, grounded in fresher related documents",
    description=(
        "AI capability: Suggest document updates. Distinct from GET "
        "/quality/outdated (a time-based flag with no content) and GET "
        "/documents/{id}/version-candidates (structural version-linking "
        "suggestions with no content either) - this looks at what other, "
        "more recently updated documents in the corpus say on the same "
        "topic and asks the LLM what the target document may be missing or "
        "have superseded, citing only those related documents rather than "
        "unsourced model knowledge. Returns a plain no-comparison-available "
        "result if no fresher related document exists."
    ),
)
def document_suggest_updates(document_id: str, db: Session = Depends(get_db)):
    result = suggest_document_updates(db, document_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return result


@router.post(
    "/documents/{document_id}/score-quality",
    response_model=QualityScoreResponse,
    summary="Score one document's documentation quality",
    description=(
        "Bonus: AI Documentation Quality Scoring. Deterministic 0-100 "
        "breakdown (completeness/freshness/originality) persisted to "
        "Document.quality_score."
    ),
)
def document_score_quality(document_id: str, db: Session = Depends(get_db)):
    breakdown = score_document_quality(db, document_id)
    if breakdown is None:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return breakdown


@router.post(
    "/quality/score-all",
    response_model=list[QualityScoreResponse],
    summary="Score every document's documentation quality",
    description="Runs score-quality across the whole corpus, worst-first, and persists each Document.quality_score.",
)
def quality_score_all(db: Session = Depends(get_db)):
    return score_all_documents(db)


@router.get(
    "/quality/missing-knowledge",
    response_model=list[MissingKnowledgeAlert],
    summary="Topics the corpus talks about a lot but has no dedicated document for",
    description=(
        "Knowledge Intelligence: Missing Knowledge Alerts. Entities mentioned "
        "by at least min_mentions distinct documents with no document title "
        "matching them, sorted by mention count descending."
    ),
)
def quality_missing_knowledge(min_mentions: int = 3, entity_label: str | None = None):
    return detect_missing_knowledge(min_mentions=min_mentions, entity_label=entity_label)


@router.get(
    "/documents/{document_id}/version-candidates",
    response_model=list[VersionCandidateResponse],
    summary="Suggest documents that may be an earlier/later version of this one",
    description=(
        "Bonus: Document Version Intelligence. Ranks other documents by "
        "full-text embedding similarity to this one, in the band between "
        "'clearly unrelated' and 'exact duplicate' (see GET /quality/duplicates "
        "for the latter). These are suggestions only - confirm one with "
        "POST /documents/{document_id}/link-version."
    ),
)
def document_version_candidates(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return detect_version_candidates(db, document_id)


@router.post(
    "/documents/{document_id}/link-version",
    response_model=VersionLinkResponse,
    summary="Record that this document is a newer version of another",
    description=(
        "Sets document_id.supersedes_id, bumps its version number, and "
        "marks the superseded document status='stale'. Rejects self-links "
        "and links that would create a cycle."
    ),
)
def document_link_version(document_id: str, supersedes_id: str, db: Session = Depends(get_db)):
    try:
        document = link_version(db, document_id, supersedes_id)
    except VersionLinkError as exc:
        status_code = 404 if "not found" in str(exc) else 400
        raise HTTPException(status_code=status_code, detail=str(exc))
    return VersionLinkResponse(
        document_id=document.id,
        supersedes_id=document.supersedes_id,
        version=document.version,
        status=document.status,
    )


@router.get(
    "/documents/{document_id}/version-history",
    response_model=list[VersionHistoryEntry],
    summary="Full version chain this document belongs to, oldest first",
)
def document_version_history(document_id: str, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    return get_version_history(db, document_id)