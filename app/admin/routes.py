import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.models import Document, Category, ApprovalLog
from app.ingestion.pipeline import ingest_document
from app.rag.intelligence import detect_duplicates, detect_outdated, detect_knowledge_gaps

router = APIRouter(prefix="/admin", tags=["admin"])

UPLOAD_DIR = Path("/tmp/ekie_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    source_type: str = Form(...),
    category_id: str | None = Form(None),
    uploaded_by: str | None = Form(None),
    db: Session = Depends(get_db),
):
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    document = Document(
        title=file.filename,
        source_type=source_type,
        category_id=category_id,
        uploaded_by=uploaded_by,
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    document = ingest_document(db, document, str(dest))
    return {"document_id": document.id, "status": document.status}


@router.post("/documents/{document_id}/approve")
def approve_document(document_id: str, reviewer: str, decision: str, comment: str = "", db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if not document:
        return {"error": "document not found"}

    document.status = "approved" if decision == "approved" else "rejected"
    db.add(document)
    db.add(ApprovalLog(document_id=document_id, reviewer=reviewer, decision=decision, comment=comment))
    db.commit()
    return {"document_id": document_id, "status": document.status}


@router.get("/categories")
def list_categories(db: Session = Depends(get_db)):
    return db.query(Category).all()


@router.post("/categories")
def create_category(name: str, description: str = "", db: Session = Depends(get_db)):
    category = Category(name=name, description=description)
    db.add(category)
    db.commit()
    db.refresh(category)
    return category


@router.get("/analytics/usage")
def usage_analytics(db: Session = Depends(get_db)):
    row = db.execute(text("""
        SELECT COUNT(*) AS total_queries,
               AVG(retrieval_score) AS avg_retrieval_score,
               SUM(CASE WHEN was_helpful THEN 1 ELSE 0 END) AS helpful_count
        FROM usage_logs
    """)).fetchone()
    return {
        "total_queries": row.total_queries,
        "avg_retrieval_score": round(row.avg_retrieval_score, 3) if row.avg_retrieval_score else None,
        "helpful_count": row.helpful_count,
    }


@router.get("/quality/duplicates")
def quality_duplicates(db: Session = Depends(get_db)):
    return detect_duplicates(db)


@router.get("/quality/outdated")
def quality_outdated(db: Session = Depends(get_db)):
    return detect_outdated(db)


@router.get("/quality/gaps")
def quality_gaps(db: Session = Depends(get_db)):
    return detect_knowledge_gaps(db)
