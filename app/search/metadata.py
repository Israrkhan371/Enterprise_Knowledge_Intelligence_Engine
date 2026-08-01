"""
Metadata search: filters documents by category, source type, approval
status, and/or upload-date range — not by textual relevance to a query,
which is what semantic/keyword/hybrid search do instead. This was the
missing piece for the "Metadata Search" requirement; category_filter
already existed on semantic_search() for narrowing a relevance search,
but there was no way to list/filter documents by metadata alone (e.g.
"show me every SOP document uploaded in the last week").
"""
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.models import Document, Category


def metadata_search(
    db: Session,
    category: str | None = None,
    source_type: str | None = None,
    status: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    top_k: int = 10,
) -> list[dict]:
    query = db.query(Document)

    if category:
        query = query.join(Category, Document.category_id == Category.id).filter(Category.name == category)
    if source_type:
        query = query.filter(Document.source_type == source_type)
    if status:
        query = query.filter(Document.status == status)
    if date_from:
        query = query.filter(Document.created_at >= date_from)
    if date_to:
        query = query.filter(Document.created_at <= date_to)

    documents = query.order_by(Document.created_at.desc()).limit(top_k).all()

    return [
        {
            "id": str(doc.id),
            "title": doc.title,
            "source_type": doc.source_type,
            "category": doc.category.name if doc.category_id else None,
            "status": doc.status,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        }
        for doc in documents
    ]
