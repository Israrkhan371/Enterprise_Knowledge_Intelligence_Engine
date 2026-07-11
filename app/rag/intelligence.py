from datetime import datetime, timedelta

import anthropic
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.embeddings.embedder import embed_texts

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def detect_duplicates(db: Session, similarity_threshold: float = 0.92) -> list[dict]:
    """Flags document pairs whose chunk embeddings are near-identical."""
    rows = db.execute(text("SELECT id, document_id, text FROM document_chunks")).fetchall()
    if len(rows) < 2:
        return []

    texts_ = [r.text for r in rows]
    vectors = embed_texts(texts_)

    duplicates = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if rows[i].document_id == rows[j].document_id:
                continue
            sim = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            if sim >= similarity_threshold:
                duplicates.append({
                    "document_a": str(rows[i].document_id),
                    "document_b": str(rows[j].document_id),
                    "similarity": round(sim, 3),
                })
    return duplicates


def detect_outdated(db: Session, staleness_days: int = 180) -> list[dict]:
    """Flags documents not updated within the staleness window."""
    cutoff = datetime.utcnow() - timedelta(days=staleness_days)
    rows = db.execute(
        text("SELECT id, title, updated_at FROM documents WHERE updated_at < :cutoff"),
        {"cutoff": cutoff},
    ).fetchall()
    return [{"document_id": str(r.id), "title": r.title, "last_updated": r.updated_at.isoformat()} for r in rows]


def compare_documents(text_a: str, text_b: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"Compare these two documents. Summarize key differences and overlaps.\n\nDocument A:\n{text_a[:4000]}\n\nDocument B:\n{text_b[:4000]}",
        }],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def summarize_document(text_: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Summarize this document in 4-6 sentences:\n\n{text_[:6000]}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


def detect_knowledge_gaps(db: Session, min_score_threshold: float = 0.3, min_occurrences: int = 3) -> list[dict]:
    """
    Bonus: Knowledge Gap Detection. Groups usage_logs by low retrieval_score
    to surface recurring queries the knowledge base can't answer well.
    """
    rows = db.execute(
        text("""
            SELECT query, COUNT(*) AS occurrences, AVG(retrieval_score) AS avg_score
            FROM usage_logs
            WHERE retrieval_score IS NOT NULL
            GROUP BY query
            HAVING AVG(retrieval_score) < :threshold AND COUNT(*) >= :min_occ
            ORDER BY occurrences DESC
        """),
        {"threshold": min_score_threshold, "min_occ": min_occurrences},
    ).fetchall()
    return [{"query": r.query, "occurrences": r.occurrences, "avg_score": round(r.avg_score, 3)} for r in rows]
