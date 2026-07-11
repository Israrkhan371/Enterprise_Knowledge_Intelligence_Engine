from sqlalchemy import text
from sqlalchemy.orm import Session


def keyword_search(db: Session, query: str, top_k: int = 10) -> list[dict]:
    """
    Uses Postgres full-text search (to_tsvector/plainto_tsquery). Requires a
    GIN index on document_chunks.text for production performance:
        CREATE INDEX chunks_fts_idx ON document_chunks
        USING GIN (to_tsvector('english', text));
    """
    sql = text("""
        SELECT id, document_id, text,
               ts_rank(to_tsvector('english', text), plainto_tsquery('english', :query)) AS rank
        FROM document_chunks
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :top_k
    """)
    rows = db.execute(sql, {"query": query, "top_k": top_k}).fetchall()
    return [
        {"id": str(r.id), "document_id": str(r.document_id), "text": r.text, "rank": float(r.rank)}
        for r in rows
    ]
