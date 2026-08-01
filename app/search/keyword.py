from sqlalchemy import text
from sqlalchemy.orm import Session


def keyword_search(db: Session, query: str, top_k: int = 10) -> list[dict]:
    """
    Uses Postgres full-text search (to_tsvector/plainto_tsquery). Requires a
    GIN index on document_chunks.text for production performance:
        CREATE INDEX chunks_fts_idx ON document_chunks
        USING GIN (to_tsvector('english', text));
    (declared on DocumentChunk in app/core/models.py, confirmed live via
    pg_indexes 2026-07-30.)
    """
    sql = text("""
        SELECT id, document_id, embedding_id, text,
               ts_rank(to_tsvector('english', text), plainto_tsquery('english', :query)) AS rank
        FROM document_chunks
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :top_k
    """)
    rows = db.execute(sql, {"query": query, "top_k": top_k}).fetchall()
    return [
        {
            "id": str(r.id),
            "document_id": str(r.document_id),
            # Same value semantic_search() calls "id" (the ChromaDB vector
            # id from upsert_chunks) — None for chunks that haven't been
            # embedded yet. Included so hybrid_search() can fuse semantic
            # and keyword hits for the *same chunk* on a shared key; see
            # the comment in app/search/hybrid.py for why that mattered.
            "embedding_id": r.embedding_id,
            "text": r.text,
            "rank": float(r.rank),
        }
        for r in rows
    ]
