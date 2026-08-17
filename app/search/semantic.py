from sqlalchemy.orm import Session

from app.embeddings.embedder import embed_query
from app.embeddings.vector_store import query_similar


def semantic_search(
    query: str,
    top_k: int = 10,
    category_filter: str | None = None,
    db: Session | None = None,
) -> list[dict]:
    embedding = embed_query(query)
    where = {"category": category_filter} if category_filter else None
    results = query_similar(embedding, top_k=top_k, where=where)

    hits = []
    for i, doc_text in enumerate(results.get("documents", [[]])[0]):
        chunk_id = results["ids"][0][i]
        metadata = results["metadatas"][0][i] or {}
        # document_id is stored in Chroma metadata at upsert time (see
        # app/embeddings/vector_store.py::upsert_chunks). Also surfaced at
        # the top level (not just nested in metadata) because
        # app/rag/generate.py's build_context_block() and hybrid_search()'s
        # RRF fusion both key off a top-level "document_id" for
        # keyword_search() hits, and previously fell back to the
        # vector-store id ("id") for semantic-only hits when it was missing
        # here — a real citation-mapping bug found 2026-07-30 while
        # auditing hybrid search's fusion key. Falls back to parsing it off
        # the chunk id ("{document_id}::{uuid}") only if metadata is ever
        # missing it, so downstream consumers never silently regress back
        # to a raw chunk id.
        document_id = metadata.get("document_id") or chunk_id.split("::", 1)[0]
        hits.append({
            "text": doc_text,
            "id": chunk_id,
            "document_id": document_id,
            "distance": results["distances"][0][i],
            "metadata": metadata,
        })

    if db is not None and hits:
        # ChromaDB has no foreign-key relationship to Postgres — unlike
        # document_chunks (which has a real FK to documents.id, so
        # keyword_search() can never return an orphaned document_id),
        # ChromaDB vectors survive independently of whatever happens in
        # Postgres. A document deleted directly in Postgres (a manual
        # TRUNCATE/DELETE during testing/reset, not anything the app itself
        # does today — there's no delete endpoint) leaves its vectors and
        # citable chunk text behind indefinitely, so a later /ask could
        # cite a document that no longer exists. Confirmed live 2026-08-16:
        # a stale ChromaDB chunk from an earlier session's already-deleted
        # "Deployment SOP" surfaced as a real citation in a fresh answer.
        # Filtering here, right before returning, is the general fix
        # regardless of what causes Postgres/ChromaDB to drift apart in the
        # future (this isn't reachable via the app today, but a delete
        # endpoint or a bad migration could reintroduce the same drift
        # later) — cheaper than trying to enumerate and prevent every way
        # the two stores could get out of sync.
        from app.core.models import Document

        candidate_ids = {h["document_id"] for h in hits if h.get("document_id")}
        existing_ids = {
            row[0] for row in db.query(Document.id).filter(Document.id.in_(candidate_ids)).all()
        }
        hits = [h for h in hits if h.get("document_id") in existing_ids]

    return hits