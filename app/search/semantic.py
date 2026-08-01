from app.embeddings.embedder import embed_query
from app.embeddings.vector_store import query_similar


def semantic_search(query: str, top_k: int = 10, category_filter: str | None = None) -> list[dict]:
    embedding = embed_query(query)
    where = {"category": category_filter} if category_filter else None
    results = query_similar(embedding, top_k=top_k, where=where)

    hits = []
    for i, doc_text in enumerate(results.get("documents", [[]])[0]):
        chunk_id = results["ids"][0][i]
        metadata = results["metadatas"][0][i] or {}
        # document_id is stored in Chroma metadata at upsert time (see
        # app/embeddings/vector_store.py::upsert_chunks). Fall back to
        # parsing it off the chunk id ("{document_id}::{uuid}") only if
        # metadata is ever missing it, so downstream consumers that key
        # off document_id (app/evaluation/eval.py, app/rag/generate.py)
        # never silently fall back to a raw chunk id instead.
        document_id = metadata.get("document_id") or chunk_id.split("::", 1)[0]
        hits.append({
            "text": doc_text,
            "id": chunk_id,
            "document_id": document_id,
            "distance": results["distances"][0][i],
            "metadata": metadata,
        })
    return hits
