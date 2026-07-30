from app.embeddings.embedder import embed_query
from app.embeddings.vector_store import query_similar


def semantic_search(query: str, top_k: int = 10, category_filter: str | None = None) -> list[dict]:
    embedding = embed_query(query)
    where = {"category": category_filter} if category_filter else None
    results = query_similar(embedding, top_k=top_k, where=where)

    hits = []
    for i, doc_text in enumerate(results.get("documents", [[]])[0]):
        metadata = results["metadatas"][0][i]
        hits.append({
            "text": doc_text,
            "id": results["ids"][0][i],
            # Also surfaced at the top level (not just nested in metadata)
            # because app/rag/generate.py's build_context_block() and
            # hybrid_search()'s RRF fusion both key off a top-level
            # "document_id" for keyword_search() hits, and previously fell
            # back to the vector-store id ("id") for semantic-only hits
            # when it was missing here — a real citation-mapping bug found
            # 2026-07-30 while auditing hybrid search's fusion key.
            "document_id": metadata.get("document_id"),
            "distance": results["distances"][0][i],
            "metadata": metadata,
        })
    return hits
