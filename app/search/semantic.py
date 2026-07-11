from app.embeddings.embedder import embed_query
from app.embeddings.vector_store import query_similar


def semantic_search(query: str, top_k: int = 10, category_filter: str | None = None) -> list[dict]:
    embedding = embed_query(query)
    where = {"category": category_filter} if category_filter else None
    results = query_similar(embedding, top_k=top_k, where=where)

    hits = []
    for i, doc_text in enumerate(results.get("documents", [[]])[0]):
        hits.append({
            "text": doc_text,
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits
