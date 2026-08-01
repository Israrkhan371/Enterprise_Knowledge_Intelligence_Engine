import uuid
from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return client.get_or_create_collection(name=settings.chroma_collection)


def upsert_chunks(
    document_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    category: str | None = None,
) -> list[str]:
    if not chunks:
        return []
    collection = get_collection()
    ids = [f"{document_id}::{uuid.uuid4()}" for _ in chunks]

    # ChromaDB metadata values must be str/int/float/bool — None is
    # rejected outright ("Expected metadata value to be a str, int,
    # float or bool, got None"). Omit the "category" key entirely for
    # uncategorized documents rather than storing it as None/empty
    # string, so category_filter queries in semantic_search() only ever
    # match chunks that genuinely have a category, with no ambiguity
    # between "uncategorized" and "categorized as empty string".
    base_metadata = {"document_id": document_id}
    if category:
        base_metadata["category"] = category

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[dict(base_metadata) for _ in chunks],
    )
    return ids


def query_similar(embedding: list[float], top_k: int = 10, where: dict | None = None):
    collection = get_collection()
    return collection.query(query_embeddings=[embedding], n_results=top_k, where=where)