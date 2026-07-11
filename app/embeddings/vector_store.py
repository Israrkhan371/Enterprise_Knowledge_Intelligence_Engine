import uuid
from functools import lru_cache

import chromadb

from app.core.config import settings


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return client.get_or_create_collection(name=settings.chroma_collection)


def upsert_chunks(document_id: str, chunks: list[str], embeddings: list[list[float]]) -> list[str]:
    if not chunks:
        return []
    collection = get_collection()
    ids = [f"{document_id}::{uuid.uuid4()}" for _ in chunks]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"document_id": document_id} for _ in chunks],
    )
    return ids


def query_similar(embedding: list[float], top_k: int = 10, where: dict | None = None):
    collection = get_collection()
    return collection.query(query_embeddings=[embedding], n_results=top_k, where=where)
