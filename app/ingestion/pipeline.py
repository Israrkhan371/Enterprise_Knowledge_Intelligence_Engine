from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.ingestion.loaders import load_by_source_type
from app.ingestion.ocr import needs_ocr, ocr_scanned_pdf
from app.ingestion.chunking import chunk_text
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import upsert_chunks
from app.graph.extract import extract_entities, extract_relationships
from app.graph.build import GraphStore


def _build_knowledge_graph(document: Document, text: str) -> None:
    """
    Full-document AI pass: NER + relation extraction, written into Neo4j.
    Runs once on the whole document (not per-chunk) because entities and
    their relationships are document-level concepts - chunking would cut
    sentences and co-occurrence context in half at chunk boundaries.
    """
    entities = extract_entities(text)
    relationships = extract_relationships(text, entities)

    graph = GraphStore()
    try:
        graph.upsert_document_node(
            document_id=document.id,
            title=document.title,
            source_type=document.source_type,
        )
        if entities:
            graph.upsert_entities(document_id=document.id, entities=entities)
        if relationships:
            graph.upsert_relationships(relationships)
    finally:
        graph.close()


def ingest_document(db: Session, document: Document, file_path: str) -> Document:
    """Runs one document through the full ingestion pipeline and persists chunks."""
    text = load_by_source_type(document.source_type, file_path)

    if document.source_type == "pdf" and needs_ocr(text):
        text = ocr_scanned_pdf(file_path)

    document.raw_text = text

    # --- Full-document AI (NER, relation extraction -> knowledge graph) ---
    # Runs on the complete raw text, before chunking. Summarization,
    # duplicate/version detection, and metadata extraction are separate
    # full-document steps slated for later tasks and are not added here.
    _build_knowledge_graph(document, text)

    # --- Chunk-level AI (embeddings -> vector store, for semantic/hybrid RAG) ---
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)
    vector_ids = upsert_chunks(
        document_id=document.id,
        chunks=chunks,
        embeddings=embeddings,
    )

    for idx, (chunk, vec_id) in enumerate(zip(chunks, vector_ids)):
        db.add(DocumentChunk(
            document_id=document.id,
            chunk_index=idx,
            text=chunk,
            embedding_id=vec_id,
        ))

    document.status = "pending"  # awaiting admin approval
    db.add(document)
    db.commit()
    db.refresh(document)
    return document
