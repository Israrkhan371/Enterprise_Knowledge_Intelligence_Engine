from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.ingestion.loaders import load_by_source_type
from app.ingestion.ocr import needs_ocr, ocr_scanned_pdf
from app.ingestion.chunking import chunk_text
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import upsert_chunks


def ingest_document(db: Session, document: Document, file_path: str) -> Document:
    """Runs one document through the full ingestion pipeline and persists chunks."""
    text = load_by_source_type(document.source_type, file_path)

    if document.source_type == "pdf" and needs_ocr(text):
        text = ocr_scanned_pdf(file_path)

    document.raw_text = text
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
