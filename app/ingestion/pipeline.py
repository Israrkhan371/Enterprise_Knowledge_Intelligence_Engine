from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.ingestion.loaders import load_by_source_type, load_github_repo
from app.ingestion.ocr import needs_ocr, ocr_scanned_pdf
from app.ingestion.chunking import chunk_text
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import upsert_chunks


def _chunk_embed_store(db: Session, document: Document, text: str) -> Document:
    """
    Shared tail end of ingestion: chunk -> embed -> store in vector DB ->
    persist DocumentChunk rows -> mark the Document pending approval.
    Used by both ingest_document() (single file) and ingest_github_repo()
    (one call per file in the repo) so the two paths can't drift apart.
    """
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


def ingest_document(db: Session, document: Document, file_path: str) -> Document:
    """Runs one document through the full ingestion pipeline and persists chunks."""
    text = load_by_source_type(document.source_type, file_path)

    if document.source_type == "pdf" and needs_ocr(text):
        text = ocr_scanned_pdf(file_path)

    return _chunk_embed_store(db, document, text)


def ingest_github_repo(
    db: Session,
    repo_url: str,
    category_id: str | None = None,
    uploaded_by: str | None = None,
    github_token: str | None = None,
) -> list[Document]:
    """
    Fans a GitHub repo out into one Document row per file. load_github_repo()
    returns list[dict], not str, so it can't reuse ingest_document() directly
    — each file gets its own Document + chunk/embed/store pass instead.
    """
    files = load_github_repo(repo_url, github_token=github_token)

    documents = []
    for file in files:
        document = Document(
            title=file["path"],
            source_type="github",
            source_uri=f"{repo_url.rstrip('/')}/blob/main/{file['path']}",
            category_id=category_id,
            uploaded_by=uploaded_by,
        )
        documents.append(_chunk_embed_store(db, document, file["text"]))

    return documents
