from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.ingestion.loaders import load_by_source_type, load_github_repo
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


def _chunk_embed_store(db: Session, document: Document, text: str) -> Document:
    """
    Shared tail end of ingestion: chunk -> embed -> store in vector DB ->
    persist DocumentChunk rows -> mark the Document pending approval.
    Used by both ingest_document() (single file) and ingest_github_repo()
    (one call per file in the repo) so the two paths can't drift apart.
    """
    # Assign document.id before it's used below. Document.id has a
    # Python-side `default=gen_uuid` callable (see app/core/models.py),
    # but SQLAlchemy only invokes that default at flush/INSERT time, not
    # at object construction. Without this db.add()+db.flush() up front,
    # document.id stays None through the entire chunk/embed/store
    # sequence below, and ChromaDB's metadata validation then rejects
    # None outright ("Expected metadata value to be a str, int, float or
    # bool, got None"). Found via manual end-to-end testing — the
    # embedder and vector_store unit tests both passed in isolation
    # because they used a hardcoded string document_id, which masked
    # this integration bug completely; only calling the real pipeline
    # function end-to-end surfaced it. flush() (not commit()) assigns
    # the primary key without ending the transaction, so a failure later
    # in this function can still be rolled back by the caller.
    db.add(document)
    db.flush()

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
