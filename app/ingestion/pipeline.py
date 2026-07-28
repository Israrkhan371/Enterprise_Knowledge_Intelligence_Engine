import logging

from sqlalchemy.orm import Session

from app.core.models import Document, DocumentChunk
from app.ingestion.loaders import load_by_source_type, load_github_repo
from app.ingestion.ocr import needs_ocr, ocr_scanned_pdf
from app.ingestion.chunking import chunk_text
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import upsert_chunks
from app.graph.build import GraphStore
from app.graph.extract import extract_entities, extract_relationships

logger = logging.getLogger(__name__)


def _populate_graph(document: Document, text: str) -> None:
    """
    Runs entity/relationship extraction over the *full* document text (not
    the embedding chunks - see hybrid architecture note below) and upserts
    the result into Neo4j.

    Hybrid architecture: chunk-level text feeds embeddings/RAG, while
    full-document text feeds NER/graph here. Chunking is tuned for
    retrieval context (800 words, 120 overlap), which would fragment
    entities and sentences across boundaries and undercount/duplicate
    relationships if reused for NER - so this runs on the whole document
    in one pass instead of per-chunk.

    Called after the Postgres/ChromaDB writes are committed, and never
    raises: a Neo4j hiccup (or the container still being up mid-restart)
    shouldn't fail an otherwise-successful ingestion. Failures are logged
    so they're visible without blocking the document from reaching
    "pending".
    """
    store = GraphStore()
    try:
        store.upsert_document_node(
            document_id=document.id,
            title=document.title,
            source_type=document.source_type,
        )
        entities = extract_entities(text)
        if not entities:
            return
        store.upsert_entities(document.id, entities)
        relationships = extract_relationships(text, entities)
        if relationships:
            store.upsert_relationships(relationships)
    except Exception:
        logger.exception(
            "Graph population failed for document_id=%s (title=%r); "
            "document ingestion still succeeded.",
            document.id, document.title,
        )
    finally:
        store.close()


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
    # bool, got None"). flush() (not commit()) assigns the primary key
    # without ending the transaction, so a failure later in this function
    # can still be rolled back by the caller.
    db.add(document)
    db.flush()

    document.raw_text = text

    # --- Chunk-level AI (embeddings -> vector store, for semantic/hybrid RAG) ---
    chunks = chunk_text(text)

    embeddings = embed_texts(chunks)

    # Resolve the category *name* (not the raw category_id UUID) for
    # ChromaDB metadata, since semantic_search()'s category_filter takes a
    # human-readable string (e.g. "test-category"), not a UUID. document
    # was already flushed above, so document.category triggers a normal
    # SQLAlchemy relationship lookup within this session if category_id
    # is set; stays None for uncategorized documents.
    category_name = document.category.name if document.category_id else None

    vector_ids = upsert_chunks(
        document_id=document.id,
        chunks=chunks,
        embeddings=embeddings,
        category=category_name,
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

    # Graph population runs after the Postgres/ChromaDB commit above, on
    # the full document text (see _populate_graph docstring for why it
    # doesn't reuse `chunks`). It never raises, so a Neo4j failure can't
    # undo the ingestion that already succeeded.
    _populate_graph(document, text)

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