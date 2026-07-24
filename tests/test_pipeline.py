"""
Tests for app/ingestion/pipeline.py.

Regression test for a real bug found via manual end-to-end testing (not
caught by the embedder/vector_store unit tests, which both used a
hardcoded string document_id and so never exercised the real ID-assignment
timing): _chunk_embed_store() used document.id before the Document was
ever added+flushed to the database session. Document.id has a Python-side
`default=gen_uuid` callable (see app/core/models.py), but SQLAlchemy only
invokes that default at flush/INSERT time — so document.id was None for
the entire chunk/embed/store sequence, and ChromaDB's metadata validation
rejected None outright when it was passed to upsert_chunks().

These tests use mocks rather than a live Postgres/ChromaDB connection,
so they can run without any external services — but they exercise the
*exact* ordering bug directly: the mocked db.flush() assigns document.id
the same way a real flush would, and the test fails if upsert_chunks()
or DocumentChunk() ever receive None where a real ID is expected.
"""
from unittest.mock import MagicMock, patch

from app.core.models import Document
from app.ingestion.pipeline import _chunk_embed_store


def _make_mock_db(assign_id_on_flush: str = "fake-doc-id-123"):
    """
    A mock SQLAlchemy Session whose flush() simulates real ORM behavior:
    assigns document.id (via the Document's own gen_uuid default) only
    once the object has been added and flushed — never before.
    """
    db = MagicMock()
    added_documents = []

    def fake_add(obj):
        if isinstance(obj, Document):
            added_documents.append(obj)

    def fake_flush():
        for doc in added_documents:
            if doc.id is None:
                doc.id = assign_id_on_flush

    db.add.side_effect = fake_add
    db.flush.side_effect = fake_flush
    return db


@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_chunk_embed_store_never_passes_none_document_id(mock_embed_texts, mock_upsert_chunks):
    """
    The core regression test: document.id must be a real value (not None)
    by the time it's passed to upsert_chunks() — this is exactly the
    condition that raised chromadb.api.types.ValueError in manual testing
    before the fix (db.add(document) + db.flush() added at the top of
    _chunk_embed_store).
    """
    mock_embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_upsert_chunks.return_value = ["vec-id-1", "vec-id-2"]

    db = _make_mock_db(assign_id_on_flush="fake-doc-id-123")
    document = Document(title="Test", source_type="markdown", source_uri="test")

    assert document.id is None  # sanity check: not yet flushed

    _chunk_embed_store(db, document, "word " * 300)

    # The real assertion: whatever document_id upsert_chunks was called
    # with must be the real assigned ID, never None.
    _, kwargs = mock_upsert_chunks.call_args
    assert kwargs["document_id"] is not None
    assert kwargs["document_id"] == "fake-doc-id-123"


@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_chunk_embed_store_flushes_before_using_document_id(mock_embed_texts, mock_upsert_chunks):
    """
    More direct check on ordering: db.flush() must be called before any
    DocumentChunk gets added to the session (which happens after
    upsert_chunks returns) — asserting call order, not just the end
    result, so this test would fail even if some other change
    accidentally reintroduced a race between assignment and use.

    Note: db.add(document) legitimately happens immediately before
    db.flush() (you can't flush an object that was never added), so
    this checks flush comes before the *chunk* adds, not that flush is
    literally the first call overall.
    """
    from app.core.models import DocumentChunk

    mock_embed_texts.return_value = [[0.1, 0.2]]
    mock_upsert_chunks.return_value = ["vec-id-1"]

    db = _make_mock_db()
    document = Document(title="Test", source_type="markdown", source_uri="test")

    _chunk_embed_store(db, document, "word " * 50)

    flush_call_index = None
    first_chunk_add_index = None
    for i, call in enumerate(db.method_calls):
        name, args, kwargs = call
        if name == "flush" and flush_call_index is None:
            flush_call_index = i
        if name == "add" and args and isinstance(args[0], DocumentChunk) and first_chunk_add_index is None:
            first_chunk_add_index = i

    assert flush_call_index is not None, "db.flush() was never called"
    assert first_chunk_add_index is not None, "no DocumentChunk was ever added"
    assert flush_call_index < first_chunk_add_index, (
        "db.flush() must happen before any DocumentChunk is added, "
        "confirming document.id was assigned before it's used"
    )
    _, kwargs = mock_upsert_chunks.call_args
    assert kwargs["document_id"] is not None


@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_document_chunk_rows_use_real_document_id(mock_embed_texts, mock_upsert_chunks):
    """
    The same None-id bug would also have affected DocumentChunk rows
    (document_id=document.id in the loop below upsert_chunks) — confirm
    those get the real ID too, not just the vector store call.
    """
    mock_embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_upsert_chunks.return_value = ["vec-id-1", "vec-id-2"]

    db = _make_mock_db(assign_id_on_flush="fake-doc-id-456")
    document = Document(title="Test", source_type="markdown", source_uri="test")

    _chunk_embed_store(db, document, "word " * 300)

    from app.core.models import DocumentChunk
    chunk_adds = [
        call.args[0] for call in db.add.call_args_list
        if call.args and isinstance(call.args[0], DocumentChunk)
    ]
    assert len(chunk_adds) > 0
    for chunk in chunk_adds:
        assert chunk.document_id == "fake-doc-id-456"


@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_chunk_embed_store_sets_status_pending_and_commits(mock_embed_texts, mock_upsert_chunks):
    """Confirms the rest of the function still behaves correctly after the fix."""
    mock_embed_texts.return_value = [[0.1, 0.2]]
    mock_upsert_chunks.return_value = ["vec-id-1"]

    db = _make_mock_db()
    document = Document(title="Test", source_type="markdown", source_uri="test")

    result = _chunk_embed_store(db, document, "word " * 50)

    assert result.status == "pending"
    assert result.raw_text == "word " * 50
    db.commit.assert_called_once()
