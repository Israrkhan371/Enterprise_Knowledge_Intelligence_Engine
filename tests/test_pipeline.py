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

import pytest

from app.core.models import Document
from app.ingestion.pipeline import _chunk_embed_store, _populate_graph


@pytest.fixture(autouse=True)
def _no_real_graph_store():
    """
    Keeps the pre-existing tests in this file (which don't care about graph
    behavior) from ever touching a real Neo4j driver now that
    _chunk_embed_store() calls _populate_graph() on every run. Tests that
    specifically exercise _populate_graph() override this with their own
    explicit patch of GraphStore.
    """
    with patch("app.ingestion.pipeline.GraphStore") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


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


# ---------------------------------------------------------------------------
# Graph wiring: entity extraction -> GraphStore, called from
# _chunk_embed_store() so Neo4j actually gets populated during ingestion
# (previously extract.py/build.py existed but nothing in the ingestion path
# called them - see Friday checkpoint notes).
# ---------------------------------------------------------------------------

@patch("app.ingestion.pipeline.extract_relationships")
@patch("app.ingestion.pipeline.extract_entities")
@patch("app.ingestion.pipeline.GraphStore")
def test_populate_graph_upserts_document_entities_and_relationships(
    mock_graph_store_cls, mock_extract_entities, mock_extract_relationships,
):
    mock_store = MagicMock()
    mock_graph_store_cls.return_value = mock_store
    mock_extract_entities.return_value = [{"text": "Python", "label": "TECH"}]
    mock_extract_relationships.return_value = [
        {"source": "Python", "target": "FastAPI", "relation": "co_occurs_with", "context": "..."}
    ]

    document = Document(title="Doc", source_type="markdown", source_uri="test")
    document.id = "doc-1"

    _populate_graph(document, "Python and FastAPI are used together.")

    mock_store.upsert_document_node.assert_called_once_with(
        document_id="doc-1", title="Doc", source_type="markdown",
    )
    mock_extract_entities.assert_called_once_with("Python and FastAPI are used together.")
    mock_store.upsert_entities.assert_called_once_with(
        "doc-1", [{"text": "Python", "label": "TECH"}],
    )
    mock_extract_relationships.assert_called_once_with(
        "Python and FastAPI are used together.", [{"text": "Python", "label": "TECH"}],
    )
    mock_store.upsert_relationships.assert_called_once_with(
        [{"source": "Python", "target": "FastAPI", "relation": "co_occurs_with", "context": "..."}]
    )
    mock_store.close.assert_called_once()


@patch("app.ingestion.pipeline.extract_relationships")
@patch("app.ingestion.pipeline.extract_entities")
@patch("app.ingestion.pipeline.GraphStore")
def test_populate_graph_skips_entity_and_relationship_upserts_when_no_entities(
    mock_graph_store_cls, mock_extract_entities, mock_extract_relationships,
):
    """A document with no recognizable entities still gets its Document
    node created (so it's queryable in the graph), but shouldn't call
    upsert_entities/extract_relationships with an empty list."""
    mock_store = MagicMock()
    mock_graph_store_cls.return_value = mock_store
    mock_extract_entities.return_value = []

    document = Document(title="Doc", source_type="markdown", source_uri="test")
    document.id = "doc-2"

    _populate_graph(document, "no named entities here")

    mock_store.upsert_document_node.assert_called_once()
    mock_store.upsert_entities.assert_not_called()
    mock_extract_relationships.assert_not_called()
    mock_store.close.assert_called_once()


@patch("app.ingestion.pipeline.extract_entities")
@patch("app.ingestion.pipeline.GraphStore")
def test_populate_graph_swallows_exceptions_and_still_closes_store(
    mock_graph_store_cls, mock_extract_entities,
):
    """A Neo4j failure (e.g. the container still restarting) must not
    propagate out of _populate_graph and fail an otherwise-successful
    ingestion - and the driver must still be closed."""
    mock_store = MagicMock()
    mock_store.upsert_document_node.side_effect = RuntimeError("neo4j unavailable")
    mock_graph_store_cls.return_value = mock_store

    document = Document(title="Doc", source_type="markdown", source_uri="test")
    document.id = "doc-3"

    _populate_graph(document, "some text")  # must not raise

    mock_extract_entities.assert_not_called()
    mock_store.close.assert_called_once()


@patch("app.ingestion.pipeline._populate_graph")
@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_chunk_embed_store_calls_populate_graph_with_full_text(
    mock_embed_texts, mock_upsert_chunks, mock_populate_graph,
):
    """_populate_graph() must run on the full raw document text, not the
    embedding chunks - chunk boundaries would fragment sentences and
    undercount/duplicate entities and relationships."""
    mock_embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_upsert_chunks.return_value = ["vec-id-1", "vec-id-2"]

    db = _make_mock_db(assign_id_on_flush="fake-doc-id-789")
    document = Document(title="Doc", source_type="markdown", source_uri="test")
    full_text = "word " * 2000  # long enough to span multiple chunks

    result = _chunk_embed_store(db, document, full_text)

    mock_populate_graph.assert_called_once_with(result, full_text)
    # sanity: this is genuinely more than one chunk, so the test would be
    # meaningless (couldn't distinguish full text from chunk text) otherwise
    from app.ingestion.chunking import chunk_text
    assert len(chunk_text(full_text)) > 1


@patch("app.ingestion.pipeline._populate_graph")
@patch("app.ingestion.pipeline.upsert_chunks")
@patch("app.ingestion.pipeline.embed_texts")
def test_chunk_embed_store_calls_populate_graph_after_commit(
    mock_embed_texts, mock_upsert_chunks, mock_populate_graph,
):
    """Graph population must happen after the Postgres/ChromaDB commit,
    so a Neo4j-side failure can never roll back a successful ingestion."""
    mock_embed_texts.return_value = [[0.1, 0.2]]
    mock_upsert_chunks.return_value = ["vec-id-1"]

    db = _make_mock_db()
    document = Document(title="Doc", source_type="markdown", source_uri="test")

    def _assert_committed_already(*a, **k):
        db.commit.assert_called_once()
    mock_populate_graph.side_effect = _assert_committed_already

    _chunk_embed_store(db, document, "word " * 50)

    mock_populate_graph.assert_called_once()
