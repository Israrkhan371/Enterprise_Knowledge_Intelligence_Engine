"""
Tests for semantic_search()'s optional db-based orphan filter.

Regression coverage for a real bug found via live testing 2026-08-16:
ChromaDB has no foreign-key relationship to Postgres (unlike
document_chunks, which has a real FK to documents.id), so a document
deleted directly in Postgres (a manual TRUNCATE/reset during testing --
there's no delete endpoint in the app itself) leaves its vectors and
citable chunk text behind in ChromaDB indefinitely. A stale chunk from an
already-deleted document surfaced as a real citation in a fresh /ask
answer, which also explained a batch of citation_check flags that looked
like a verification bug but were actually correct: the answer was citing
real (but stale, orphaned) content that partially disagreed with the
current, real document.

Mocks query_similar/embed_query (no live ChromaDB/embedding model) and
uses a MagicMock db to control which document_ids "exist" in Postgres,
so these run without a live Postgres connection either.
"""
from unittest.mock import MagicMock, patch

from app.search.semantic import semantic_search


def _fake_chroma_response(ids, texts, distances, metadatas):
    return {
        "ids": [ids],
        "documents": [texts],
        "distances": [distances],
        "metadatas": [metadatas],
    }


def _mock_db_with_existing_ids(existing_ids):
    """A MagicMock db whose .query(Document.id).filter(...).all() returns
    one row per id in existing_ids, matching how the real filter reads it."""
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [(i,) for i in existing_ids]
    return db


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_without_db_does_not_filter(mock_embed, mock_query):
    """Backward compatibility: omitting db (the default) must behave
    exactly as before this fix -- no filtering, no Postgres dependency."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query.return_value = _fake_chroma_response(
        ids=["orphan-doc::abc"], texts=["stale text"], distances=[0.5],
        metadatas=[{"document_id": "orphan-doc"}],
    )

    results = semantic_search("query", top_k=3)

    assert len(results) == 1
    assert results[0]["document_id"] == "orphan-doc"


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_with_db_drops_orphaned_document_ids(mock_embed, mock_query):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query.return_value = _fake_chroma_response(
        ids=["real-doc::abc", "orphan-doc::xyz"],
        texts=["real content", "stale orphaned content"],
        distances=[0.3, 0.6],
        metadatas=[{"document_id": "real-doc"}, {"document_id": "orphan-doc"}],
    )
    db = _mock_db_with_existing_ids({"real-doc"})  # orphan-doc does NOT exist in Postgres

    results = semantic_search("query", top_k=3, db=db)

    assert len(results) == 1
    assert results[0]["document_id"] == "real-doc"


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_with_db_keeps_all_hits_when_none_are_orphaned(mock_embed, mock_query):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query.return_value = _fake_chroma_response(
        ids=["doc-1::a", "doc-2::b"],
        texts=["text one", "text two"],
        distances=[0.1, 0.2],
        metadatas=[{"document_id": "doc-1"}, {"document_id": "doc-2"}],
    )
    db = _mock_db_with_existing_ids({"doc-1", "doc-2"})

    results = semantic_search("query", top_k=3, db=db)

    assert len(results) == 2


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_with_db_returns_empty_list_when_all_orphaned(mock_embed, mock_query):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query.return_value = _fake_chroma_response(
        ids=["gone-1::a", "gone-2::b"],
        texts=["stale one", "stale two"],
        distances=[0.1, 0.2],
        metadatas=[{"document_id": "gone-1"}, {"document_id": "gone-2"}],
    )
    db = _mock_db_with_existing_ids(set())  # nothing exists

    results = semantic_search("query", top_k=3, db=db)

    assert results == []


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_with_db_skips_the_query_entirely_for_no_hits(mock_embed, mock_query):
    """No wasted Postgres round-trip when ChromaDB itself returned nothing."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_query.return_value = _fake_chroma_response(ids=[], texts=[], distances=[], metadatas=[])
    db = MagicMock()

    results = semantic_search("query", top_k=3, db=db)

    assert results == []
    db.query.assert_not_called()
