"""
Tests for app/embeddings/vector_store.py.

Regression test for a real bug found via manual end-to-end testing: any
call to semantic_search(..., category_filter=...) always returned []
regardless of what value was passed. The root cause was here, not in
semantic_search() itself: upsert_chunks() only ever stored
{"document_id": ...} as ChromaDB metadata, so a "category" field never
existed on a single stored chunk, and ChromaDB's `where` clause always
filtered against a field with zero possible matches.

These tests mock the ChromaDB collection so they run without a live
ChromaDB connection, but exercise the exact metadata shape that gets
passed to collection.add() — the same shape semantic_search()'s `where`
clause is matched against.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.embeddings.vector_store import upsert_chunks


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    with patch("app.embeddings.vector_store.get_collection", return_value=collection):
        yield collection


def test_upsert_chunks_stores_category_when_provided(mock_collection):
    upsert_chunks(
        document_id="doc-1",
        chunks=["chunk one", "chunk two"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        category="engineering-docs",
    )

    _, kwargs = mock_collection.add.call_args
    metadatas = kwargs["metadatas"]

    assert len(metadatas) == 2
    for metadata in metadatas:
        assert metadata["document_id"] == "doc-1"
        assert metadata["category"] == "engineering-docs"


def test_upsert_chunks_omits_category_key_when_none(mock_collection):
    """
    This is the exact bug: without this, category_filter queries in
    semantic_search() can never match anything, for any document,
    regardless of what category value is passed to the filter.
    """
    upsert_chunks(
        document_id="doc-2",
        chunks=["uncategorized chunk"],
        embeddings=[[0.5, 0.6]],
        category=None,
    )

    _, kwargs = mock_collection.add.call_args
    metadatas = kwargs["metadatas"]

    assert len(metadatas) == 1
    assert metadatas[0] == {"document_id": "doc-2"}
    assert "category" not in metadatas[0]


def test_upsert_chunks_defaults_to_no_category_when_omitted(mock_collection):
    """category is an optional kwarg — callers that don't pass it at all
    (not just callers who explicitly pass None) must get the same safe
    behavior, since this is the exact call shape the bug had before the
    fix (upsert_chunks() didn't accept a category argument at all)."""
    upsert_chunks(
        document_id="doc-3",
        chunks=["chunk"],
        embeddings=[[0.7, 0.8]],
    )

    _, kwargs = mock_collection.add.call_args
    metadatas = kwargs["metadatas"]

    assert metadatas[0] == {"document_id": "doc-3"}


def test_upsert_chunks_returns_empty_list_for_no_chunks(mock_collection):
    result = upsert_chunks(document_id="doc-4", chunks=[], embeddings=[])

    assert result == []
    mock_collection.add.assert_not_called()


def test_upsert_chunks_generates_unique_ids_per_chunk(mock_collection):
    ids = upsert_chunks(
        document_id="doc-5",
        chunks=["a", "b", "c"],
        embeddings=[[0.1], [0.2], [0.3]],
    )

    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert all(i.startswith("doc-5::") for i in ids)