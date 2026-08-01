"""
Tests for app/search/semantic.py.

semantic_search() itself was never buggy — it correctly builds a `where`
clause from category_filter and passes it through to query_similar(). The
bug (confirmed via manual testing) was upstream: upsert_chunks() never
stored a "category" key in ChromaDB metadata at all, so this `where`
clause always filtered against a field with zero possible matches,
regardless of what value was passed. See test_vector_store.py for the
regression test on the actual fix.

These tests mock embed_query() and query_similar() so they run without a
live embedding model or ChromaDB connection, and lock in semantic_search()'s
own behavior: what `where` clause it builds, and how it shapes results.
"""
from unittest.mock import patch

from app.search.semantic import semantic_search


def _fake_chroma_response(ids, texts, distances, metadatas):
    return {
        "ids": [ids],
        "documents": [texts],
        "distances": [distances],
        "metadatas": [metadatas],
    }


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_builds_no_where_clause_without_category_filter(
    mock_embed_query, mock_query_similar
):
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_query_similar.return_value = _fake_chroma_response(
        ids=["id-1"], texts=["Cats are mammals."], distances=[0.5],
        metadatas=[{"document_id": "doc-1"}],
    )

    semantic_search("What are cats?", top_k=3)

    _, kwargs = mock_query_similar.call_args
    assert kwargs["where"] is None


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_builds_category_where_clause_when_filter_given(
    mock_embed_query, mock_query_similar
):
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_query_similar.return_value = _fake_chroma_response(
        ids=[], texts=[], distances=[], metadatas=[],
    )

    semantic_search("What are cats?", top_k=3, category_filter="test-category")

    _, kwargs = mock_query_similar.call_args
    assert kwargs["where"] == {"category": "test-category"}


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_returns_empty_list_for_no_matches(
    mock_embed_query, mock_query_similar
):
    """
    This is the exact shape of the bug before the fix: a category filter
    with no matching chunks in ChromaDB returns a well-formed empty
    result, not an error — which is why the bug was silent rather than
    loud. semantic_search() itself handles this correctly; the fix
    ensures there's actually something for it to match once a real
    category is stored.
    """
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_query_similar.return_value = _fake_chroma_response(
        ids=[], texts=[], distances=[], metadatas=[],
    )

    results = semantic_search("What are cats?", top_k=3, category_filter="test-category")

    assert results == []


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_shapes_results_correctly(mock_embed_query, mock_query_similar):
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_query_similar.return_value = _fake_chroma_response(
        ids=["doc-1::abc"],
        texts=["Cats are mammals."],
        distances=[0.42],
        metadatas=[{"document_id": "doc-1", "category": "test-category"}],
    )

    results = semantic_search("What are cats?", top_k=3, category_filter="test-category")

    assert len(results) == 1
    assert results[0]["text"] == "Cats are mammals."
    assert results[0]["id"] == "doc-1::abc"
    assert results[0]["distance"] == 0.42
    assert results[0]["metadata"]["category"] == "test-category"


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query")
def test_semantic_search_passes_top_k_through(mock_embed_query, mock_query_similar):
    mock_embed_query.return_value = [0.1, 0.2, 0.3]
    mock_query_similar.return_value = _fake_chroma_response(
        ids=[], texts=[], distances=[], metadatas=[],
    )

    semantic_search("query", top_k=7)

    _, kwargs = mock_query_similar.call_args
    assert kwargs["top_k"] == 7