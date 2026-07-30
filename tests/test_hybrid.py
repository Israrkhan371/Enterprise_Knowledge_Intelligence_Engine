"""
Tests for app/search/hybrid.py.

The main thing under test is a real bug found 2026-07-30: reciprocal_rank_
fusion() fused on "id", but semantic_search()'s "id" (a ChromaDB vector id
like "doc-1::abc") and keyword_search()'s "id" (a Postgres document_chunks
primary key) are two disjoint id spaces for the *same* chunk — so a chunk
ranked highly by both semantic and keyword search could never actually get
boosted by fusion, silently defeating the point of RRF. hybrid_search() now
fuses on embedding_id/id instead, which both sides genuinely share.

semantic_search() and keyword_search() are mocked here (same approach as
test_semantic.py/test_keyword.py) so these tests don't need a live
ChromaDB, embedding model, or Postgres connection.
"""
from unittest.mock import patch

from app.search.hybrid import hybrid_search, reciprocal_rank_fusion


# --- reciprocal_rank_fusion() in isolation --------------------------------

def test_rrf_boosts_item_ranked_in_both_lists():
    list_a = [{"id": "x", "text": "shared"}, {"id": "y", "text": "only in a"}]
    list_b = [{"id": "x", "text": "shared"}, {"id": "z", "text": "only in b"}]

    fused = reciprocal_rank_fusion([list_a, list_b])

    # "x" appears (rank 0) in both lists, so its fused score is the sum of
    # both lists' RRF contributions, and it should outrank items that only
    # appear once.
    assert fused[0]["id"] == "x"
    assert fused[0]["fused_score"] > fused[1]["fused_score"]
    assert fused[0]["fused_score"] > fused[2]["fused_score"]


def test_rrf_merges_fields_from_both_lists_for_a_shared_item():
    """
    Regression test for the payload-overwrite issue: before the fix, a
    chunk present in both lists lost whichever fields were unique to the
    first list once the second list's dict overwrote it wholesale (e.g.
    losing "distance"/"metadata" from a semantic hit under a keyword hit's
    identically-keyed dict).
    """
    list_a = [{"id": "x", "distance": 0.1, "metadata": {"category": "eng"}}]
    list_b = [{"id": "x", "rank": 0.9, "text": "actual chunk text"}]

    fused = reciprocal_rank_fusion([list_a, list_b])

    assert fused[0]["distance"] == 0.1
    assert fused[0]["metadata"] == {"category": "eng"}
    assert fused[0]["rank"] == 0.9
    assert fused[0]["text"] == "actual chunk text"


def test_rrf_handles_disjoint_lists():
    list_a = [{"id": "a1"}]
    list_b = [{"id": "b1"}]

    fused = reciprocal_rank_fusion([list_a, list_b])

    assert {item["id"] for item in fused} == {"a1", "b1"}


# --- hybrid_search()'s fusion-key normalization ---------------------------

@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_fuses_same_chunk_found_by_both_methods(mock_semantic, mock_keyword):
    """
    The core regression test: a chunk found by both semantic search (keyed
    by its ChromaDB vector id) and keyword search (keyed by its Postgres
    row id, but carrying the *same* vector id as embedding_id) must be
    recognized as the same chunk and fused into a single boosted result,
    not returned as two separate entries.
    """
    mock_semantic.return_value = [
        {"id": "doc-1::vec-abc", "document_id": "doc-1", "text": "Pinecone is a vector database.", "distance": 0.1},
    ]
    mock_keyword.return_value = [
        {"id": "chunk-99", "document_id": "doc-1", "embedding_id": "doc-1::vec-abc", "text": "Pinecone is a vector database.", "rank": 0.8},
    ]

    db = object()  # unused by the mocks, but keyword_search's signature expects a db arg
    results = hybrid_search(db, "Pinecone", top_k=5)

    assert len(results) == 1
    assert results[0]["distance"] == 0.1  # kept from the semantic hit
    assert results[0]["rank"] == 0.8       # kept from the keyword hit
    assert "_fusion_key" not in results[0]  # internal key must not leak out


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_keeps_chunks_found_by_only_one_method_separate(mock_semantic, mock_keyword):
    mock_semantic.return_value = [{"id": "vec-only", "document_id": "doc-1", "text": "semantic-only chunk"}]
    mock_keyword.return_value = [{"id": "chunk-1", "document_id": "doc-2", "embedding_id": None, "text": "keyword-only chunk"}]

    db = object()
    results = hybrid_search(db, "query", top_k=5)

    assert len(results) == 2


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_does_not_merge_two_different_unembedded_keyword_chunks(mock_semantic, mock_keyword):
    """
    Two different chunks that both happen to have embedding_id=None (never
    embedded yet) must not collide onto the same fusion key and get merged
    into one result — each needs its own unique fallback key.
    """
    mock_semantic.return_value = []
    mock_keyword.return_value = [
        {"id": "chunk-1", "document_id": "doc-1", "embedding_id": None, "text": "first"},
        {"id": "chunk-2", "document_id": "doc-2", "embedding_id": None, "text": "second"},
    ]

    db = object()
    results = hybrid_search(db, "query", top_k=5)

    assert len(results) == 2
    assert {r["id"] for r in results} == {"chunk-1", "chunk-2"}


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_passes_category_filter_to_semantic_search(mock_semantic, mock_keyword):
    mock_semantic.return_value = []
    mock_keyword.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=5, category_filter="engineering-docs")

    _, kwargs = mock_semantic.call_args
    assert kwargs["category_filter"] == "engineering-docs"


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_respects_top_k(mock_semantic, mock_keyword):
    mock_semantic.return_value = [{"id": f"vec-{i}", "document_id": "d", "text": "t"} for i in range(5)]
    mock_keyword.return_value = []

    db = object()
    results = hybrid_search(db, "query", top_k=2)

    assert len(results) == 2
