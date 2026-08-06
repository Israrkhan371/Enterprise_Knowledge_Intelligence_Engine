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

The fusion-focused tests below all pass use_reranker=False: hybrid_search()
now reranks its fused pool with a cross-encoder by default (see
app/search/rerank.py and the reranker-specific tests at the bottom of this
file), and that model isn't mocked here — these tests are about RRF fusion
correctness, not reranking, so the reranker is switched off to keep them
fast and network-free.
"""
from unittest.mock import patch

from app.core.config import settings
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
    results = hybrid_search(db, "Pinecone", top_k=5, use_reranker=False)

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
    results = hybrid_search(db, "query", top_k=5, use_reranker=False)

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
    results = hybrid_search(db, "query", top_k=5, use_reranker=False)

    assert len(results) == 2
    assert {r["id"] for r in results} == {"chunk-1", "chunk-2"}


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_passes_category_filter_to_semantic_search(mock_semantic, mock_keyword):
    mock_semantic.return_value = []
    mock_keyword.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=5, category_filter="engineering-docs", use_reranker=False)

    _, kwargs = mock_semantic.call_args
    assert kwargs["category_filter"] == "engineering-docs"


@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_respects_top_k(mock_semantic, mock_keyword):
    mock_semantic.return_value = [{"id": f"vec-{i}", "document_id": "d", "text": "t"} for i in range(5)]
    mock_keyword.return_value = []

    db = object()
    results = hybrid_search(db, "query", top_k=2, use_reranker=False)

    assert len(results) == 2


# --- hybrid_search()'s cross-encoder reranking stage ----------------------

@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_reranks_by_default(mock_semantic, mock_keyword, mock_rerank):
    """
    use_reranker defaults to True, so the fused list should be handed to
    the cross-encoder rather than returned as raw RRF output.
    """
    mock_semantic.return_value = [{"id": "vec-1", "document_id": "doc-1", "text": "chunk one"}]
    mock_keyword.return_value = []
    mock_rerank.return_value = [{"id": "vec-1", "document_id": "doc-1", "text": "chunk one", "rerank_score": 4.2}]

    db = object()
    results = hybrid_search(db, "query", top_k=5)

    assert mock_rerank.called
    assert results == [{"id": "vec-1", "document_id": "doc-1", "text": "chunk one", "rerank_score": 4.2}]


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_use_reranker_false_skips_cross_encoder(mock_semantic, mock_keyword, mock_rerank):
    mock_semantic.return_value = [{"id": "vec-1", "document_id": "doc-1", "text": "chunk one"}]
    mock_keyword.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=5, use_reranker=False)

    mock_rerank.assert_not_called()


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_passes_query_and_top_k_to_reranker(mock_semantic, mock_keyword, mock_rerank):
    mock_semantic.return_value = [{"id": "vec-1", "document_id": "doc-1", "text": "chunk one"}]
    mock_keyword.return_value = []
    mock_rerank.return_value = []

    db = object()
    hybrid_search(db, "some query", top_k=3)

    args, kwargs = mock_rerank.call_args
    assert args[0] == "some query"
    assert kwargs["top_k"] == 3


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_widens_candidate_pool_when_reranking(mock_semantic, mock_keyword, mock_rerank):
    """
    With reranking on, each leg should be asked for settings.rerank_pool_size
    (or the explicit rerank_pool_size override) candidates rather than just
    top_k — a cross-encoder can only reorder what it's shown, so truncating
    to top_k before reranking would silently discard better chunks that
    ranked outside the top_k in the bi-encoder/BM25 lists.
    """
    mock_semantic.return_value = []
    mock_keyword.return_value = []
    mock_rerank.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=5, rerank_pool_size=50)

    _, kwargs = mock_semantic.call_args
    assert kwargs["top_k"] == 50
    keyword_args, keyword_kwargs = mock_keyword.call_args
    assert keyword_kwargs.get("top_k", keyword_args[-1] if keyword_args else None) == 50


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_skips_reranker_call_when_fused_list_is_empty(mock_semantic, mock_keyword, mock_rerank):
    mock_semantic.return_value = []
    mock_keyword.return_value = []

    db = object()
    results = hybrid_search(db, "query", top_k=5)

    mock_rerank.assert_not_called()
    assert results == []


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_clamps_rerank_pool_size_to_max(mock_semantic, mock_keyword, mock_rerank):
    """
    rerank_pool_size can come straight from a caller (e.g. the
    /search/hybrid query param) — an absurdly large request must not
    translate into fetching that many rows from each leg or scoring that
    many candidates through the cross-encoder.
    """
    mock_semantic.return_value = []
    mock_keyword.return_value = []
    mock_rerank.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=5, rerank_pool_size=100_000)

    _, kwargs = mock_semantic.call_args
    assert kwargs["top_k"] == settings.rerank_pool_size_max
    keyword_args, keyword_kwargs = mock_keyword.call_args
    assert keyword_kwargs.get("top_k", keyword_args[-1] if keyword_args else None) == settings.rerank_pool_size_max


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_reranks_full_fused_list_not_a_pool_size_slice(mock_semantic, mock_keyword, mock_rerank):
    """
    Regression test: RRF can return up to 2 * pool_size unique items when a
    chunk is found by only one of the two search legs. Passing
    fused[:pool_size] to the reranker (rather than the full fused list)
    would silently drop up to half of those candidates before the
    cross-encoder ever saw them.
    """
    mock_semantic.return_value = [{"id": f"vec-{i}", "document_id": "d", "text": f"semantic chunk {i}"} for i in range(3)]
    mock_keyword.return_value = [
        {"id": f"kw-{i}", "document_id": "d", "embedding_id": None, "text": f"keyword chunk {i}"} for i in range(3)
    ]
    mock_rerank.return_value = []

    db = object()
    hybrid_search(db, "query", top_k=2, rerank_pool_size=3)

    passed_hits = mock_rerank.call_args[0][1]
    # 3 semantic-only + 3 keyword-only = 6 unique fused candidates, none
    # should have been dropped before reaching the reranker.
    assert len(passed_hits) == 6


@patch("app.search.hybrid.cross_encoder_rerank")
@patch("app.search.hybrid.keyword_search")
@patch("app.search.hybrid.semantic_search")
def test_hybrid_search_falls_back_to_fused_results_if_reranker_raises(mock_semantic, mock_keyword, mock_rerank):
    """
    A reranker failure (model load error, timeout, OOM, etc.) must degrade
    to the already-good fused results instead of taking down the whole
    request — /search/hybrid and /ask both depend on this.
    """
    mock_semantic.return_value = [
        {"id": "vec-1", "document_id": "doc-1", "text": "first"},
        {"id": "vec-2", "document_id": "doc-2", "text": "second"},
    ]
    mock_keyword.return_value = []
    mock_rerank.side_effect = TimeoutError("reranker timed out")

    db = object()
    results = hybrid_search(db, "query", top_k=5)

    assert len(results) == 2
    assert all("rerank_score" not in r for r in results)
