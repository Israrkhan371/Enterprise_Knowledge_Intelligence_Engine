"""
Tests for detect_duplicates() and its helper _fetch_stored_embeddings() in
app/rag/intelligence.py.

detect_duplicates() talks to Postgres (via db.execute(text(...))), ChromaDB
(via get_collection(), to reuse embeddings computed at ingestion time), and
sentence-transformers (via embed_texts(), only as a fallback for chunks with
no stored embedding). All three are mocked here, the same approach used in
tests/test_intelligence_compare.py for _embedding_similarity(). No live DB,
Chroma, or model calls are made by these tests.
"""
from unittest.mock import MagicMock, patch

from app.rag import intelligence


def _row(id_, document_id, text_, embedding_id=None):
    """Build a fake SQLAlchemy Row supporting .id/.document_id/.text/.embedding_id access."""
    row = MagicMock()
    row.id = id_
    row.document_id = document_id
    row.text = text_
    row.embedding_id = embedding_id
    return row


def _mock_db(rows):
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = rows
    return db


def _mock_collection(get_return):
    collection = MagicMock()
    collection.get.return_value = get_return
    return collection


# ---------------------------------------------------------------------------
# _fetch_stored_embeddings
# ---------------------------------------------------------------------------

def test_fetch_stored_embeddings_empty_input_skips_chroma_call():
    with patch("app.rag.intelligence.get_collection") as mock_get_collection:
        result = intelligence._fetch_stored_embeddings([])

    assert result == {}
    mock_get_collection.assert_not_called()


def test_fetch_stored_embeddings_returns_id_to_vector_map():
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence._fetch_stored_embeddings(["e1", "e2"])

    assert result == {"e1": [1.0, 0.0], "e2": [0.0, 1.0]}


def test_fetch_stored_embeddings_chroma_failure_returns_empty_dict():
    with patch("app.rag.intelligence.get_collection", side_effect=RuntimeError("chroma down")):
        result = intelligence._fetch_stored_embeddings(["e1"])

    assert result == {}


# ---------------------------------------------------------------------------
# detect_duplicates
# ---------------------------------------------------------------------------

def test_returns_empty_list_when_fewer_than_two_chunks():
    db = _mock_db([_row("c1", "doc-a", "only chunk", "e1")])
    with patch("app.rag.intelligence.get_collection") as mock_get_collection, \
         patch("app.rag.intelligence.embed_texts") as mock_embed:
        result = intelligence.detect_duplicates(db)

    assert result == []
    mock_get_collection.assert_not_called()
    mock_embed.assert_not_called()


def test_returns_empty_list_when_no_chunks():
    db = _mock_db([])
    with patch("app.rag.intelligence.embed_texts") as mock_embed:
        result = intelligence.detect_duplicates(db)

    assert result == []
    mock_embed.assert_not_called()


def test_uses_stored_embeddings_without_calling_embed_texts():
    """Chunks that already have an embedding_id are looked up in Chroma; embed_texts
    is never called when nothing is missing."""
    rows = [
        _row("c1", "doc-a", "chunk text a", "e1"),
        _row("c2", "doc-b", "chunk text b", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [0.995, 0.0998]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection), \
         patch("app.rag.intelligence.embed_texts") as mock_embed:
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    mock_embed.assert_not_called()
    assert len(result) == 1
    assert result[0]["document_a"] == "doc-a"
    assert result[0]["document_b"] == "doc-b"
    assert result[0]["similarity"] >= 0.92


def test_falls_back_to_embed_texts_for_chunks_missing_embedding_id():
    rows = [
        _row("c1", "doc-a", "chunk text a", "e1"),
        _row("c2", "doc-b", "chunk text b", None),  # never embedded/stored
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1"], "embeddings": [[1.0, 0.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection), \
         patch("app.rag.intelligence.embed_texts", return_value=[[0.995, 0.0998]]) as mock_embed:
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    mock_embed.assert_called_once_with(["chunk text b"])
    assert len(result) == 1
    assert result[0]["document_a"] == "doc-a"
    assert result[0]["document_b"] == "doc-b"


def test_falls_back_to_embed_texts_when_chroma_lookup_misses_an_id():
    """Chroma can return fewer ids than requested (e.g. a stale/deleted entry) -
    those chunks should be treated as missing and re-embedded, not silently dropped."""
    rows = [
        _row("c1", "doc-a", "chunk text a", "e1"),
        _row("c2", "doc-b", "chunk text b", "e2"),
    ]
    db = _mock_db(rows)
    # Chroma only has e1, not e2
    collection = _mock_collection({"ids": ["e1"], "embeddings": [[1.0, 0.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection), \
         patch("app.rag.intelligence.embed_texts", return_value=[[0.995, 0.0998]]) as mock_embed:
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    mock_embed.assert_called_once_with(["chunk text b"])
    assert len(result) == 1


def test_embedding_fallback_failure_excludes_affected_chunks_instead_of_raising():
    """If the fallback embed_texts() call fails, chunks that needed it are dropped
    from the comparison instead of the whole request raising."""
    rows = [
        _row("c1", "doc-a", "chunk text a", "e1"),
        _row("c2", "doc-b", "chunk text b", None),
        _row("c3", "doc-c", "chunk text c", None),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1"], "embeddings": [[1.0, 0.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection), \
         patch("app.rag.intelligence.embed_texts", side_effect=RuntimeError("model unavailable")):
        # Must not raise.
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    # Only c1 has a usable vector -> fewer than 2 valid chunks -> no pairs possible.
    assert result == []


def test_does_not_flag_pair_below_threshold():
    rows = [
        _row("c1", "doc-a", "chunk text a", "e1"),
        _row("c2", "doc-b", "chunk text b", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [0.0, 1.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert result == []


def test_same_document_pairs_are_excluded_even_if_identical():
    rows = [
        _row("c1", "doc-a", "chunk 1", "e1"),
        _row("c2", "doc-a", "chunk 2", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [1.0, 0.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert result == []


def test_boundary_similarity_equal_to_threshold_is_included():
    rows = [
        _row("c1", "doc-a", "chunk a", "e1"),
        _row("c2", "doc-b", "chunk b", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection(
        {"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [0.92, 0.3919183588453085]]}
    )
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert len(result) == 1
    assert result[0]["similarity"] == 0.92


def test_custom_similarity_threshold_is_respected():
    rows = [
        _row("c1", "doc-a", "chunk a", "e1"),
        _row("c2", "doc-b", "chunk b", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [0.6, 0.8]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.5)

    assert len(result) == 1
    assert result[0]["similarity"] == 0.6


def test_multiple_matching_chunk_pairs_collapse_to_one_row_per_document_pair():
    """doc-a has 2 chunks, doc-b has 1, and both doc-a chunks match doc-b's chunk.
    That's 2 matching chunk pairs but only ONE (doc-a, doc-b) document pair - the
    result should have exactly one row, not one per matching chunk pair."""
    rows = [
        _row("c1", "doc-a", "chunk a1", "e1"),
        _row("c2", "doc-a", "chunk a2", "e2"),
        _row("c3", "doc-b", "chunk b1", "e3"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({
        "ids": ["e1", "e2", "e3"],
        "embeddings": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    })
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert len(result) == 1
    assert result[0]["document_a"] == "doc-a"
    assert result[0]["document_b"] == "doc-b"


def test_document_pair_uses_max_similarity_across_matching_chunks():
    """When a document pair has multiple matching chunk pairs at different
    similarities, the reported similarity is the max, not the first found."""
    rows = [
        _row("c1", "doc-a", "chunk a1", "e1"),  # matches c3 at ~0.6 (below threshold alone)
        _row("c2", "doc-a", "chunk a2", "e2"),  # matches c3 at ~0.98 (the real duplicate)
        _row("c3", "doc-b", "chunk b1", "e3"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({
        "ids": ["e1", "e2", "e3"],
        "embeddings": [[0.6, 0.8], [1.0, 0.0], [0.98, 0.19899748742132399]],
    })
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert len(result) == 1
    assert result[0]["similarity"] == round(0.98, 3)


def test_document_pair_ordering_is_stable_regardless_of_row_order():
    """document_a/document_b are alphabetically sorted, so the same pair always
    reports the same way regardless of which document's chunk appeared first."""
    rows = [
        _row("c1", "doc-z", "chunk z", "e1"),
        _row("c2", "doc-a", "chunk a", "e2"),
    ]
    db = _mock_db(rows)
    collection = _mock_collection({"ids": ["e1", "e2"], "embeddings": [[1.0, 0.0], [1.0, 0.0]]})
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.92)

    assert result[0]["document_a"] == "doc-a"
    assert result[0]["document_b"] == "doc-z"


def test_results_sorted_by_similarity_descending():
    rows = [
        _row("c1", "doc-a", "chunk a", "e1"),
        _row("c2", "doc-b", "chunk b", "e2"),
        _row("c3", "doc-c", "chunk c", "e3"),
    ]
    db = _mock_db(rows)
    # (doc-a, doc-b) similar at ~0.6; (doc-a, doc-c) and (doc-b, doc-c) near 1.0
    collection = _mock_collection({
        "ids": ["e1", "e2", "e3"],
        "embeddings": [[0.6, 0.8], [1.0, 0.0], [1.0, 0.0]],
    })
    with patch("app.rag.intelligence.get_collection", return_value=collection):
        result = intelligence.detect_duplicates(db, similarity_threshold=0.5)

    sims = [d["similarity"] for d in result]
    assert sims == sorted(sims, reverse=True)
