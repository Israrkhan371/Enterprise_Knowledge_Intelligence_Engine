"""
Tests for app/search/keyword.py and its supporting GIN index.

keyword_search() is a thin wrapper around a raw Postgres full-text query
(to_tsvector/plainto_tsquery), so these tests mock db.execute() rather than
requiring a live Postgres connection — same pattern as test_semantic.py's
mocking of ChromaDB. They lock in: the params passed to the query, and how
result rows get reshaped into dicts.

Separately, test_document_chunks_has_fts_index() checks that
app.core.models.DocumentChunk actually declares the GIN index that
keyword_search()'s docstring promises exists — without it, keyword search
falls back to a sequential scan once the table has real volume.
"""
from unittest.mock import MagicMock

from app.search.keyword import keyword_search


def _make_mock_db(rows):
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = rows
    return db



def _fake_row(id_="chunk-1", document_id="doc-1", text="some chunk text", rank=0.5, embedding_id="vec-1"):
    row = MagicMock()
    row.id = id_
    row.document_id = document_id
    row.text = text
    row.rank = rank

    row.embedding_id = embedding_id
    return row


def test_keyword_search_passes_query_and_top_k_to_execute():
    db = _make_mock_db(rows=[])

    keyword_search(db, "florbington789", top_k=5)

    _, params = db.execute.call_args[0]
    assert params == {"query": "florbington789", "top_k": 5}


def test_keyword_search_defaults_top_k_to_ten():
    db = _make_mock_db(rows=[])

    keyword_search(db, "some query")

    _, params = db.execute.call_args[0]
    assert params["top_k"] == 10


def test_keyword_search_shapes_results_correctly():
    db = _make_mock_db(rows=[

        _fake_row(
            id_="chunk-1", document_id="doc-1", text="Pinecone is a vector database.",
            rank=0.061, embedding_id="doc-1::vec-abc",
        ),
    ])

    results = keyword_search(db, "Pinecone", top_k=3)

    assert results == [
        {
            "id": "chunk-1",
            "document_id": "doc-1",
            "embedding_id": "doc-1::vec-abc",
            "text": "Pinecone is a vector database.",
            "rank": 0.061,
        }
    ]

def test_keyword_search_passes_through_none_embedding_id():
    """
    A chunk that hasn't been embedded yet has embedding_id=NULL in Postgres.
    keyword_search() must pass that through as None rather than coercing it
    (e.g. to an empty string), since hybrid_search() relies on being able to
    distinguish "no embedding yet" from a real embedding_id when fusing.
    """
    db = _make_mock_db(rows=[_fake_row(embedding_id=None)])

    results = keyword_search(db, "query")

    assert results[0]["embedding_id"] is None

def test_keyword_search_returns_empty_list_for_no_matches():
    db = _make_mock_db(rows=[])

    results = keyword_search(db, "nonexistent-term")

    assert results == []


def test_keyword_search_coerces_id_and_rank_types():
    """
    id/document_id come back from asyncpg/psycopg as UUID objects, and rank
    as a Decimal-like numeric type in some drivers — keyword_search() must
    coerce both to plain str/float so results are JSON-serializable by the
    /search/keyword endpoint.
    """
    row = _fake_row(id_="not-a-plain-string-until-cast", rank="0.42")
    db = _make_mock_db(rows=[row])

    results = keyword_search(db, "query")

    assert isinstance(results[0]["id"], str)
    assert isinstance(results[0]["rank"], float)
    assert results[0]["rank"] == 0.42


def test_keyword_search_preserves_db_result_order():
    """
    ORDER BY rank DESC happens in SQL; keyword_search() must not re-sort or
    otherwise reorder what the database returns.
    """
    rows = [
        _fake_row(id_="chunk-high", rank=0.9),
        _fake_row(id_="chunk-mid", rank=0.5),
        _fake_row(id_="chunk-low", rank=0.1),
    ]
    db = _make_mock_db(rows=rows)

    results = keyword_search(db, "query", top_k=3)

    assert [r["id"] for r in results] == ["chunk-high", "chunk-mid", "chunk-low"]


def test_document_chunks_has_fts_index():
    """
    Regression guard for the missing-GIN-index issue: keyword_search()'s own
    docstring says it needs

        CREATE INDEX chunks_fts_idx ON document_chunks
        USING GIN (to_tsvector('english', text));

    but nothing declared it. This checks app.core.models.DocumentChunk
    actually defines a GIN index over a to_tsvector('english', ...)
    expression on the chunks table, so create_all() creates it for real.
    """
    from app.core.models import DocumentChunk

    indexes = DocumentChunk.__table__.indexes
    assert indexes, "DocumentChunk has no indexes declared"

    fts_indexes = [ix for ix in indexes if ix.kwargs.get("postgresql_using") == "gin"]
    assert fts_indexes, "No GIN index declared on document_chunks"

    fts_index = fts_indexes[0]
    expressions = [str(col) for col in fts_index.expressions]
    assert any("to_tsvector" in expr for expr in expressions), (
        f"GIN index isn't built on a to_tsvector(...) expression: {expressions}"
    )
