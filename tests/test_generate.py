"""
Tests for app/rag/generate.py. Mocks hybrid_search() and the Gemini client
so these run without a live DB, ChromaDB, or API key.

build_context_block()'s document_id handling is specifically covered here
because of a real bug found 2026-07-30: semantic_search() hits didn't
carry a top-level "document_id" (only nested in "metadata"), so
build_context_block()'s `hit.get("document_id", hit.get("id"))` fallback
silently substituted the ChromaDB vector id for the real document id in
citations sourced from semantic-only hits. Fixed upstream in
app/search/semantic.py; the tests here confirm generate.py's own handling
is correct given a properly-shaped hit.
"""
from unittest.mock import MagicMock, patch

from app.rag.generate import build_context_block, generate_answer


def test_build_context_block_numbers_sources_from_one():
    hits = [
        {"document_id": "doc-1", "text": "first chunk"},
        {"document_id": "doc-2", "text": "second chunk"},
    ]

    context = build_context_block(hits)

    assert "[1]" in context
    assert "[2]" in context
    assert "doc-1" in context
    assert "doc-2" in context


def test_build_context_block_uses_document_id_when_present():
    """The regression case: a properly-shaped hit (post-fix) has a
    top-level document_id that must be used, not the chunk/vector id."""
    hits = [{"document_id": "real-document-id", "id": "doc-1::vec-abc", "text": "chunk text"}]

    context = build_context_block(hits)

    assert "real-document-id" in context
    assert "vec-abc" not in context


def test_build_context_block_falls_back_to_id_if_document_id_truly_missing():
    """Defensive fallback for a malformed/legacy hit dict with no
    document_id key at all — better to show *something* than crash."""
    hits = [{"id": "fallback-id", "text": "chunk text"}]

    context = build_context_block(hits)

    assert "fallback-id" in context


@patch("app.rag.generate._client")
@patch("app.rag.generate.hybrid_search")
def test_generate_answer_passes_query_and_top_k_to_hybrid_search(mock_hybrid, mock_client):
    mock_hybrid.return_value = []
    mock_response = MagicMock()
    mock_response.text = "answer"
    mock_client.models.generate_content.return_value = mock_response

    db = object()
    generate_answer(db, "what is EKIE?", top_k=4)

    args, kwargs = mock_hybrid.call_args
    assert args[1] == "what is EKIE?"
    assert kwargs["top_k"] == 4


@patch("app.rag.generate._client")
@patch("app.rag.generate.hybrid_search")
def test_generate_answer_returns_answer_and_indexed_sources(mock_hybrid, mock_client):
    mock_hybrid.return_value = [
        {"document_id": "doc-1", "text": "EKIE is a knowledge engine, described in great detail here."},
    ]
    mock_response = MagicMock()
    mock_response.text = "EKIE is a knowledge engine. [1]"
    mock_client.models.generate_content.return_value = mock_response

    db = object()
    result = generate_answer(db, "what is EKIE?")

    assert result["answer"] == "EKIE is a knowledge engine. [1]"
    assert result["sources"][0]["index"] == 1
    assert result["sources"][0]["document_id"] == "doc-1"


@patch("app.rag.generate._client")
@patch("app.rag.generate.hybrid_search")
def test_generate_answer_truncates_source_text_to_300_chars(mock_hybrid, mock_client):
    long_text = "x" * 1000
    mock_hybrid.return_value = [{"document_id": "doc-1", "text": long_text}]
    mock_response = MagicMock()
    mock_response.text = "answer [1]"
    mock_client.models.generate_content.return_value = mock_response

    db = object()
    result = generate_answer(db, "query")

    assert len(result["sources"][0]["text"]) == 300


@patch("app.rag.generate._client")
@patch("app.rag.generate.hybrid_search")
def test_generate_answer_handles_empty_gemini_response(mock_hybrid, mock_client):
    mock_hybrid.return_value = []
    mock_response = MagicMock()
    mock_response.text = None
    mock_client.models.generate_content.return_value = mock_response

    db = object()
    result = generate_answer(db, "query")

    assert result["answer"] == ""
