"""
Tests for the on-demand document summarization feature in
app/rag/intelligence.py: summarize_document() (single-call leaf) and
summarize_document_full() (inline-vs-chunked orchestration).

Mirrors the approach in tests/test_intelligence_compare.py: the Gemini API
(via the module-level _client) is mocked via unittest.mock.patch. No live
model or network calls are made by these tests.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.rag import intelligence


# ---------------------------------------------------------------------------
# summarize_document_full - inline vs chunked routing
# ---------------------------------------------------------------------------

def test_summarize_document_full_blank_text_skips_llm():
    """Blank/whitespace-only input short-circuits before any LLM call."""
    with patch.object(intelligence._client.models, "generate_content") as mock_generate:
        result = intelligence.summarize_document_full("   ")

    mock_generate.assert_not_called()
    assert result == "Document is empty - nothing to summarize."


def test_summarize_document_full_short_doc_uses_single_inline_call():
    """Document at/under _MAX_INLINE_CHARS -> one call, full untruncated text."""
    text = "a" * intelligence._MAX_INLINE_CHARS
    fake_response = MagicMock(text="short-doc summary")

    with patch.object(intelligence._client.models, "generate_content", return_value=fake_response) as mock_generate:
        result = intelligence.summarize_document_full(text)

    assert result == "short-doc summary"
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert text in kwargs["contents"]


def test_summarize_document_full_long_doc_uses_chunk_summarize_then_combine():
    """Document over _MAX_INLINE_CHARS -> chunk+summarize each, then one combining call."""
    long_text = "word " * 5000  # well over _MAX_INLINE_CHARS and over one chunk_text() chunk
    expected_chunks = intelligence.chunk_text(long_text)
    assert len(expected_chunks) > 1  # sanity check this text actually needs multiple chunks

    call_log = []

    def fake_generate_content(model, contents, config=None):
        call_log.append(contents)
        response = MagicMock()
        if len(call_log) <= len(expected_chunks):
            response.text = f"chunk summary {len(call_log)}"
        else:
            response.text = "final combined summary"
        return response

    with patch.object(intelligence._client.models, "generate_content", side_effect=fake_generate_content):
        result = intelligence.summarize_document_full(long_text)

    assert result == "final combined summary"
    # One call per chunk, plus one final combining call - no recursive re-chunking.
    assert len(call_log) == len(expected_chunks) + 1
    # The final combining prompt should reference chunk summaries, not raw text.
    assert "chunk summary 1" in call_log[-1]


def test_summarize_document_full_does_not_recurse_on_oversized_chunk():
    """
    Regression test for the recursion bug this design avoids: even if an
    individual chunk is itself over _MAX_INLINE_CHARS, summarize_document()
    (the leaf called per-chunk) must never re-chunk - it always makes exactly
    one LLM call per invocation.
    """
    oversized_chunk = "x" * (intelligence._MAX_INLINE_CHARS + 500)
    fake_response = MagicMock(text="leaf summary")

    with patch.object(intelligence._client.models, "generate_content", return_value=fake_response) as mock_generate:
        result = intelligence.summarize_document(oversized_chunk)

    assert result == "leaf summary"
    mock_generate.assert_called_once()


# ---------------------------------------------------------------------------
# summarize_document_full - error handling (mirrors compare_documents_full)
# ---------------------------------------------------------------------------

def test_summarize_document_full_propagates_timeout_error():
    """A timeout in the LLM call is NOT swallowed into the fallback message -
    it propagates so the API route can return a 504."""
    with patch("app.rag.intelligence.summarize_document", side_effect=TimeoutError("Gemini request timed out after 30s")):
        with pytest.raises(TimeoutError):
            intelligence.summarize_document_full("short document text")


def test_summarize_document_full_falls_back_for_non_timeout_llm_errors():
    """Non-timeout LLM errors degrade to a fallback summary instead of raising."""
    with patch("app.rag.intelligence.summarize_document", side_effect=RuntimeError("some other Gemini error")):
        result = intelligence.summarize_document_full("short document text")

    assert result == "Summary unavailable: the summarization service failed to generate a response."


def test_summarize_document_full_chunked_path_propagates_timeout_error():
    """Timeout during the chunked path (either per-chunk or the combine call)
    also propagates, not just the inline path."""
    long_text = "word " * 5000
    with patch("app.rag.intelligence._chunk_summarize", side_effect=TimeoutError("Gemini request timed out after 30s")):
        with pytest.raises(TimeoutError):
            intelligence.summarize_document_full(long_text)


def test_summarize_document_full_chunked_path_falls_back_for_non_timeout_errors():
    long_text = "word " * 5000
    with patch("app.rag.intelligence._chunk_summarize", side_effect=RuntimeError("Gemini error")):
        result = intelligence.summarize_document_full(long_text)

    assert result == "Summary unavailable: the summarization service failed to generate a response."
