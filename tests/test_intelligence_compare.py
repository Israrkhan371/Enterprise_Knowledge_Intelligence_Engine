"""
Tests for the document comparison feature in app/rag/intelligence.py:
_embedding_similarity(), _diff_texts(), and the compare_documents_full()
orchestration (embedding similarity + line diff + LLM summary).

app/rag/intelligence.py talks to a real sentence-transformers model
(via embed_texts) and the Gemini API (via the module-level _client) - both
are mocked here via unittest.mock.patch, the same approach already used in
tests/test_pipeline.py (which patches app.ingestion.pipeline.GraphStore)
and tests/test_graph_build.py (which patches Neo4j's GraphDatabase). No
live model or network calls are made by these tests.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.rag import intelligence


# ---------------------------------------------------------------------------
# _embedding_similarity
# ---------------------------------------------------------------------------

def test_embedding_similarity_normal():
    """Normalized vectors -> dot product is the cosine similarity, rounded to 3dp."""
    with patch("app.rag.intelligence.embed_texts", return_value=[[1.0, 0.0], [0.6, 0.8]]) as mock_embed:
        result = intelligence._embedding_similarity("doc a text", "doc b text")

    mock_embed.assert_called_once_with(["doc a text", "doc b text"])
    assert result == 0.6


@pytest.mark.parametrize("text_a,text_b", [
    ("", "something"),
    ("something", ""),
    ("   ", "something"),
    ("", ""),
])
def test_embedding_similarity_blank_text_returns_none_without_embedding(text_a, text_b):
    """Blank input on either side short-circuits before ever calling embed_texts."""
    with patch("app.rag.intelligence.embed_texts") as mock_embed:
        result = intelligence._embedding_similarity(text_a, text_b)

    assert result is None
    mock_embed.assert_not_called()


def test_embedding_similarity_handles_embedding_failure():
    """An embedding backend failure degrades to None instead of raising."""
    with patch("app.rag.intelligence.embed_texts", side_effect=RuntimeError("model unavailable")):
        result = intelligence._embedding_similarity("doc a text", "doc b text")

    assert result is None


# ---------------------------------------------------------------------------
# _diff_texts
# ---------------------------------------------------------------------------

def test_diff_texts_identical_returns_empty_list():
    assert intelligence._diff_texts("line1\nline2", "line1\nline2") == []


def test_diff_texts_detects_change():
    diff = intelligence._diff_texts("line1\nline2\nline3", "line1\nCHANGED\nline3")

    assert any(line.startswith("-line2") for line in diff)
    assert any(line.startswith("+CHANGED") for line in diff)


def test_diff_texts_truncates_large_diffs():
    text_a = "\n".join(f"a{i}" for i in range(1000))
    text_b = "\n".join(f"b{i}" for i in range(1000))

    diff = intelligence._diff_texts(text_a, text_b)

    assert len(diff) == intelligence._MAX_DIFF_LINES + 1  # +1 for the truncation marker
    assert "truncated" in diff[-1]


# ---------------------------------------------------------------------------
# compare_documents_full
# ---------------------------------------------------------------------------

def test_compare_documents_full_both_blank_skips_embedding_and_llm():
    """Two blank documents short-circuit before any embedding or LLM call."""
    with patch("app.rag.intelligence.embed_texts") as mock_embed, \
         patch("app.rag.intelligence.compare_documents") as mock_compare:
        result = intelligence.compare_documents_full("", "   ")

    mock_embed.assert_not_called()
    mock_compare.assert_not_called()
    assert result == {
        "similarity": None,
        "diff": [],
        "summary": "Both documents are empty - nothing to compare.",
    }


def test_compare_documents_full_llm_failure_falls_back_gracefully():
    """LLM failure degrades the summary but still returns a valid similarity/diff."""
    with patch("app.rag.intelligence.embed_texts", return_value=[[1.0, 0.0], [0.9, 0.1]]), \
         patch("app.rag.intelligence.compare_documents", side_effect=RuntimeError("Gemini timeout")):
        result = intelligence.compare_documents_full("line1\nline2", "line1\nCHANGED")

    assert result["summary"] == "Summary unavailable: the comparison service failed to generate a response."
    assert result["similarity"] == 0.9
    assert result["diff"]  # still computed locally, unaffected by the LLM failure


def test_compare_documents_full_happy_path_combines_all_three_fields():
    """Normal case: similarity, diff, and summary are all populated together."""
    with patch("app.rag.intelligence.embed_texts", return_value=[[1.0, 0.0], [0.8, 0.6]]), \
         patch("app.rag.intelligence.compare_documents", return_value="Docs differ in the second line."):
        result = intelligence.compare_documents_full("line1\nline2", "line1\nCHANGED")

    assert result["similarity"] == 0.8
    assert any(line.startswith("-line2") for line in result["diff"])
    assert result["summary"] == "Docs differ in the second line."


def test_compare_documents_forwards_both_texts_to_the_llm():
    """compare_documents() (used inside compare_documents_full) sends both documents' text to Gemini."""
    fake_response = MagicMock(text="comparison result")
    with patch.object(intelligence._client.models, "generate_content", return_value=fake_response) as mock_generate:
        result = intelligence.compare_documents("Document A body", "Document B body")

    assert result == "comparison result"
    _, kwargs = mock_generate.call_args
    assert "Document A body" in kwargs["contents"]
    assert "Document B body" in kwargs["contents"]


# ---------------------------------------------------------------------------
# compare_documents - chunked summary path for long documents
# ---------------------------------------------------------------------------

def test_compare_documents_short_docs_use_single_inline_call():
    """Both documents at/under _MAX_INLINE_CHARS -> one call, full untruncated text."""
    text_a = "a" * intelligence._MAX_INLINE_CHARS
    text_b = "b" * intelligence._MAX_INLINE_CHARS
    fake_response = MagicMock(text="short-doc summary")

    with patch.object(intelligence._client.models, "generate_content", return_value=fake_response) as mock_generate:
        result = intelligence.compare_documents(text_a, text_b)

    assert result == "short-doc summary"
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    # Full text present, not truncated to the old 4000-char slice.
    assert text_a in kwargs["contents"]
    assert text_b in kwargs["contents"]


def test_compare_documents_long_docs_use_chunk_summarize_then_combine():
    """Either document over _MAX_INLINE_CHARS -> chunk+summarize each, then one combining call."""
    long_text_a = "word " * 5000  # well over _MAX_INLINE_CHARS and over one chunk_text() chunk
    long_text_b = "term " * 5000

    call_log = []

    def fake_generate_content(model, contents, config=None):
        call_log.append(contents)
        response = MagicMock()
        if len(call_log) <= len(intelligence.chunk_text(long_text_a)) + len(intelligence.chunk_text(long_text_b)):
            response.text = f"chunk summary {len(call_log)}"
        else:
            response.text = "final combined summary"
        return response

    with patch.object(intelligence._client.models, "generate_content", side_effect=fake_generate_content):
        result = intelligence.compare_documents(long_text_a, long_text_b)

    assert result == "final combined summary"
    # One call per chunk of each doc, plus one final combining call.
    expected_chunk_calls = len(intelligence.chunk_text(long_text_a)) + len(intelligence.chunk_text(long_text_b))
    assert len(call_log) == expected_chunk_calls + 1
    # The final combining prompt should reference chunk summaries, not raw text.
    assert "chunk summary 1" in call_log[-1]


def test_chunk_summarize_covers_full_document_not_just_the_start():
    """_chunk_summarize should summarize every chunk chunk_text() produces, not just the first."""
    long_text = "sentence " * 5000
    expected_chunks = intelligence.chunk_text(long_text)
    assert len(expected_chunks) > 1  # sanity check this text actually needs multiple chunks

    with patch("app.rag.intelligence.summarize_document", side_effect=lambda c: f"summary of: {c[:10]}") as mock_summarize:
        summaries = intelligence._chunk_summarize(long_text)

    assert mock_summarize.call_count == len(expected_chunks)
    assert len(summaries) == len(expected_chunks)


# ---------------------------------------------------------------------------
# Gemini call timeout handling
# ---------------------------------------------------------------------------

def test_generate_content_raises_timeout_error_on_slow_call():
    """A Gemini call that exceeds the configured timeout raises TimeoutError."""
    import time

    def slow_generate_content(model, contents, config=None):
        time.sleep(2)
        return MagicMock(text="too late")

    with patch.object(intelligence._client.models, "generate_content", side_effect=slow_generate_content), \
         patch.object(intelligence.settings, "gemini_timeout_seconds", 0.1):
        with pytest.raises(TimeoutError):
            intelligence._generate_content("some prompt")


def test_generate_content_succeeds_within_timeout():
    """A normal, fast Gemini call is unaffected by the timeout wrapper."""
    fake_response = MagicMock(text="fast summary")
    with patch.object(intelligence._client.models, "generate_content", return_value=fake_response), \
         patch.object(intelligence.settings, "gemini_timeout_seconds", 5.0):
        result = intelligence._generate_content("some prompt")

    assert result == "fast summary"


def test_compare_documents_full_propagates_timeout_error():
    """A timeout in the LLM summary call is NOT swallowed into the fallback
    message - it propagates so the API route can return a 504."""
    with patch("app.rag.intelligence.embed_texts", return_value=[[1.0, 0.0], [0.9, 0.1]]), \
         patch("app.rag.intelligence.compare_documents", side_effect=TimeoutError("Gemini request timed out after 30s")):
        with pytest.raises(TimeoutError):
            intelligence.compare_documents_full("line1\nline2", "line1\nCHANGED")


def test_compare_documents_full_still_falls_back_for_non_timeout_llm_errors():
    """Non-timeout LLM errors keep the existing graceful-degradation behavior."""
    with patch("app.rag.intelligence.embed_texts", return_value=[[1.0, 0.0], [0.9, 0.1]]), \
         patch("app.rag.intelligence.compare_documents", side_effect=RuntimeError("some other Gemini error")):
        result = intelligence.compare_documents_full("line1\nline2", "line1\nCHANGED")

    assert result["summary"] == "Summary unavailable: the comparison service failed to generate a response."


def test_call_with_timeout_raises_timeout_error_directly():
    """Unit test of the shared helper itself, independent of intelligence.py."""
    import time
    from app.rag.gemini_utils import call_with_timeout

    def slow_fn():
        time.sleep(2)
        return "done"

    with pytest.raises(TimeoutError):
        call_with_timeout(slow_fn, timeout_seconds=0.1)


def test_call_with_timeout_returns_result_when_fast_enough():
    from app.rag.gemini_utils import call_with_timeout

    result = call_with_timeout(lambda: "quick result", timeout_seconds=5.0)

    assert result == "quick result"
