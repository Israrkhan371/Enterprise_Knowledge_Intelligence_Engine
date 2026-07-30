"""
Tests for app/search/context_aware.py's rewrite_query(), now wired to a
real Gemini call (previously a stub that always returned current_query
unchanged, regardless of history). Mocks the Gemini client the same way
app/rag/generate.py's callers would, so these run without a live API key
or network access.
"""
from unittest.mock import MagicMock, patch

from app.search.context_aware import rewrite_query


def test_rewrite_query_returns_unchanged_when_no_history():
    """No history means there's nothing to resolve pronouns/references
    against, so this should short-circuit without calling the LLM at all."""
    with patch("app.search.context_aware._client") as mock_client:
        result = rewrite_query("What is FastAPI?", [])

        assert result == "What is FastAPI?"
        mock_client.models.generate_content.assert_not_called()


@patch("app.search.context_aware._client")
def test_rewrite_query_calls_gemini_with_history_when_present(mock_client):
    mock_response = MagicMock()
    mock_response.text = "What are FastAPI's dependencies?"
    mock_client.models.generate_content.return_value = mock_response

    history = [
        {"role": "user", "content": "What is FastAPI?"},
        {"role": "assistant", "content": "FastAPI is a Python web framework."},
    ]
    result = rewrite_query("what about its dependencies?", history)

    assert result == "What are FastAPI's dependencies?"
    mock_client.models.generate_content.assert_called_once()


@patch("app.search.context_aware._client")
def test_rewrite_query_includes_recent_history_in_the_prompt(mock_client):
    mock_response = MagicMock()
    mock_response.text = "rewritten"
    mock_client.models.generate_content.return_value = mock_response

    history = [{"role": "user", "content": "Tell me about Neo4j."}]
    rewrite_query("how does it scale?", history)

    _, kwargs = mock_client.models.generate_content.call_args
    assert "Neo4j" in kwargs["contents"]
    assert "how does it scale?" in kwargs["contents"]


@patch("app.search.context_aware._client")
def test_rewrite_query_only_uses_last_four_turns(mock_client):
    mock_response = MagicMock()
    mock_response.text = "rewritten"
    mock_client.models.generate_content.return_value = mock_response

    history = [{"role": "user", "content": f"turn-{i}"} for i in range(10)]
    rewrite_query("follow-up", history)

    _, kwargs = mock_client.models.generate_content.call_args
    assert "turn-9" in kwargs["contents"]
    assert "turn-6" in kwargs["contents"]
    assert "turn-5" not in kwargs["contents"]  # only the last 4 turns (6,7,8,9)


@patch("app.search.context_aware._client")
def test_rewrite_query_falls_back_to_original_on_empty_response(mock_client):
    mock_response = MagicMock()
    mock_response.text = ""
    mock_client.models.generate_content.return_value = mock_response

    result = rewrite_query("original query", [{"role": "user", "content": "prior turn"}])

    assert result == "original query"


@patch("app.search.context_aware._client")
def test_rewrite_query_falls_back_to_original_on_api_error(mock_client):
    """
    A Gemini call failing (rate limit, network blip, bad API key) must not
    raise out of rewrite_query() and break /ask entirely — it should
    degrade gracefully to the un-rewritten query, same as how a Neo4j
    failure doesn't block ingestion in _populate_graph().
    """
    mock_client.models.generate_content.side_effect = Exception("API error")

    result = rewrite_query("original query", [{"role": "user", "content": "prior turn"}])

    assert result == "original query"


@patch("app.search.context_aware._client")
def test_rewrite_query_strips_whitespace_from_response(mock_client):
    mock_response = MagicMock()
    mock_response.text = "  rewritten query  \n"
    mock_client.models.generate_content.return_value = mock_response

    result = rewrite_query("original", [{"role": "user", "content": "prior turn"}])

    assert result == "rewritten query"
