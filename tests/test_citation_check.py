"""
Tests for app/rag/citation_check.py. get_reranker() is mocked so these run
without a live cross-encoder model — the tests instead control the
returned relevance scores directly to exercise the threshold branch logic.
"""
from unittest.mock import MagicMock, patch

from app.rag.citation_check import extract_cited_indices, verify_citations


def test_extract_cited_indices_finds_all_bracket_numbers():
    assert extract_cited_indices("Fact one [1]. Fact two [2]. Also [1] again.") == [1, 2]


def test_extract_cited_indices_returns_empty_for_no_citations():
    assert extract_cited_indices("An answer with no citations at all.") == []


@patch("app.rag.citation_check.get_reranker")
def test_verify_citations_passes_when_relevance_is_high(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [5.0]  # well above the default 0.0 threshold
    mock_get_reranker.return_value = mock_reranker
    sources = [{"index": 1, "text": "EKIE is a knowledge platform."}]

    result = verify_citations("EKIE is a knowledge platform. [1]", sources)

    assert result["verified"] is True
    assert result["flags"] == []
    assert result["cited_sources"] == [1]


@patch("app.rag.citation_check.get_reranker")
def test_verify_citations_flags_low_relevance(mock_get_reranker):
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [-4.0]  # well below the default 0.0 threshold
    mock_get_reranker.return_value = mock_reranker
    sources = [{"index": 1, "text": "completely unrelated source text"}]

    result = verify_citations("An unsupported claim. [1]", sources)

    assert result["verified"] is False
    assert len(result["flags"]) == 1
    flag = result["flags"][0]
    assert flag["kind"] == "low_relevance"
    assert flag["cited_source_index"] == 1
    assert flag["score"] == -4.0
    assert "low relevance" in flag["issue"]


def test_verify_citations_flags_citation_with_no_matching_source():
    # No sources at all -> [1] can't be resolved, doesn't need the reranker
    # mocked since that branch returns before calling it.
    result = verify_citations("A claim citing something missing. [1]", sources=[])

    assert result["verified"] is False
    flag = result["flags"][0]
    assert flag["kind"] == "no_matching_source"
    assert flag["cited_source_index"] == 1
    assert flag["score"] is None
    assert "no matching source" in flag["issue"]


@patch("app.rag.citation_check.get_reranker")
def test_verify_citations_handles_multiple_sentences_independently(mock_get_reranker):
    # First sentence: high relevance (verified). Second: low (flagged).
    mock_reranker = MagicMock()
    mock_reranker.predict.side_effect = [[5.0], [-4.0]]
    mock_get_reranker.return_value = mock_reranker
    sources = [
        {"index": 1, "text": "supported fact"},
        {"index": 2, "text": "unrelated text"},
    ]

    result = verify_citations("Supported fact. [1] Unsupported claim. [2]", sources)

    assert result["verified"] is False
    assert len(result["flags"]) == 1
    assert result["flags"][0]["cited_source_index"] == 2


def test_verify_citations_returns_verified_true_for_no_citations():
    result = verify_citations("A plain answer with no citations.", sources=[])

    assert result["verified"] is True
    assert result["cited_sources"] == []
    assert result["flags"] == []


@patch("app.rag.citation_check.get_reranker")
def test_verify_citations_respects_custom_relevance_threshold(mock_get_reranker):
    # A score that would pass the default 0.0 threshold but not a stricter one.
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [1.0]
    mock_get_reranker.return_value = mock_reranker
    sources = [{"index": 1, "text": "marginally related source text"}]

    result = verify_citations("A marginal claim. [1]", sources, relevance_threshold=2.0)

    assert result["verified"] is False
    assert "low relevance" in result["flags"][0]["issue"]
