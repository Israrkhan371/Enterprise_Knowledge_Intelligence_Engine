"""
Tests for app/rag/citation_check.py. embed_texts() is mocked so these run
without a live embedding model — the tests instead control the returned
vectors directly to exercise the similarity-threshold branch logic.
"""
from unittest.mock import patch

from app.rag.citation_check import extract_cited_indices, verify_citations


def test_extract_cited_indices_finds_all_bracket_numbers():
    assert extract_cited_indices("Fact one [1]. Fact two [2]. Also [1] again.") == [1, 2]


def test_extract_cited_indices_returns_empty_for_no_citations():
    assert extract_cited_indices("An answer with no citations at all.") == []


@patch("app.rag.citation_check.embed_texts")
def test_verify_citations_passes_when_similarity_is_high(mock_embed_texts):
    # Identical normalized vectors -> cosine similarity 1.0, well above threshold.
    mock_embed_texts.return_value = [[1.0, 0.0], [1.0, 0.0]]
    sources = [{"index": 1, "text": "EKIE is a knowledge platform."}]

    result = verify_citations("EKIE is a knowledge platform. [1]", sources)

    assert result["verified"] is True
    assert result["flags"] == []
    assert result["cited_sources"] == [1]


@patch("app.rag.citation_check.embed_texts")
def test_verify_citations_flags_low_similarity(mock_embed_texts):
    # Orthogonal vectors -> cosine similarity 0.0, below the default 0.55 threshold.
    mock_embed_texts.return_value = [[1.0, 0.0], [0.0, 1.0]]
    sources = [{"index": 1, "text": "completely unrelated source text"}]

    result = verify_citations("An unsupported claim. [1]", sources)

    assert result["verified"] is False
    assert len(result["flags"]) == 1
    assert "low similarity" in result["flags"][0]["issue"]


def test_verify_citations_flags_citation_with_no_matching_source():
    # No sources at all -> [1] can't be resolved, doesn't need embed_texts mocked
    # since that branch returns before calling it.
    result = verify_citations("A claim citing something missing. [1]", sources=[])

    assert result["verified"] is False
    assert "no matching source" in result["flags"][0]["issue"]


@patch("app.rag.citation_check.embed_texts")
def test_verify_citations_handles_multiple_sentences_independently(mock_embed_texts):
    # First sentence: high similarity (verified). Second: low (flagged).
    mock_embed_texts.side_effect = [
        [[1.0, 0.0], [1.0, 0.0]],   # sentence 1 vs source 1
        [[1.0, 0.0], [0.0, 1.0]],   # sentence 2 vs source 2
    ]
    sources = [
        {"index": 1, "text": "supported fact"},
        {"index": 2, "text": "unrelated text"},
    ]

    result = verify_citations("Supported fact. [1] Unsupported claim. [2]", sources)

    assert result["verified"] is False
    assert len(result["flags"]) == 1
    assert "[2]" in result["flags"][0]["issue"]


def test_verify_citations_returns_verified_true_for_no_citations():
    result = verify_citations("A plain answer with no citations.", sources=[])

    assert result["verified"] is True
    assert result["cited_sources"] == []
    assert result["flags"] == []
