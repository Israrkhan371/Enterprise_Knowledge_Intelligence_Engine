"""
Tests for app/rag/intelligence.py::detect_outdated()'s llm_cross_check
option - the "LLM cross-check" half of the "staleness heuristic + LLM
cross-check" case-study requirement, which previously only had the
staleness-heuristic half implemented.

Postgres and the LLM call are mocked. No live DB or Gemini calls made.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.rag import intelligence


def _row(**kwargs):
    row = MagicMock()
    for k, v in kwargs.items():
        setattr(row, k, v)
    return row


def test_default_behavior_unchanged_without_cross_check():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc", updated_at=datetime(2025, 1, 1)),
    ]

    result = intelligence.detect_outdated(db)

    assert result == [
        {"document_id": "doc-1", "title": "Old Doc", "last_updated": "2025-01-01T00:00:00"}
    ]
    assert "llm_verdict" not in result[0]


def test_cross_check_adds_verdict_when_related_content_found():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc", updated_at=datetime(2025, 1, 1)),
    ]
    db.execute.return_value.fetchone.return_value = _row(raw_text="old content")

    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", return_value="Looks superseded.") as mock_llm:
            result = intelligence.detect_outdated(db, llm_cross_check=True)

    assert result[0]["llm_verdict"] == "Looks superseded."
    mock_llm.assert_called_once()


def test_cross_check_notes_no_related_content_without_calling_llm():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc", updated_at=datetime(2025, 1, 1)),
    ]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=[]):
        with patch("app.rag.intelligence._generate_content") as mock_llm:
            result = intelligence.detect_outdated(db, llm_cross_check=True)

    assert "No newer related content" in result[0]["llm_verdict"]
    mock_llm.assert_not_called()


def test_cross_check_llm_timeout_propagates():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc", updated_at=datetime(2025, 1, 1)),
    ]
    db.execute.return_value.fetchone.return_value = _row(raw_text="old content")
    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", side_effect=TimeoutError):
            try:
                intelligence.detect_outdated(db, llm_cross_check=True)
                assert False, "expected TimeoutError to propagate"
            except TimeoutError:
                pass


def test_cross_check_llm_failure_degrades_gracefully_per_document():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc", updated_at=datetime(2025, 1, 1)),
    ]
    db.execute.return_value.fetchone.return_value = _row(raw_text="old content")
    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", side_effect=RuntimeError("gemini down")):
            result = intelligence.detect_outdated(db, llm_cross_check=True)

    assert result[0]["llm_verdict"] == "LLM cross-check failed."


def test_cross_check_handles_multiple_flagged_documents_independently():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        _row(id="doc-1", title="Old Doc 1", updated_at=datetime(2025, 1, 1)),
        _row(id="doc-2", title="Old Doc 2", updated_at=datetime(2025, 2, 1)),
    ]
    db.execute.return_value.fetchone.return_value = _row(raw_text="content")

    def related_side_effect(db_, doc_id, *a, **kw):
        return [] if doc_id == "doc-1" else [
            {"document_id": "doc-3", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}
        ]

    with patch("app.rag.intelligence._find_newer_related_documents", side_effect=related_side_effect):
        with patch("app.rag.intelligence._generate_content", return_value="Superseded."):
            result = intelligence.detect_outdated(db, llm_cross_check=True)

    assert "No newer related content" in result[0]["llm_verdict"]
    assert result[1]["llm_verdict"] == "Superseded."