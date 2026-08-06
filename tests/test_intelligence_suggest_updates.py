"""
Tests for suggest_document_updates() in app/rag/intelligence.py (Week 3
checkpoint: "Suggest document updates" AI capability).

Follows the same MagicMock db + patched embed_texts/LLM approach as
tests/test_intelligence_compare.py and tests/test_version_intelligence.py -
no live model, embedding, or network calls.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.rag import intelligence


def _doc(id_, title="doc", raw_text="some text", updated_at=None):
    d = MagicMock()
    d.id = id_
    d.title = title
    d.raw_text = raw_text
    d.updated_at = updated_at
    return d


def _mock_db(docs_by_id):
    db = MagicMock()
    db.get.side_effect = lambda model, id_: docs_by_id.get(id_)
    return db


def test_returns_none_when_document_missing():
    db = _mock_db({})
    assert intelligence.suggest_document_updates(db, "missing-id") is None


def test_no_related_documents_short_circuits_without_llm_call():
    """No semantic hits at all -> plain no-comparison message, no Gemini call."""
    target = _doc("a", raw_text="some content", updated_at=datetime.utcnow())
    db = _mock_db({"a": target})

    with patch("app.search.semantic.semantic_search", return_value=[]) as mock_search, \
         patch.object(intelligence, "_generate_content") as mock_generate:
        result = intelligence.suggest_document_updates(db, "a")

    mock_search.assert_called_once()
    mock_generate.assert_not_called()
    assert result["document_id"] == "a"
    assert result["related_documents"] == []
    assert "No fresher" in result["suggested_updates"]


def test_hits_with_no_updated_at_are_skipped_not_fresher():
    """A related hit whose Document row has no updated_at can't be shown to
    be fresher, so it's excluded rather than assumed related."""
    now = datetime.utcnow()
    target = _doc("a", raw_text="text", updated_at=now)
    stale_related = _doc("b", raw_text="other text", updated_at=None)
    db = _mock_db({"a": target, "b": stale_related})

    with patch("app.search.semantic.semantic_search", return_value=[{"document_id": "b"}]), \
         patch.object(intelligence, "_generate_content") as mock_generate:
        result = intelligence.suggest_document_updates(db, "a")

    mock_generate.assert_not_called()
    assert result["related_documents"] == []


def test_only_fresher_documents_are_treated_as_related():
    """An older or equally-old semantic hit is excluded; only a genuinely
    fresher document counts as 'related' for update suggestions."""
    now = datetime.utcnow()
    target = _doc("a", raw_text="text", updated_at=now)
    older = _doc("b", title="Older Doc", raw_text="old", updated_at=now - timedelta(days=10))
    fresher = _doc("c", title="Fresher Doc", raw_text="new", updated_at=now + timedelta(days=10))
    db = _mock_db({"a": target, "b": older, "c": fresher})

    hits = [{"document_id": "b"}, {"document_id": "c"}]
    with patch("app.search.semantic.semantic_search", return_value=hits), \
         patch.object(intelligence, "_generate_content", return_value="Consider adding X.") as mock_generate:
        result = intelligence.suggest_document_updates(db, "a")

    assert [r["document_id"] for r in result["related_documents"]] == ["c"]
    mock_generate.assert_called_once()
    prompt = mock_generate.call_args[0][0]
    assert "Fresher Doc" in prompt
    assert "Older Doc" not in prompt
    assert result["suggested_updates"] == "Consider adding X."


def test_duplicate_document_ids_in_hits_are_deduplicated():
    """Multiple chunk hits from the same document should only count once."""
    now = datetime.utcnow()
    target = _doc("a", raw_text="text", updated_at=now)
    fresher = _doc("c", title="Fresher Doc", raw_text="new", updated_at=now + timedelta(days=10))
    db = _mock_db({"a": target, "c": fresher})

    hits = [{"document_id": "c"}, {"document_id": "c"}, {"document_id": "c"}]
    with patch("app.search.semantic.semantic_search", return_value=hits), \
         patch.object(intelligence, "_generate_content", return_value="ok"):
        result = intelligence.suggest_document_updates(db, "a")

    assert len(result["related_documents"]) == 1


def test_is_outdated_reflects_staleness_window():
    now = datetime.utcnow()
    stale_target = _doc("a", raw_text="text", updated_at=now - timedelta(days=400))
    db = _mock_db({"a": stale_target})

    with patch("app.search.semantic.semantic_search", return_value=[]):
        result = intelligence.suggest_document_updates(db, "a", staleness_days=180)

    assert result["is_outdated"] is True


def test_llm_failure_falls_back_gracefully_but_still_lists_related_docs():
    now = datetime.utcnow()
    target = _doc("a", raw_text="text", updated_at=now)
    fresher = _doc("c", title="Fresher Doc", raw_text="new", updated_at=now + timedelta(days=10))
    db = _mock_db({"a": target, "c": fresher})

    with patch("app.search.semantic.semantic_search", return_value=[{"document_id": "c"}]), \
         patch.object(intelligence, "_generate_content", side_effect=RuntimeError("Gemini down")):
        result = intelligence.suggest_document_updates(db, "a")

    assert "unavailable" in result["suggested_updates"]
    assert len(result["related_documents"]) == 1  # still populated, unaffected by the LLM failure


def test_llm_timeout_propagates_rather_than_being_swallowed():
    now = datetime.utcnow()
    target = _doc("a", raw_text="text", updated_at=now)
    fresher = _doc("c", title="Fresher Doc", raw_text="new", updated_at=now + timedelta(days=10))
    db = _mock_db({"a": target, "c": fresher})

    with patch("app.search.semantic.semantic_search", return_value=[{"document_id": "c"}]), \
         patch.object(intelligence, "_generate_content", side_effect=TimeoutError("timed out")):
        with pytest.raises(TimeoutError):
            intelligence.suggest_document_updates(db, "a")


def test_blank_document_text_skips_semantic_search():
    target = _doc("a", title="Untitled", raw_text="   ", updated_at=datetime.utcnow())
    target.title = ""  # both raw_text and title blank -> no seed text at all
    db = _mock_db({"a": target})

    with patch("app.search.semantic.semantic_search") as mock_search:
        result = intelligence.suggest_document_updates(db, "a")

    mock_search.assert_not_called()
    assert result["related_documents"] == []