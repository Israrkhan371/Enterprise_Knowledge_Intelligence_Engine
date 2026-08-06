"""
Tests for app/graph/coverage.py: detect_missing_knowledge(). Neo4j is
mocked throughout via GraphStore.driver.session, following the same
pattern as tests/test_graph_build.py.
"""
from unittest.mock import MagicMock, patch

from app.graph import coverage


def _make_store_with_records(records: list[dict]):
    """dict(record) in the real code just needs something dict()-copyable,
    so plain dicts stand in fine for neo4j Record objects here."""
    with patch("app.graph.coverage.GraphStore.__init__", return_value=None):
        store = coverage.GraphStore()
    store.close = MagicMock()

    mock_session = MagicMock()
    mock_session.run.return_value = records

    mock_session_cm = MagicMock()
    mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
    mock_session_cm.__exit__ = MagicMock(return_value=False)
    store.driver = MagicMock()
    store.driver.session = MagicMock(return_value=mock_session_cm)
    return store, mock_session


def test_flags_entity_with_no_dedicated_document():
    records = [{
        "entity": "Kubernetes", "label": "TECH", "mention_count": 5,
        "titles": ["Deployment Guide", "Onboarding Handbook"],
    }]
    store, _ = _make_store_with_records(records)

    with patch("app.graph.coverage.GraphStore", return_value=store):
        result = coverage.detect_missing_knowledge(min_mentions=3)

    assert len(result) == 1
    assert result[0]["entity"] == "Kubernetes"
    assert result[0]["mentioned_in_document_count"] == 5


def test_excludes_entity_with_dedicated_document():
    records = [{
        "entity": "Kubernetes", "label": "TECH", "mention_count": 5,
        "titles": ["Kubernetes Deployment Guide", "Onboarding Handbook"],
    }]
    store, _ = _make_store_with_records(records)

    with patch("app.graph.coverage.GraphStore", return_value=store):
        result = coverage.detect_missing_knowledge(min_mentions=3)

    assert result == []


def test_returns_empty_list_on_neo4j_failure():
    with patch("app.graph.coverage.GraphStore", side_effect=RuntimeError("connection refused")):
        result = coverage.detect_missing_knowledge()
    assert result == []


def test_has_dedicated_document_is_case_insensitive():
    assert coverage._has_dedicated_document("kubernetes", ["Kubernetes Guide"]) is True
    assert coverage._has_dedicated_document("Kubernetes", ["kubernetes basics"]) is True
    assert coverage._has_dedicated_document("Kubernetes", ["Unrelated Title"]) is False


def test_has_dedicated_document_handles_empty_entity_name():
    assert coverage._has_dedicated_document("", ["Some Title"]) is False
