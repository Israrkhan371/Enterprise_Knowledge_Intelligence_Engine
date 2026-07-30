"""
Tests for GraphStore.upsert_cooccurrence's idempotency guard
(app/graph/build.py) — reprocessing the same document must not
double-count sentence/paragraph/document co-occurrences. Neo4j is mocked
throughout; these don't need a live database.
"""
from unittest.mock import MagicMock, patch

from app.graph.build import GraphStore


def _make_store_with_mock_session(read_result: dict):
    """Builds a GraphStore whose driver.session() context manager returns a
    MagicMock session. The 2nd session.run() call (the "read current
    state" query) returns `read_result`; the 1st (MERGE) and 3rd (SET) are
    just recorded on the mock for inspection."""
    with patch("app.graph.build.GraphDatabase"):
        store = GraphStore()

    mock_session = MagicMock()
    read_record = MagicMock()
    read_record.__getitem__.side_effect = lambda key: read_result[key]
    run_results = [MagicMock(), MagicMock(single=MagicMock(return_value=read_record)), MagicMock()]
    mock_session.run.side_effect = run_results

    mock_session_cm = MagicMock()
    mock_session_cm.__enter__ = MagicMock(return_value=mock_session)
    mock_session_cm.__exit__ = MagicMock(return_value=False)
    store.driver.session = MagicMock(return_value=mock_session_cm)
    return store, mock_session


def test_new_document_increments_counts():
    store, mock_session = _make_store_with_mock_session(
        {"evidence": [], "docs": [], "repos": []}
    )
    pair_agg = {
        ("Python", "FastAPI"): {
            "sentence_count": 3, "paragraph_count": 1, "document_count": 0,
            "evidence": {"import_statement"}, "sample_context": "...",
        }
    }

    store.upsert_cooccurrence("doc-1", "markdown", pair_agg)

    set_call = mock_session.run.call_args_list[2]
    kwargs = set_call.kwargs
    assert kwargs["sentence_count"] == 3
    assert kwargs["paragraph_count"] == 1
    assert kwargs["document_count"] == 0
    assert kwargs["docs"] == ["doc-1"]


def test_reprocessing_the_same_document_does_not_double_count():
    """The core regression: doc-1 was already recorded for this pair, so
    re-running upsert_cooccurrence with doc-1 again must not add to the
    counters a second time, even though the aggregated data looks
    identical to a fresh ingestion."""
    store, mock_session = _make_store_with_mock_session(
        {"evidence": ["import_statement"], "docs": ["doc-1"], "repos": []}
    )
    pair_agg = {
        ("Python", "FastAPI"): {
            "sentence_count": 3, "paragraph_count": 1, "document_count": 0,
            "evidence": {"import_statement"}, "sample_context": "...",
        }
    }

    store.upsert_cooccurrence("doc-1", "markdown", pair_agg)

    set_call = mock_session.run.call_args_list[2]
    kwargs = set_call.kwargs
    assert kwargs["sentence_count"] == 0
    assert kwargs["paragraph_count"] == 0
    assert kwargs["document_count"] == 0
    # supporting_documents stays correctly de-duplicated either way
    assert kwargs["docs"] == ["doc-1"]


def test_a_second_new_document_still_increments_normally():
    """A pair already supported by doc-1 should still accumulate counts
    normally when a genuinely new document (doc-2) is processed."""
    store, mock_session = _make_store_with_mock_session(
        {"evidence": ["import_statement"], "docs": ["doc-1"], "repos": []}
    )
    pair_agg = {
        ("Python", "FastAPI"): {
            "sentence_count": 2, "paragraph_count": 0, "document_count": 1,
            "evidence": set(), "sample_context": "",
        }
    }

    store.upsert_cooccurrence("doc-2", "github", pair_agg)

    set_call = mock_session.run.call_args_list[2]
    kwargs = set_call.kwargs
    assert kwargs["sentence_count"] == 2
    assert kwargs["document_count"] == 1
    assert sorted(kwargs["docs"]) == ["doc-1", "doc-2"]
    assert kwargs["repos"] == ["doc-2"]
