"""
Tests for app/evaluation/eval.py.

Covers the fix for a real fixture-fragility bug: eval_set.json originally
had to be hand-filled with raw Document.id values, but Document.id is a
random uuid4 assigned at ingest time (gen_uuid() in app/core/models.py) —
unknowable ahead of ingestion, and different every time the corpus is
reseeded. resolve_relevant_ids() replaces that with a title -> id lookup
against the live documents table, done at evaluation time instead of
fixture-authoring time.
"""
from unittest.mock import MagicMock, patch

from app.evaluation.eval import resolve_relevant_ids, run_evaluation


def _mock_db_with_documents(rows):
    """rows: list[(id, title)] returned by the Document.id/Document.title query."""
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = rows
    return db


def test_resolve_relevant_ids_looks_up_titles():
    db = _mock_db_with_documents([
        ("doc-1", "Python Coding Standards"),
        ("doc-2", "Intern Onboarding Checklist"),
    ])
    item = {"relevant_document_titles": ["Python Coding Standards", "Intern Onboarding Checklist"]}

    resolved = resolve_relevant_ids(db, item)

    assert resolved == {"doc-1", "doc-2"}


def test_resolve_relevant_ids_merges_direct_ids_and_titles():
    db = _mock_db_with_documents([("doc-2", "Intern Onboarding Checklist")])
    item = {
        "relevant_document_ids": ["doc-1"],
        "relevant_document_titles": ["Intern Onboarding Checklist"],
    }

    resolved = resolve_relevant_ids(db, item)

    assert resolved == {"doc-1", "doc-2"}


def test_resolve_relevant_ids_missing_title_does_not_raise():
    db = _mock_db_with_documents([])  # nothing in the DB matches
    item = {"relevant_document_titles": ["Some Title Nobody Ingested Yet"]}

    resolved = resolve_relevant_ids(db, item)

    assert resolved == set()


@patch("app.evaluation.eval.mlflow")
@patch("app.evaluation.eval.hybrid_search")
@patch("app.evaluation.eval.load_eval_set")
def test_run_evaluation_skips_unresolvable_entries_instead_of_scoring_zero(
    mock_load_eval_set, mock_hybrid_search, mock_mlflow
):
    """
    An eval entry whose title doesn't match any ingested document must be
    skipped (and counted), not silently scored as a precision/recall-
    tanking 0 — that would hide a bad fixture entry inside what looks like
    a real retrieval-quality regression.
    """
    mock_load_eval_set.return_value = [
        {"query": "resolvable query", "relevant_document_titles": ["Real Doc"]},
        {"query": "typo'd query", "relevant_document_titles": ["Doc That Was Never Ingested"]},
    ]
    mock_hybrid_search.return_value = [{"document_id": "doc-1"}]

    db = MagicMock()
    # First resolve_relevant_ids() call (query 1) finds the doc; second (query 2) finds nothing.
    db.query.return_value.filter.return_value.all.side_effect = [
        [("doc-1", "Real Doc")],
        [],
    ]

    results = run_evaluation(db, k=10)

    assert results["num_queries"] == 1
    assert results["skipped"] == 1
    assert "error" not in results


@patch("app.evaluation.eval.mlflow")
@patch("app.evaluation.eval.load_eval_set")
def test_run_evaluation_returns_error_when_nothing_resolves(mock_load_eval_set, mock_mlflow):
    mock_load_eval_set.return_value = [
        {"query": "typo'd query", "relevant_document_titles": ["Nonexistent Doc"]},
    ]
    db = _mock_db_with_documents([])

    results = run_evaluation(db, k=10)

    assert "error" in results
    assert results["skipped"] == 1
