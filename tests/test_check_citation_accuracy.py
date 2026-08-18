"""
Tests for scripts/check_citation_accuracy.py.

generate_answer(), verify_citations(), and mlflow are all mocked — this
script's whole job is orchestration (call generate_answer for every eval
query, run verify_citations on each, aggregate), and that logic is exactly
what's under test here, not the live Gemini/embedding calls underneath it
(those are already covered by tests/test_generate.py and
tests/test_citation_check.py).
"""
import json
from unittest.mock import MagicMock, patch

from scripts.check_citation_accuracy import run


@patch("scripts.check_citation_accuracy.mlflow")
@patch("scripts.check_citation_accuracy.verify_citations")
@patch("scripts.check_citation_accuracy.generate_answer")
@patch("scripts.check_citation_accuracy.load_eval_set")
@patch("scripts.check_citation_accuracy.SessionLocal")
def test_run_computes_accuracy_only_over_cited_answers(
    mock_session_local, mock_load_eval_set, mock_generate_answer, mock_verify_citations, mock_mlflow, tmp_path, monkeypatch
):
    # Redirect the results file into tmp_path so this test doesn't write
    # into the real repo's docs/ directory.
    monkeypatch.setattr("scripts.check_citation_accuracy.RESULTS_PATH", tmp_path / "citation_accuracy_results.json")

    mock_load_eval_set.return_value = [
        {"query": "cited and verified"},
        {"query": "cited but flagged"},
        {"query": "answer with no citations at all"},
    ]
    mock_session_local.return_value = MagicMock()

    mock_generate_answer.side_effect = [
        {"answer": "Fact. [1]", "sources": [{"index": 1, "text": "..."}]},
        {"answer": "Claim. [1]", "sources": [{"index": 1, "text": "..."}]},
        {"answer": "No citation here.", "sources": [{"index": 1, "text": "..."}]},
    ]
    mock_verify_citations.side_effect = [
        {"cited_sources": [1], "flags": [], "verified": True},
        {"cited_sources": [1], "flags": [{"sentence": "Claim.", "issue": "low similarity (0.10) to source [1]"}], "verified": False},
        {"cited_sources": [], "flags": [], "verified": True},
    ]

    summary = run(k=6)

    assert summary["num_queries"] == 3
    assert summary["num_scored"] == 3
    assert summary["num_answers_with_no_citations"] == 1
    assert summary["num_answers_with_citations"] == 2
    # 1 of 2 cited answers fully checked out -> 0.5, not 2/3 -- the uncited
    # answer must not inflate the rate just because verify_citations()
    # defaults an uncited answer to verified=True.
    assert summary["citation_accuracy_rate"] == 0.5
    assert summary["flag_breakdown"] == {"claim_not_supported_by_cited_source": 1}


@patch("scripts.check_citation_accuracy.mlflow")
@patch("scripts.check_citation_accuracy.verify_citations")
@patch("scripts.check_citation_accuracy.generate_answer")
@patch("scripts.check_citation_accuracy.load_eval_set")
@patch("scripts.check_citation_accuracy.SessionLocal")
def test_run_handles_generation_failure_without_aborting(
    mock_session_local, mock_load_eval_set, mock_generate_answer, mock_verify_citations, mock_mlflow, tmp_path, monkeypatch
):
    monkeypatch.setattr("scripts.check_citation_accuracy.RESULTS_PATH", tmp_path / "citation_accuracy_results.json")

    mock_load_eval_set.return_value = [{"query": "times out"}, {"query": "succeeds"}]
    mock_session_local.return_value = MagicMock()
    mock_generate_answer.side_effect = [TimeoutError(), {"answer": "Fine. [1]", "sources": [{"index": 1, "text": "..."}]}]
    mock_verify_citations.return_value = {"cited_sources": [1], "flags": [], "verified": True}

    summary = run(k=6)

    assert summary["num_scored"] == 1
    assert summary["num_errored"] == 1
    assert summary["citation_accuracy_rate"] == 1.0


@patch("scripts.check_citation_accuracy.mlflow")
@patch("scripts.check_citation_accuracy.verify_citations")
@patch("scripts.check_citation_accuracy.generate_answer")
@patch("scripts.check_citation_accuracy.load_eval_set")
@patch("scripts.check_citation_accuracy.SessionLocal")
def test_run_writes_full_breakdown_to_results_file(
    mock_session_local, mock_load_eval_set, mock_generate_answer, mock_verify_citations, mock_mlflow, tmp_path, monkeypatch
):
    results_path = tmp_path / "citation_accuracy_results.json"
    monkeypatch.setattr("scripts.check_citation_accuracy.RESULTS_PATH", results_path)

    mock_load_eval_set.return_value = [{"query": "q1"}]
    mock_session_local.return_value = MagicMock()
    mock_generate_answer.return_value = {"answer": "Fact. [1]", "sources": [{"index": 1, "text": "..."}]}
    mock_verify_citations.return_value = {"cited_sources": [1], "flags": [], "verified": True}

    run(k=6)

    written = json.loads(results_path.read_text())
    assert written["summary"]["num_queries"] == 1
    assert written["per_query"][0]["query"] == "q1"
    assert written["per_query"][0]["verified"] is True


def test_run_returns_error_when_no_eval_set():
    with patch("scripts.check_citation_accuracy.load_eval_set", return_value=[]):
        result = run()
    assert "error" in result
