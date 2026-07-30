"""
Tests for the evidence-detection and text-splitting helpers in
app/graph/extract.py that back extract_cooccurrences(). These are pure
regex/string functions and deliberately don't touch get_nlp()/spaCy, so
they run without the en_core_web_sm model installed.
"""
from app.graph.extract import (
    EVIDENCE_PATTERNS,
    _detect_evidence,
    _pair_evidence_window,
    _pair_key,
    _split_paragraphs,
)


def test_detect_evidence_import_statement():
    assert "import_statement" in _detect_evidence("from fastapi import FastAPI")


def test_detect_evidence_package_file():
    assert "package_file" in _detect_evidence("Install with pip install fastapi, see requirements.txt")


def test_detect_evidence_dependency_keyword():
    assert "dependency_keyword" in _detect_evidence("FastAPI is built on Starlette and requires Python 3.8+")


def test_detect_evidence_deployment_reference():
    assert "deployment_reference" in _detect_evidence("The service is deployed to Kubernetes via a Dockerfile")


def test_detect_evidence_connection_reference():
    assert "connection_reference" in _detect_evidence("The app connects to PostgreSQL using a connection string")


def test_detect_evidence_returns_empty_for_plain_prose():
    assert _detect_evidence("Python and FastAPI are popular choices for APIs.") == set()


def test_detect_evidence_can_match_multiple_signals():
    text = "import fastapi; the service depends on PostgreSQL and is deployed to Kubernetes"
    evidence = _detect_evidence(text)
    assert {"import_statement", "dependency_keyword", "deployment_reference"} <= evidence


def test_pair_key_is_order_independent():
    assert _pair_key("Python", "FastAPI") == _pair_key("FastAPI", "Python")


def test_split_paragraphs_splits_on_blank_lines():
    text = "First paragraph about Python.\n\nSecond paragraph about FastAPI."
    paragraphs = _split_paragraphs(text)
    assert len(paragraphs) == 2
    assert "Python" in paragraphs[0]
    assert "FastAPI" in paragraphs[1]


def test_split_paragraphs_handles_single_block():
    paragraphs = _split_paragraphs("Just one paragraph, no blank lines.")
    assert len(paragraphs) == 1


def test_split_paragraphs_handles_empty_text():
    assert _split_paragraphs("") == []


def test_evidence_patterns_cover_expected_labels():
    expected = {"import_statement", "package_file", "deployment_reference",
                "connection_reference", "dependency_keyword"}
    assert set(EVIDENCE_PATTERNS.keys()) == expected


def test_pair_evidence_window_scopes_to_the_nearer_mention():
    text = "import fastapi at the top; Docker was also mentioned in passing here."
    window = _pair_evidence_window(text, "fastapi", "Docker")
    assert "import" in window
    assert "Docker" in window


def test_pair_evidence_window_does_not_leak_unrelated_evidence_to_other_pairs():
    """Regression test for the misattribution bug: an import statement near
    one entity must not get credited to a completely different pair whose
    own mentions sit far away from it in the same sentence/paragraph."""
    text = ("import fastapi as the framework entry point. " + ("filler text here. " * 12) +
            "Kubernetes runs the deployment while Docker builds the images.")
    kubernetes_docker_window = _pair_evidence_window(text, "Kubernetes", "Docker")
    assert "import_statement" not in _detect_evidence(kubernetes_docker_window)
    # sanity check: the import evidence is genuinely detectable near its own entity
    assert "import_statement" in _detect_evidence(text[:60])


def test_pair_evidence_window_picks_closest_occurrence_when_entity_repeats():
    text = "Python is popular. " + ("filler text. " * 20) + "Python depends on nothing, but FastAPI is built on Python here."
    window = _pair_evidence_window(text, "Python", "FastAPI")
    assert "built on" in window


def test_pair_evidence_window_falls_back_to_whole_chunk_if_entity_missing():
    assert _pair_evidence_window("no entities here", "Python", "FastAPI") == "no entities here"
