"""
Tests for extract_entities() in app/graph/extract.py.

get_nlp() (a real spaCy pipeline) is mocked out so these run without the
en_core_web_sm model installed — same reasoning as test_recommend_learning_path.py
mocking extract_entities() one layer up. Here we're mocking one layer lower:
get_nlp() returns a fake `nlp` callable that produces a fake spaCy Doc whose
.ents list we control directly, so we can drive extract_entities()'s own
filtering logic (stoplist, TECH canonicalization, dedup, and the filename
pattern) with realistic-shaped fake entity spans.

Regression coverage for a real bug found 2026-08-04: load_lms() and
load_code() both prefix file content with a "# {filename}" header (e.g.
"# lesson1.html" from a SCORM zip, "# File: utils.py" from load_code()),
and spaCy sometimes tags that bare filename as PRODUCT/ORG. A
_FILENAME_PATTERN regex was written to reject these, but never actually
referenced in extract_entities()'s filter chain -- defined, never wired.
"""
from unittest.mock import MagicMock, patch

from app.graph.extract import extract_entities


def _make_span(text, label, pos=None):
    span = MagicMock()
    span.text = text
    span.label_ = label
    span.__len__ = lambda self: len(text.split())
    if pos is not None:
        span.root.pos_ = pos
    return span


def _mock_nlp_returning(ents):
    fake_doc = MagicMock()
    fake_doc.ents = ents
    fake_nlp = MagicMock(return_value=fake_doc)
    return fake_nlp


def test_extract_entities_keeps_relevant_labels():
    ents = [_make_span("Acme Corp", "ORG")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == [{"text": "Acme Corp", "label": "ORG"}]


def test_extract_entities_drops_irrelevant_labels():
    ents = [_make_span("Tuesday", "DATE")]  # DATE is not in RELEVANT_LABELS
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == []


def test_extract_entities_filters_stoplist_terms():
    ents = [_make_span("CTO", "ORG"), _make_span("Marcus Webb", "PERSON")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == [{"text": "Marcus Webb", "label": "PERSON"}]


def test_extract_entities_canonicalizes_tech_terms_regardless_of_casing():
    ents = [_make_span("chromadb", "ORG")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == [{"text": "ChromaDB", "label": "TECH"}]


def test_extract_entities_dedupes_repeated_entities():
    ents = [_make_span("Python", "PRODUCT"), _make_span("Python", "PRODUCT")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert len(results) == 1


def test_extract_entities_rejects_scorm_html_filenames():
    """Regression test: load_lms()'s '# lesson1.html' header must not
    become a graph entity."""
    ents = [_make_span("lesson1.html", "PRODUCT")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == []


def test_extract_entities_rejects_scorm_nested_path_filenames():
    ents = [_make_span("SCORM_content/lesson2.htm", "ORG")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == []


def test_extract_entities_rejects_code_loader_filenames():
    """Regression test: load_code()'s '# File: utils.py' header must not
    become a graph entity either -- same class of bug, different loader."""
    ents = [_make_span("utils.py", "PRODUCT")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == []


def test_extract_entities_rejects_scorm_and_imsmanifest_stoplist_terms():
    ents = [_make_span("SCORM", "ORG"), _make_span("imsmanifest", "PRODUCT")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == []


def test_extract_entities_does_not_reject_real_entities_that_contain_dots():
    """The filename filter must be specific to loader-header-shaped noise,
    not so broad it eats real entities like 'Node.js' (a real TECH_TERMS
    gazetteer entry) just because they contain a dot."""
    ents = [_make_span("Node.js", "PRODUCT")]
    with patch("app.graph.extract.get_nlp", return_value=_mock_nlp_returning(ents)):
        results = extract_entities("some text")

    assert results == [{"text": "Node.js", "label": "TECH"}]