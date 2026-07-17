"""
Tests for app/ingestion/loaders.py.

Deliberately self-contained: instead of depending on personal sample files
outside the repo (which won't exist on a fresh clone / CI runner / Engineer
B's machine), each PDF/docx fixture is generated on the fly with pypdf /
python-docx — both already in requirements-lock.txt as unstructured deps.
"""
import pytest
from pypdf import PdfWriter
from docx import Document as DocxDocument

from app.ingestion.loaders import (
    load_pdf,
    load_docx,
    load_markdown,
    load_code,
    load_transcript,
    load_github_repo,
    load_by_source_type,
    SOURCE_LOADERS,
)

EXPECTED_TEXT = "EKIE loader test fixture — hello from pytest."


@pytest.fixture
def sample_pdf_path(tmp_path):
    path = tmp_path / "sample.pdf"
    # Build a minimal valid PDF with real text via pypdf's low-level API
    # (avoids adding reportlab as a new dependency just for this fixture).
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=150)
    # pypdf's blank page has no text layer; unstructured's hi_res strategy
    # still runs its layout model over the (blank) page image and returns
    # an empty element list rather than erroring, so we assert on "ran
    # without raising and returned a str" here, and cover real extracted
    # text in test_load_pdf_on_real_sample_if_present below instead.
    writer.write(str(path))
    return str(path)


@pytest.fixture
def sample_docx_path(tmp_path):
    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_paragraph(EXPECTED_TEXT)
    doc.save(str(path))
    return str(path)


@pytest.fixture
def sample_markdown_path(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(f"# Heading\n\n{EXPECTED_TEXT}\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_code_path(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(f"# {EXPECTED_TEXT}\ndef f():\n    return 1\n", encoding="utf-8")
    return str(path)


# --- docx -------------------------------------------------------------

def test_load_docx_returns_expected_text(sample_docx_path):
    text = load_docx(sample_docx_path)
    assert EXPECTED_TEXT in text


def test_load_docx_raises_on_missing_file():
    with pytest.raises(Exception):
        load_docx("/nonexistent/path/sample.docx")


# --- pdf ----------------------------------------------------------------
# Generating a PDF with a real extractable text layer needs a PDF-writing
# library beyond pypdf's blank-page helper, so this suite focuses on "runs
# without raising and returns a str" for the synthetic fixture, plus an
# opt-in check against a real sample if one is dropped in tests/fixtures/
# (matches how you Level-1 tested this manually with sample.pdf).

def test_load_pdf_runs_without_raising(sample_pdf_path):
    text = load_pdf(sample_pdf_path)
    assert isinstance(text, str)


def test_load_pdf_raises_on_missing_file():
    with pytest.raises(Exception):
        load_pdf("/nonexistent/path/sample.pdf")


def test_load_pdf_on_real_sample_if_present():
    """
    Optional stronger check: drop a real PDF at tests/fixtures/sample.pdf
    (e.g. copy the one you Level-1 tested manually) and this asserts real
    text actually comes out, not just "didn't crash".
    """
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "sample.pdf"
    if not fixture.exists():
        pytest.skip("tests/fixtures/sample.pdf not present — skipping real-text check")
    text = load_pdf(str(fixture))
    assert len(text) > 0


# --- markdown -------------------------------------------------------------

def test_load_markdown_returns_expected_text(sample_markdown_path):
    text = load_markdown(sample_markdown_path)
    assert EXPECTED_TEXT in text
    assert "# Heading" in text


def test_load_markdown_raises_on_missing_file():
    # Path.read_text on a missing file raises FileNotFoundError, not a
    # silent empty string — confirms load_markdown doesn't swallow errors.
    with pytest.raises(FileNotFoundError):
        load_markdown("/nonexistent/path/sample.md")


# --- code -------------------------------------------------------------

def test_load_code_returns_expected_text(sample_code_path):
    text = load_code(sample_code_path)
    assert EXPECTED_TEXT in text
    assert "def f():" in text


def test_load_code_includes_filename_header(sample_code_path):
    text = load_code(sample_code_path)
    assert text.startswith("# File: sample.py")


# --- transcript -------------------------------------------------------

def test_load_transcript_returns_expected_text(tmp_path):
    path = tmp_path / "sample.vtt"
    path.write_text(EXPECTED_TEXT, encoding="utf-8")
    text = load_transcript(str(path))
    assert EXPECTED_TEXT in text


# --- github_repo (mocked — no real network call) -----------------------

def test_load_github_repo_returns_list_of_dicts(monkeypatch):
    """
    load_github_repo hits the real GitHub API, so this mocks httpx.Client
    entirely rather than making a live network call in tests. Confirms the
    documented return shape: list[{"path": ..., "text": ...}].
    """
    import httpx

    class FakeResponse:
        def __init__(self, json_data=None, text=""):
            self._json = json_data
            self.text = text

        def raise_for_status(self):
            pass

        def json(self):
            return self._json

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            if url.endswith("/contents"):
                return FakeResponse(json_data=[
                    {"type": "file", "name": "README.md", "path": "README.md",
                     "download_url": "https://example.com/README.md"},
                    {"type": "file", "name": "image.png", "path": "image.png",
                     "download_url": "https://example.com/image.png"},
                ])
            return FakeResponse(text=EXPECTED_TEXT)

    monkeypatch.setattr(httpx, "Client", FakeClient)

    results = load_github_repo("https://github.com/example/repo")

    # image.png should be filtered out — only .md/.py/.rst/.txt are pulled
    assert len(results) == 1
    assert results[0]["path"] == "README.md"
    assert results[0]["text"] == EXPECTED_TEXT


# --- dispatch table / load_by_source_type ------------------------------

def test_source_loaders_registry_has_expected_keys():
    assert set(SOURCE_LOADERS.keys()) == {"pdf", "docx", "markdown", "code", "transcript"}


def test_github_excluded_from_source_loaders():
    # Documented on purpose: load_github_repo returns list[dict], not str,
    # so it can't go through load_by_source_type()/ingest_document().
    assert "github" not in SOURCE_LOADERS


def test_load_by_source_type_github_raises_helpful_error():
    with pytest.raises(ValueError, match="ingest_github_repo"):
        load_by_source_type("github", "irrelevant-path")


def test_load_by_source_type_unknown_raises():
    with pytest.raises(ValueError, match="No loader registered"):
        load_by_source_type("carrier-pigeon", "irrelevant-path")


def test_load_by_source_type_dispatches_correctly(sample_markdown_path):
    text = load_by_source_type("markdown", sample_markdown_path)
    assert EXPECTED_TEXT in text
