"""
Tests for app/ingestion/ocr.py.

Like test_loaders.py, this is self-contained: rather than depending on a
personal scanned PDF file, we generate a synthetic "scanned page" by
rendering text onto an image with Pillow and saving it directly as a PDF.
This produces a real image-only PDF with no text layer — the same shape
as an actual scanned document — so ocr_scanned_pdf() has to do real work
(render page -> image -> Tesseract) rather than reading an existing text
layer.

Note on font size: OCR accuracy depends on rendered text being large and
clear. A small/default font can produce minor misreads (e.g. "OCR" ->
"OC R") that are normal OCR behavior, not bugs — so the fixture uses a
large bold font, and assertions check for individual words rather than
exact whole-string equality, to avoid a flaky test over-fitting to one
Tesseract version's quirks.
"""
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ingestion.ocr import ocr_scanned_pdf, needs_ocr

OCR_WORDS = ["hello", "world", "fixture"]


def _load_font(size: int):
    # DejaVu Sans Bold ships with the fonts-dejavu-core package, already
    # present in the Docker image (pulled in as a system dependency of
    # other packages); fall back to PIL's built-in default if unavailable
    # so this test doesn't hard-fail in an environment without it.
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


@pytest.fixture
def synthetic_scanned_pdf(tmp_path):
    """
    Builds a single-page, image-only PDF containing large, clear text —
    simulating a real scanned document with no extractable text layer.
    """
    img = Image.new("RGB", (1200, 300), color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(48)
    draw.text((40, 100), " ".join(OCR_WORDS), fill="black", font=font)

    path = tmp_path / "scanned.pdf"
    img.save(str(path), "PDF")
    return str(path)


DARK_OCR_WORDS = ["dark", "background", "label"]


@pytest.fixture
def synthetic_dark_background_pdf(tmp_path):
    """
    Simulates a low-contrast, dark-background page (e.g. an architecture
    diagram or dark-themed slide) — a real, verified-poor-performing case
    for plain Tesseract before preprocessing was added. Light text on a
    dark navy background, the inverse of the usual scanned-document case.
    """
    img = Image.new("RGB", (1200, 300), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)
    font = _load_font(48)
    draw.text((40, 100), " ".join(DARK_OCR_WORDS), fill=(230, 230, 235), font=font)

    path = tmp_path / "dark_scan.pdf"
    img.save(str(path), "PDF")
    return str(path)


# --- needs_ocr() heuristic ------------------------------------------------

def test_needs_ocr_true_for_empty_text():
    assert needs_ocr("") is True


def test_needs_ocr_true_for_whitespace_only():
    assert needs_ocr("   \n\t  ") is True


def test_needs_ocr_true_for_short_garbage_text():
    # Simulates a PDF where native extraction returned a tiny fragment
    # (e.g. a page number or watermark) rather than real content.
    assert needs_ocr("12") is True


def test_needs_ocr_false_for_real_length_text():
    real_text = "word " * 50  # well over the default 40-char threshold
    assert needs_ocr(real_text) is False


def test_needs_ocr_respects_custom_threshold():
    assert needs_ocr("short", min_chars=3) is False
    assert needs_ocr("sh", min_chars=3) is True


# --- ocr_scanned_pdf() ----------------------------------------------------

def test_ocr_scanned_pdf_extracts_real_text(synthetic_scanned_pdf):
    text = ocr_scanned_pdf(synthetic_scanned_pdf)
    lowered = text.lower()
    for word in OCR_WORDS:
        assert word in lowered


def test_ocr_scanned_pdf_returns_nonempty_string(synthetic_scanned_pdf):
    text = ocr_scanned_pdf(synthetic_scanned_pdf)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_ocr_scanned_pdf_raises_on_missing_file():
    with pytest.raises(Exception):
        ocr_scanned_pdf("/nonexistent/path/scanned.pdf")


def test_ocr_scanned_pdf_handles_dark_background(synthetic_dark_background_pdf):
    """
    Regression test for a real, verified fix: plain Tesseract with no
    preprocessing performed poorly on a real dark-background architecture
    diagram (recovered only ~6 garbled words). Adding grayscale +
    autocontrast + binarization before OCR (see _preprocess_for_ocr in
    ocr.py) recovered nearly the full diagram text in that manual test.
    This synthetic dark-background fixture locks that fix in so it can't
    silently regress if the preprocessing step is ever changed or removed.
    """
    text = ocr_scanned_pdf(synthetic_dark_background_pdf)
    lowered = text.lower()
    for word in DARK_OCR_WORDS:
        assert word in lowered


# --- pipeline integration (needs_ocr feeding into ocr_scanned_pdf) --------

def test_needs_ocr_true_then_ocr_recovers_content(synthetic_scanned_pdf):
    """
    Simulates the real pipeline.py flow: native PDF text extraction on a
    scanned document returns almost nothing, needs_ocr() correctly flags
    it, and ocr_scanned_pdf() then recovers the actual content.
    """
    # A scanned PDF's "native" extraction would return ~nothing real —
    # simulated here directly rather than re-running load_pdf, since
    # load_pdf's own behavior is already covered in test_loaders.py.
    simulated_native_extraction = ""
    assert needs_ocr(simulated_native_extraction) is True

    recovered_text = ocr_scanned_pdf(synthetic_scanned_pdf)
    lowered = recovered_text.lower()
    for word in OCR_WORDS:
        assert word in lowered
