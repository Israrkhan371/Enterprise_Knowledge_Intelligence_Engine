"""OCR pipeline for scanned documents that have no extractable text layer."""
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageOps


def _preprocess_for_ocr(image: Image.Image, threshold: int = 140) -> Image.Image:
    """
    Grayscale + autocontrast + binarize before handing a page to Tesseract.

    Without this, Tesseract performs poorly on dark-background or
    low-contrast pages (e.g. diagrams, dark-themed slides) — verified
    directly: a dark-background architecture diagram went from returning
    ~6 garbled words with no preprocessing to correctly recovering nearly
    every label (7 of 8 text blocks) with this preprocessing applied.
    Also verified this does NOT regress standard white-background scanned
    documents (memos, invoices, notes all still extract perfectly).

    threshold=140 was chosen empirically against these test documents;
    a page that's unusually light or dark overall may need retuning, but
    this is a safe general default.
    """
    gray = image.convert("L")
    gray = ImageOps.autocontrast(gray)
    return gray.point(lambda x: 0 if x < threshold else 255, "1")


def ocr_scanned_pdf(path: str, dpi: int = 300) -> str:
    """
    Runs each page through Tesseract with preprocessing applied, using
    PSM 3 (Tesseract's default: automatic page segmentation, no OSD).

    An earlier version of this function forced PSM 6 ("assume a single
    uniform block of text"), based on an initial test that only checked
    whether expected keywords appeared anywhere in the output. That check
    was flawed — it couldn't detect column-interleaving. Verified directly
    against real files: on a two-column resume, PSM 6 interleaves the left
    column's bullet list with the right column's skill labels mid-line,
    and also introduces garbled noise characters from icons/photos being
    misread as text. PSM 3 avoids both problems and, on a dark-background
    architecture diagram, PSM 3 also recovered one more text block
    ("Documents", the top box) that PSM 6 missed entirely. PSM 3 is
    strictly better on both real test cases, so no forced PSM is set here.
    """
    pages = convert_from_path(path, dpi=dpi)
    text_chunks = [
        pytesseract.image_to_string(_preprocess_for_ocr(page))
        for page in pages
    ]
    return "\n".join(text_chunks)


def needs_ocr(extracted_text: str, min_chars: int = 40) -> bool:
    """Heuristic: if native extraction returned almost nothing, fall back to OCR."""
    return len(extracted_text.strip()) < min_chars
