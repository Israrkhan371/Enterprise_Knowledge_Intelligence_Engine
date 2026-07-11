"""OCR pipeline for scanned documents that have no extractable text layer."""
import pytesseract
from pdf2image import convert_from_path


def ocr_scanned_pdf(path: str, dpi: int = 300) -> str:
    pages = convert_from_path(path, dpi=dpi)
    text_chunks = [pytesseract.image_to_string(page) for page in pages]
    return "\n".join(text_chunks)


def needs_ocr(extracted_text: str, min_chars: int = 40) -> bool:
    """Heuristic: if native extraction returned almost nothing, fall back to OCR."""
    return len(extracted_text.strip()) < min_chars
