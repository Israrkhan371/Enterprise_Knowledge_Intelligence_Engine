"""
Source-type loaders. Each function takes a file path (or URI) and returns
plain text ready for chunking. Swap in `unstructured` partitioners for
production-grade parsing; these wrappers keep the pipeline logic decoupled
from the parsing library so you can change libraries without touching
the rest of the ingestion flow.
"""
from pathlib import Path

# --- Compatibility shim -----------------------------------------------
# pdfminer.six >= 20260107 renamed the re-export in pdfminer.pdfparser
# from PSSyntaxError to PDFSyntaxError (the class itself still lives in
# pdfminer.psparser, unchanged). unstructured==0.15.14's PDF partitioner
# still does `from pdfminer.pdfparser import PSSyntaxError` directly,
# which raises ImportError on the newer pdfminer.six. We can't downgrade
# pdfminer.six — pdfplumber (also required by unstructured[all-docs])
# hard-pins it to exactly 20260107 — so we restore the old alias instead.
# Must run before `unstructured.partition.pdf` is imported anywhere.
import pdfminer.pdfparser as _pdfparser
if not hasattr(_pdfparser, "PSSyntaxError"):
    from pdfminer.psparser import PSSyntaxError as _PSSyntaxError
    _pdfparser.PSSyntaxError = _PSSyntaxError
# ------------------------------------------------------------------------


def load_pdf(path: str) -> str:
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(filename=path, strategy="hi_res")
    return "\n".join(str(e) for e in elements)


def load_docx(path: str) -> str:
    from unstructured.partition.docx import partition_docx
    elements = partition_docx(filename=path)
    return "\n".join(str(e) for e in elements)


def load_markdown(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_code(path: str) -> str:
    """
    Loads a single local source-code file as plain text, prefixed with its
    filename so the language/purpose survives chunking (e.g. a chunk that
    starts mid-function still carries "# File: auth/login.py" as context).
    Deliberately simple — code doesn't need unstructured's layout detection,
    just the raw text with a small amount of provenance.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return f"# File: {Path(path).name}\n\n{text}"


def load_transcript(path: str) -> str:
    # Meeting notes / recorded session transcripts are plain text or .vtt/.srt
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_github_repo(repo_url: str, github_token: str | None = None) -> list[dict]:
    """
    Pulls README, docs/, and top-level source files from a GitHub repo.
    Returns a list of {path, text} so each file becomes its own Document row.

    NOTE: returns list[dict], not str like every other loader in this module
    — a repo is inherently many documents, not one. Use ingest_github_repo()
    in app/ingestion/pipeline.py to fan these out into separate Document
    rows; do not call this via load_by_source_type() / ingest_document(),
    which assume a single str of text per source.
    """
    import httpx

    headers = {"Authorization": f"token {github_token}"} if github_token else {}
    owner_repo = repo_url.rstrip("/").split("github.com/")[-1]
    api_base = f"https://api.github.com/repos/{owner_repo}/contents"

    results = []
    with httpx.Client(headers=headers, timeout=30) as client:
        resp = client.get(api_base)
        resp.raise_for_status()
        for item in resp.json():
            if item["type"] == "file" and item["name"].lower().endswith((".md", ".py", ".rst", ".txt")):
                file_resp = client.get(item["download_url"])
                results.append({"path": item["path"], "text": file_resp.text})
    return results


SOURCE_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "markdown": load_markdown,
    "code": load_code,
    "transcript": load_transcript,
}
# github is intentionally excluded from SOURCE_LOADERS: load_github_repo()
# returns list[dict], not str, so it can't go through load_by_source_type()
# / ingest_document() without breaking chunk_text(). Use
# ingest_github_repo() in pipeline.py instead.


def load_by_source_type(source_type: str, path: str) -> str:
    if source_type == "github":
        raise ValueError(
            "source_type='github' returns multiple documents, not one — "
            "call ingest_github_repo() in app/ingestion/pipeline.py instead "
            "of load_by_source_type()/ingest_document()."
        )
    loader = SOURCE_LOADERS.get(source_type)
    if not loader:
        raise ValueError(f"No loader registered for source_type={source_type}")
    return loader(path)
