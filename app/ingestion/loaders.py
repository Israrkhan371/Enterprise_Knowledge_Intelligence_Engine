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


import re

_VTT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}.*$"
)
_SRT_TIMESTAMP_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}.*$"
)
_SRT_CUE_NUMBER_RE = re.compile(r"^\d+$")
_VTT_HEADER_RE = re.compile(r"^WEBVTT.*$")


def _strip_transcript_markup(raw_text: str) -> str:
    """
    Strips WebVTT/SRT structural lines (the "WEBVTT" header, cue numbers,
    and "00:01:23.456 --> 00:01:26.789" timestamp lines) so only the
    spoken-word content remains for chunking/embedding. Safe to run on
    plain, un-timestamped transcript text too — if none of these patterns
    match, the text passes through unchanged (aside from blank-line
    collapsing), so this one function covers both real .vtt/.srt exports
    and plain-text meeting notes.
    """
    lines = raw_text.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _VTT_HEADER_RE.match(stripped):
            continue
        if _VTT_TIMESTAMP_RE.match(stripped) or _SRT_TIMESTAMP_RE.match(stripped):
            continue
        if _SRT_CUE_NUMBER_RE.match(stripped):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def load_transcript(path: str) -> str:
    """
    Loads a recorded-session transcript. Handles plain text, WebVTT (.vtt),
    and SubRip (.srt) exports — cue numbers and timestamp lines are
    stripped so the vector store only embeds spoken content, not "1" /
    "00:00:01.000 --> 00:00:04.000" noise that would otherwise pollute
    every chunk and dilute semantic search relevance.
    """
    raw_text = Path(path).read_text(encoding="utf-8", errors="ignore")
    suffix = Path(path).suffix.lower()
    if suffix in (".vtt", ".srt"):
        return _strip_transcript_markup(raw_text)
    return raw_text


def load_meeting_notes(path: str) -> str:
    """
    Loads meeting notes. Distinct from load_transcript because meeting
    notes are typically a human-written summary (markdown or plain text)
    rather than a machine-generated, timestamped recording — but if notes
    were exported alongside timestamps (e.g. from a note-taking bot), the
    same markup-stripping logic still applies safely.
    """
    raw_text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return _strip_transcript_markup(raw_text) if raw_text.strip() else raw_text


def load_blog(source: str) -> str:
    """
    Loads a blog/article. Accepts either a local HTML file path or an
    http(s) URL — fetches it if it's a URL, then strips tags down to
    plain text via BeautifulSoup (already installed as an `unstructured`
    dependency, so no new package needed). <script>/<style> content is
    dropped first since it's never article text and would otherwise show
    up as noise in the extracted text.
    """
    from bs4 import BeautifulSoup

    if source.startswith("http://") or source.startswith("https://"):
        import httpx
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(source)
            resp.raise_for_status()
            html = resp.text
    else:
        html = Path(source).read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


_GITHUB_EXCLUDED_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "dist", "build", ".pytest_cache", "site-packages", ".mypy_cache",
}
_GITHUB_FILE_EXTENSIONS = (".md", ".py", ".rst", ".txt")
_GITHUB_MAX_DEPTH = 6


def load_github_repo(repo_url: str, github_token: str | None = None) -> list[dict]:
    """
    Recursively pulls matching source/doc files from a GitHub repo, walking
    into subfolders (e.g. app/ingestion/loaders.py, not just root-level
    files like README.md). Returns a list of {path, text} so each file
    becomes its own Document row.

    Skips common non-knowledge directories (.git, node_modules, venv,
    __pycache__, build artifacts, etc.) and stops recursing past
    _GITHUB_MAX_DEPTH so a very deep or unexpectedly large repo can't
    cause runaway API calls.

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

    results: list[dict] = []

    def _walk(client: "httpx.Client", url: str, depth: int) -> None:
        if depth > _GITHUB_MAX_DEPTH:
            return
        resp = client.get(url)
        resp.raise_for_status()
        for item in resp.json():
            if item["type"] == "dir":
                if item["name"] in _GITHUB_EXCLUDED_DIRS:
                    continue
                _walk(client, item["url"], depth + 1)
            elif item["type"] == "file" and item["name"].lower().endswith(_GITHUB_FILE_EXTENSIONS):
                file_resp = client.get(item["download_url"])
                results.append({"path": item["path"], "text": file_resp.text})

    with httpx.Client(headers=headers, timeout=30) as client:
        _walk(client, api_base, depth=0)

    return results


SOURCE_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "markdown": load_markdown,
    "code": load_code,
    "transcript": load_transcript,
    "meeting_notes": load_meeting_notes,
    "blog": load_blog,
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
