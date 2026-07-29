"""
Source-type loaders. Each function takes a file path (or URI) and returns
plain text ready for chunking. Swap in `unstructured` partitioners for
production-grade parsing; these wrappers keep the pipeline logic decoupled
from the parsing library so you can change libraries without touching
the rest of the ingestion flow.
"""
from pathlib import Path
import re

_TRIPLE_QUOTE_RE = re.compile(r'("""|\'\'\')(.*?)\1', re.DOTALL)
_BLOCK_COMMENT_RE = re.compile(r'/\*(.*?)\*/', re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'(?:^|\s)(?:#|//)\s?(.*)$', re.MULTILINE)


def extract_code_comments(text: str) -> str:
    """
    Pulls out only the human-language content from source code - docstrings
    and comments - for feeding into NER. Running entity extraction on raw
    code produces noise: class names, exception identifiers, and other
    CamelCase/code tokens get misread as proper nouns by general-purpose
    NER (e.g. "GenerationError", "IGNORECASE" tagged as entities from a
    real Python file). Comments and docstrings are where genuine
    human-written references to real people, companies, and technologies
    actually live in source code - everything else is syntax by
    definition and can never be a real entity.

    Deliberately does NOT affect what gets embedded/chunked for search -
    full source code stays fully searchable, since developers legitimately
    search by code content, not just comments. This only narrows what NER
    sees, not what semantic_search can retrieve.

    Best-effort across languages: handles Python triple-quoted docstrings,
    '#'/'//' line comments, and '/* */' block comments. Doesn't parse a
    real AST, so it can't be perfect (e.g. a '#' inside a string literal
    would be misread as a comment start) - acceptable tradeoff for what
    this narrows, since worst case is a little extra text reaching NER,
    not incorrect embeddings or broken ingestion.
    """
    parts = []

    for match in _TRIPLE_QUOTE_RE.finditer(text):
        parts.append(match.group(2))
    remaining = _TRIPLE_QUOTE_RE.sub(" ", text)

    for match in _BLOCK_COMMENT_RE.finditer(remaining):
        parts.append(match.group(1))
    remaining = _BLOCK_COMMENT_RE.sub(" ", remaining)

    for match in _LINE_COMMENT_RE.finditer(remaining):
        line = match.group(1).strip()
        if line:
            parts.append(line)

    return "\n".join(p.strip() for p in parts if p.strip())


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
    from unstructured.documents.elements import Title

    elements = partition_pdf(filename=path, strategy="hi_res")

    # Exclude Title elements from the text that feeds embeddings and NER.
    # Headings glued directly onto body text (e.g. "Parental Leave\nPrimary
    # caregivers receive...") get misread by spaCy as ORG entities, and
    # pollute chunk boundaries with short heading-only chunks.
    return "\n\n".join(str(e) for e in elements if not isinstance(e, Title))


def load_docx(path: str) -> str:
    from unstructured.partition.docx import partition_docx
    from unstructured.documents.elements import Title, Table

    elements = partition_docx(filename=path)

    parts = []
    for el in elements:
        if isinstance(el, (Title, Table)):
            continue
        text = str(el).strip()
        if not text:
            continue
        if text[-1] not in ".!?:":
            text += "."
        parts.append(text)

    filtered = "\n\n".join(parts)

    # Safety net: if filtering out Title/Table elements leaves nothing
    # (or almost nothing) — e.g. a short document where unstructured's
    # heuristic misclassifies the entire body as a Title — fall back to
    # the unfiltered text rather than silently losing all content.
    if len(filtered.strip()) < 20:
        return "\n\n".join(str(e) for e in elements if str(e).strip())

    return filtered


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


def load_api_docs(path: str) -> str:
    """
    Loads an OpenAPI/Swagger spec (JSON) and flattens it into readable
    text: API title/version/description, then each endpoint as
    "METHOD /path — summary" with its description and parameter/response
    summaries. A raw spec dump would bury the useful content in JSON
    punctuation and structural nesting, which hurts both chunking
    (arbitrary JSON line breaks) and semantic search (embeddings do
    better on prose than syntax) — this produces prose-like text instead
    so retrieval and citations work the same way they do for every other
    source type.
    """
    import json
    # utf-8-sig strips a leading BOM if present (e.g. files saved via
    # PowerShell's `Out-File -Encoding utf8`, which writes UTF-8 WITH a
    # BOM) and is a no-op otherwise. json.loads() has no tolerance for a
    # stray BOM character — it fails with "Expecting value: line 1
    # column 1" — so this loader needs the -sig variant even though the
    # other loaders in this module can get away with plain "utf-8".
    spec = json.loads(Path(path).read_text(encoding="utf-8-sig", errors="ignore"))

    lines: list[str] = []
    info = spec.get("info", {})
    if info.get("title"):
        lines.append(f"API: {info['title']}")
    if info.get("version"):
        lines.append(f"Version: {info['version']}")
    if info.get("description"):
        lines.append(info["description"])
    lines.append("")

    _HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head")

    for route, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue

            summary = operation.get("summary", "")
            header = f"{method.upper()} {route}"
            lines.append(f"{header} — {summary}" if summary else header)

            if operation.get("description"):
                lines.append(operation["description"])

            params = operation.get("parameters", [])
            param_bits = []
            for p in params:
                if not isinstance(p, dict):
                    continue
                name = p.get("name", "?")
                loc = p.get("in", "?")
                required = " (required)" if p.get("required") else ""
                param_bits.append(f"{name} [{loc}]{required}")
            if param_bits:
                lines.append("Parameters: " + ", ".join(param_bits))

            responses = operation.get("responses", {})
            resp_bits = []
            for code, resp in responses.items():
                if not isinstance(resp, dict):
                    continue
                desc = resp.get("description", "")
                resp_bits.append(f"{code}: {desc}" if desc else str(code))
            if resp_bits:
                lines.append("Responses: " + "; ".join(resp_bits))

            lines.append("")

    return "\n".join(lines)


_SQL_DATA_STATEMENT_RE = re.compile(r"^\s*(INSERT\s+INTO|COPY)\b", re.IGNORECASE)
_SQL_COPY_TERMINATOR_RE = re.compile(r"^\\\.\s*$")


def load_db_schema(path: str) -> str:
    """
    Loads a .sql dump and keeps only the structural statements (CREATE
    TABLE, ALTER TABLE, CREATE INDEX, COMMENT ON, plus any -- / block
    comments left as documentation) while dropping INSERT INTO rows and
    COPY ... FROM stdin data blocks. A schema dump commonly interleaves
    DDL with the actual table data (sometimes thousands of rows) —
    embedding raw data would bloat the vector store with row values
    that are never useful for a "what does this table look like" query,
    and could leak real records into the knowledge base. This keeps the
    shape of the schema, not its contents.
    """
    # utf-8-sig strips a leading BOM if present, no-op otherwise — see
    # load_api_docs() above for why this matters on Windows-authored files.
    raw_text = Path(path).read_text(encoding="utf-8-sig", errors="ignore")

    kept_lines: list[str] = []
    skipping_copy_block = False
    for line in raw_text.splitlines():
        if skipping_copy_block:
            if _SQL_COPY_TERMINATOR_RE.match(line):
                skipping_copy_block = False
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if _SQL_DATA_STATEMENT_RE.match(stripped):
            if stripped.upper().startswith("COPY") and "FROM STDIN" in stripped.upper():
                skipping_copy_block = True
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def load_lms(path: str) -> str:
    """
    Loads exported LMS course content. Handles two export shapes:
      - A SCORM package (.zip) containing one or more HTML content files
        (alongside imsmanifest.xml, which is ignored — it's packaging
        metadata, not course content) — every HTML file in the package
        is extracted and concatenated, each prefixed with its filename
        so a multi-lesson course doesn't collapse into one
        undifferentiated blob.
      - A single already-exported HTML file (some LMS platforms export
        a "print view" HTML directly instead of a SCORM zip).
    Both paths go through the same tag-stripping as load_blog() so the
    two loaders can't drift on how they clean HTML.
    """
    from bs4 import BeautifulSoup

    def _html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    if Path(path).suffix.lower() == ".zip":
        import zipfile
        sections: list[str] = []
        with zipfile.ZipFile(path) as archive:
            html_names = sorted(
                name for name in archive.namelist()
                if name.lower().endswith((".html", ".htm"))
            )
            for name in html_names:
                # utf-8-sig strips a leading BOM if present, no-op
                # otherwise — see load_api_docs() above for why this
                # matters on Windows-authored files.
                html = archive.read(name).decode("utf-8-sig", errors="ignore")
                text = _html_to_text(html)
                if text:
                    sections.append(f"# {name}\n\n{text}")
        return "\n\n".join(sections)

    html = Path(path).read_text(encoding="utf-8-sig", errors="ignore")
    return _html_to_text(html)


SOURCE_LOADERS = {
    "pdf": load_pdf,
    "docx": load_docx,
    "markdown": load_markdown,
    "code": load_code,
    "transcript": load_transcript,
    "meeting_notes": load_meeting_notes,
    "blog": load_blog,
    "api_docs": load_api_docs,
    "db_schema": load_db_schema,
    "lms": load_lms,
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
