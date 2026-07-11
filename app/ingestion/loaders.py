"""
Source-type loaders. Each function takes a file path (or URI) and returns
plain text ready for chunking. Swap in `unstructured` partitioners for
production-grade parsing; these wrappers keep the pipeline logic decoupled
from the parsing library so you can change libraries without touching
the rest of the ingestion flow.
"""
from pathlib import Path


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


def load_transcript(path: str) -> str:
    # Meeting notes / recorded session transcripts are plain text or .vtt/.srt
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_github_repo(repo_url: str, github_token: str | None = None) -> list[dict]:
    """
    Pulls README, docs/, and top-level source files from a GitHub repo.
    Returns a list of {path, text} so each file becomes its own Document row.
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
    "transcript": load_transcript,
}


def load_by_source_type(source_type: str, path: str) -> str:
    loader = SOURCE_LOADERS.get(source_type)
    if not loader:
        raise ValueError(f"No loader registered for source_type={source_type}")
    return loader(path)
