import difflib
import logging
from datetime import datetime, timedelta

import numpy as np
from google import genai
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.models import Document
from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import get_collection
from app.ingestion.chunking import chunk_text
from app.rag.gemini_utils import call_with_timeout

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.google_api_key)

# Safety cap on returned diff size. A full unified diff over two large
# documents could be thousands of lines - capping keeps the API response
# bounded and avoids shipping megabytes of diff text over the wire.
_MAX_DIFF_LINES = 500

# Documents at or under this size are sent to the LLM in a single call, in
# full (no truncation). Longer documents go through the chunk-summarize-
# then-combine path below, so the summary covers the whole document rather
# than only its first N characters.
_MAX_INLINE_CHARS = 4000


def _fetch_stored_embeddings(embedding_ids: list[str]) -> dict[str, list[float]]:
    """
    Looks up already-computed embeddings from ChromaDB by id (document_chunks.
    embedding_id), so callers don't have to re-run the embedding model over
    chunks that were already embedded at ingestion time. Returns an empty
    dict (never raises) on a Chroma failure - callers fall back to
    re-embedding whatever wasn't found.
    """
    if not embedding_ids:
        return {}
    try:
        collection = get_collection()
        result = collection.get(ids=embedding_ids, include=["embeddings"])
    except Exception:
        logger.exception("Failed to fetch chunk embeddings from ChromaDB; falling back to re-embedding.")
        return {}

    ids = result.get("ids") or []
    embeddings = result.get("embeddings")
    if embeddings is None:
        embeddings = []
    return {id_: list(vec) for id_, vec in zip(ids, embeddings)}


def detect_duplicates(db: Session, similarity_threshold: float = 0.92) -> list[dict]:
    """
    Flags document pairs whose chunk embeddings are near-identical.

    Reuses each chunk's embedding from ChromaDB (via document_chunks.
    embedding_id) instead of re-embedding every chunk on every call. Only
    chunks with no stored embedding, or one Chroma couldn't return, fall
    back to a fresh embed_texts() call. If that fallback embedding call
    itself fails, those chunks are excluded from this run rather than
    raising - duplicate detection is a background quality signal, not a
    critical path, so a partial result beats a hard failure.

    Similarity is computed as a single vectorized matrix multiply (numpy)
    instead of a Python-level O(n^2) double loop, so this scales to a much
    larger chunk count. Results are aggregated to one row per document
    pair (the max similarity across all of that pair's matching chunks) -
    two documents sharing several near-identical chunks would otherwise
    produce a flood of repeat rows for the same two documents.
    """
    rows = db.execute(text("SELECT id, document_id, text, embedding_id FROM document_chunks")).fetchall()
    if len(rows) < 2:
        return []

    stored = _fetch_stored_embeddings([r.embedding_id for r in rows if r.embedding_id])
    vectors: list[list[float] | None] = [
        stored.get(r.embedding_id) if r.embedding_id else None for r in rows
    ]

    missing_idx = [i for i, v in enumerate(vectors) if v is None]
    if missing_idx:
        try:
            fresh = embed_texts([rows[i].text for i in missing_idx])
        except Exception:
            logger.exception(
                "Embedding failed while computing chunk embeddings for duplicate detection; "
                "affected chunks are excluded from this comparison."
            )
            fresh = None
        if fresh is not None:
            for i, vec in zip(missing_idx, fresh):
                vectors[i] = vec

    valid = [(row, vec) for row, vec in zip(rows, vectors) if vec is not None]
    if len(valid) < 2:
        return []

    valid_rows = [row for row, _ in valid]
    matrix = np.array([vec for _, vec in valid], dtype=np.float32)

    # One matrix multiply replaces the O(n^2) Python double loop: sim_matrix[i, j]
    # is the cosine similarity between chunk i and chunk j (vectors are already
    # normalized by embed_texts(), so a plain dot product is the cosine similarity).
    sim_matrix = matrix @ matrix.T

    # Only look at the upper triangle (j > i) so each chunk pair is considered once.
    i_idx, j_idx = np.triu_indices(len(valid_rows), k=1)
    sims = sim_matrix[i_idx, j_idx]

    above_threshold = np.where(sims >= similarity_threshold)[0]

    # Aggregate to one row per document pair (max similarity across all of that
    # pair's matching chunks), instead of one row per matching chunk pair - a
    # document with several near-identical chunks would otherwise flood the
    # report with repeat rows for the same two documents.
    pair_best: dict[tuple[str, str], float] = {}
    for k in above_threshold:
        row_i, row_j = valid_rows[i_idx[k]], valid_rows[j_idx[k]]
        if row_i.document_id == row_j.document_id:
            continue
        doc_a, doc_b = sorted((str(row_i.document_id), str(row_j.document_id)))
        sim = float(sims[k])
        if sim > pair_best.get((doc_a, doc_b), -1.0):
            pair_best[(doc_a, doc_b)] = sim

    duplicates = [
        {"document_a": doc_a, "document_b": doc_b, "similarity": round(sim, 3)}
        for (doc_a, doc_b), sim in pair_best.items()
    ]
    duplicates.sort(key=lambda d: d["similarity"], reverse=True)
    return duplicates


def detect_outdated(db: Session, staleness_days: int = 180) -> list[dict]:
    """Flags documents not updated within the staleness window."""
    cutoff = datetime.utcnow() - timedelta(days=staleness_days)
    rows = db.execute(
        text("SELECT id, title, updated_at FROM documents WHERE updated_at < :cutoff"),
        {"cutoff": cutoff},
    ).fetchall()
    return [{"document_id": str(r.id), "title": r.title, "last_updated": r.updated_at.isoformat()} for r in rows]


def _generate_content(prompt: str) -> str:
    """
    Single point of contact with the Gemini text-generation API. Wraps the
    call with a hard timeout (settings.gemini_timeout_seconds, default 30s,
    configurable via GEMINI_TIMEOUT_SECONDS) so a hung request can't block
    a request indefinitely. Raises TimeoutError on expiry; other exceptions
    propagate unchanged to the caller.
    """
    response = call_with_timeout(
        _client.models.generate_content,
        timeout_seconds=settings.gemini_timeout_seconds,
        model=settings.gemini_model,
        contents=prompt,
    )
    return response.text or ""


def _chunk_summarize(text_: str) -> list[str]:
    """Splits a document into chunks (existing ingestion chunking strategy)
    and summarizes each one, so long documents can be covered in full
    without a single oversized LLM call."""
    chunks = chunk_text(text_)
    return [summarize_document(chunk) for chunk in chunks]


def compare_documents(text_a: str, text_b: str) -> str:
    """
    LLM narrative comparison of two documents.

    Short documents (both <= _MAX_INLINE_CHARS) are compared directly in a
    single call with their full text - no truncation.

    Longer documents are split into chunks, each chunk is summarized
    individually, and a final call combines the two documents' chunk
    summaries into one narrative. This keeps the summary's coverage in
    sync with the full-document diff, instead of only reflecting the
    first few thousand characters.
    """
    if len(text_a) <= _MAX_INLINE_CHARS and len(text_b) <= _MAX_INLINE_CHARS:
        return _generate_content(
            f"Compare these two documents. Summarize key differences and overlaps.\n\n"
            f"Document A:\n{text_a}\n\nDocument B:\n{text_b}"
        )

    summaries_a = _chunk_summarize(text_a)
    summaries_b = _chunk_summarize(text_b)

    combined_prompt = (
        "Compare these two documents based on their section summaries below, "
        "which together cover each document in full. Summarize the key "
        "differences and overlaps across the full documents.\n\n"
        "Document A (section summaries):\n"
        + "\n".join(f"- {s}" for s in summaries_a)
        + "\n\nDocument B (section summaries):\n"
        + "\n".join(f"- {s}" for s in summaries_b)
    )
    return _generate_content(combined_prompt)


def _embedding_similarity(text_a: str, text_b: str) -> float | None:
    """
    Cosine similarity between two full-document embeddings. embed_texts()
    returns normalized vectors (see app/embeddings/embedder.py), so a plain
    dot product is the cosine similarity - same approach detect_duplicates()
    uses above. Returns None (rather than a misleading score) when either
    text is blank, since there's nothing meaningful to embed, or when the
    embedding call itself fails - similarity is a supplementary signal here,
    so a model/infra hiccup shouldn't take down the whole comparison.
    """
    if not text_a.strip() or not text_b.strip():
        return None

    try:
        vectors = embed_texts([text_a, text_b])
    except Exception:
        logger.exception("Embedding failed while computing document similarity; returning similarity=None.")
        return None

    similarity = sum(a * b for a, b in zip(vectors[0], vectors[1]))
    return round(similarity, 3)


def _diff_texts(text_a: str, text_b: str) -> list[str]:
    """
    Line-level unified diff between two documents' text. Truncated to
    _MAX_DIFF_LINES with a marker line if it runs longer, so one giant
    document pair can't blow up the response size.
    """
    diff_lines = list(difflib.unified_diff(
        text_a.splitlines(),
        text_b.splitlines(),
        fromfile="document_a",
        tofile="document_b",
        lineterm="",
    ))

    if len(diff_lines) > _MAX_DIFF_LINES:
        omitted = len(diff_lines) - _MAX_DIFF_LINES
        diff_lines = diff_lines[:_MAX_DIFF_LINES]
        diff_lines.append(f"... diff truncated, {omitted} more line(s) omitted ...")

    return diff_lines


def compare_documents_full(text_a: str, text_b: str) -> dict:
    """
    Combined document comparison: embedding similarity + line-level diff +
    LLM narrative summary.

    Edge cases handled:
    - Both documents blank: skips embedding/diff/LLM entirely (nothing
      meaningful to compare) and returns a clear explanatory summary.
    - LLM summary call times out (settings.gemini_timeout_seconds): the
      TimeoutError propagates to the caller (the API route turns this into
      a 504), since a timeout is distinct from other failures and worth
      surfacing rather than silently degrading.
    - LLM summary call fails for any other reason (e.g. Gemini API error):
      logged and degraded to a fallback message rather than failing the
      whole request, since similarity and diff are still valid and useful
      on their own.
    """
    if not text_a.strip() and not text_b.strip():
        logger.info("compare_documents_full called with two blank documents; skipping comparison.")
        return {
            "similarity": None,
            "diff": [],
            "summary": "Both documents are empty - nothing to compare.",
        }

    try:
        summary = compare_documents(text_a, text_b)
    except TimeoutError:
        # Let the route handle this as a 504 - a timeout is a distinct,
        # actionable failure mode from other LLM errors, so it shouldn't
        # be masked behind the generic fallback summary below.
        raise
    except Exception:
        logger.exception("LLM comparison summary failed; returning fallback summary text.")
        summary = "Summary unavailable: the comparison service failed to generate a response."

    return {
        "similarity": _embedding_similarity(text_a, text_b),
        "diff": _diff_texts(text_a, text_b),
        "summary": summary,
    }


def summarize_document(text_: str) -> str:
    """
    Single-call leaf summarizer: one LLM call, no length branching.

    This is deliberately "dumb" about length (just a safety-net slice) because
    it's reused by _chunk_summarize() to summarize individual chunks - if this
    function chunked-and-recursed on long input, a chunk that's itself over
    _MAX_INLINE_CHARS (chunk_text()'s default 800-word chunks run ~4-5k chars,
    just over the 4000-char inline threshold) would re-chunk into the same
    single chunk and recurse forever. summarize_document_full() below is the
    one place that decides whether to chunk; this function never decides that
    for itself.
    """
    return _generate_content(f"Summarize this document in 4-6 sentences:\n\n{text_[:6000]}")


def _combine_summaries(summaries: list[str]) -> str:
    """Merges per-chunk summaries (from _chunk_summarize) into one coherent
    whole-document summary, the same way compare_documents' combined_prompt
    merges two documents' chunk summaries into one comparison."""
    prompt = (
        "Combine these section summaries into a single coherent 4-6 sentence "
        "summary of the full document. Together, the section summaries below "
        "cover the document in full.\n\n"
        + "\n".join(f"- {s}" for s in summaries)
    )
    return _generate_content(prompt)


def summarize_document_full(text_: str) -> str:
    """
    Full-document summary, on-demand.

    Short documents (<= _MAX_INLINE_CHARS) are summarized directly in a single
    call with their full text - no truncation.

    Longer documents are split into chunks (existing ingestion chunking
    strategy), each chunk is summarized individually via summarize_document(),
    and a final call combines those chunk summaries into one document-level
    summary - so coverage isn't limited to the first few thousand characters
    the way the old truncate-to-6000-chars behavior was.

    This is the one place that decides whether to chunk (see the docstring on
    summarize_document() for why that decision doesn't live there instead).

    Edge cases handled, mirroring compare_documents_full():
    - Blank document: skipped entirely, no embedding/LLM call needed.
    - LLM call times out (settings.gemini_timeout_seconds): propagates so the
      API route can return a 504 - distinct from other failures.
    - LLM call fails for any other reason: logged and degraded to a fallback
      message rather than failing the whole request.
    """
    if not text_.strip():
        return "Document is empty - nothing to summarize."

    try:
        if len(text_) <= _MAX_INLINE_CHARS:
            return summarize_document(text_)
        summaries = _chunk_summarize(text_)
        return _combine_summaries(summaries)
    except TimeoutError:
        raise
    except Exception:
        logger.exception("LLM document summary failed; returning fallback summary text.")
        return "Summary unavailable: the summarization service failed to generate a response."


def detect_knowledge_gaps(db: Session, min_score_threshold: float = 0.3, min_occurrences: int = 3) -> list[dict]:
    """
    Bonus: Knowledge Gap Detection. Groups usage_logs by low retrieval_score
    to surface recurring queries the knowledge base can't answer well.
    """
    rows = db.execute(
        text("""
            SELECT query, COUNT(*) AS occurrences, AVG(retrieval_score) AS avg_score
            FROM usage_logs
            WHERE retrieval_score IS NOT NULL
            GROUP BY query
            HAVING AVG(retrieval_score) < :threshold AND COUNT(*) >= :min_occ
            ORDER BY occurrences DESC
        """),
        {"threshold": min_score_threshold, "min_occ": min_occurrences},
    ).fetchall()
    return [{"query": r.query, "occurrences": r.occurrences, "avg_score": round(r.avg_score, 3)} for r in rows]


def suggest_document_updates(db: Session, document_id: str, staleness_days: int = 180, top_k: int = 3) -> dict | None:
    """
    AI capability: Suggest document updates. Unlike detect_outdated() (a pure
    time-based staleness flag) or version_intelligence's detect_version_candidates()
    (structural version-linking suggestions), this generates a content-level
    suggestion of *what* might need updating, grounded in the corpus itself
    rather than the model's own unsourced knowledge:

    1. Find other documents that are both fresher (updated_at more recent)
       and topically related (semantic search seeded by this document's own
       text), so the LLM has something concrete and traceable to compare
       against instead of inventing gaps from memory.
    2. If none are found, return a plain "nothing newer to compare against"
       result rather than asking the LLM to speculate with no grounding.
    3. Otherwise ask the LLM to point out what the target document may be
       missing or have superseded, relative *only* to those related
       documents — same "cite what's actually in the corpus" discipline as
       app/rag/generate.py's answer citations.

    Returns None if document_id doesn't exist.
    """
    document = db.get(Document, document_id)
    if not document:
        return None

    is_outdated = False
    if document.updated_at:
        is_outdated = document.updated_at < datetime.utcnow() - timedelta(days=staleness_days)

    related: list[dict] = []
    seed_text = (document.raw_text or document.title or "").strip()
    if seed_text:
        # Local import: semantic_search lives in app.search, which itself
        # doesn't depend on app.rag — importing at module load would risk
        # a circular import if that ever changes, and this path is only
        # hit when suggest_document_updates() is actually called.
        from app.search.semantic import semantic_search

        hits = semantic_search(seed_text[:4000], top_k=top_k * 4)
        seen_ids = {document_id}
        for hit in hits:
            other_id = hit.get("document_id")
            if not other_id or other_id in seen_ids:
                continue
            seen_ids.add(other_id)
            other = db.get(Document, other_id)
            if not other or not other.updated_at or not document.updated_at:
                continue
            if other.updated_at <= document.updated_at:
                continue  # only documents genuinely fresher than this one
            related.append(other)
            if len(related) >= top_k:
                break

    if not related:
        return {
            "document_id": document_id,
            "title": document.title,
            "is_outdated": is_outdated,
            "related_documents": [],
            "suggested_updates": (
                "No fresher, topically related documents were found in the "
                "corpus to compare against, so no grounded update "
                "suggestions are available."
            ),
        }

    context = "\n\n".join(
        f"[Related document: {r.title} (last updated {r.updated_at.isoformat()})]\n"
        f"{(r.raw_text or '')[:2000]}"
        for r in related
    )
    prompt = (
        "You are reviewing an internal engineering document for staleness. "
        "Given the TARGET document and several RELATED documents that are "
        "more recently updated, list concisely what the target document may "
        "be missing, outdated on, or superseded on. Base every point only on "
        "what the related documents actually say - do not invent facts. If "
        "the related documents don't indicate anything the target is missing, "
        "say so plainly.\n\n"
        f"TARGET document: {document.title} (last updated "
        f"{document.updated_at.isoformat() if document.updated_at else 'unknown'})\n"
        f"{(document.raw_text or '')[:3000]}\n\n"
        f"RELATED documents:\n{context}"
    )

    try:
        suggestions = _generate_content(prompt)
    except TimeoutError:
        raise
    except Exception:
        logger.exception("LLM update-suggestion generation failed for document_id=%s.", document_id)
        suggestions = (
            "Update suggestions unavailable: the suggestion service failed to "
            "generate a response. Related documents are listed below for "
            "manual review."
        )

    return {
        "document_id": document_id,
        "title": document.title,
        "is_outdated": is_outdated,
        "related_documents": [
            {"document_id": r.id, "title": r.title, "updated_at": r.updated_at.isoformat()}
            for r in related
        ],
        "suggested_updates": suggestions,
    }
