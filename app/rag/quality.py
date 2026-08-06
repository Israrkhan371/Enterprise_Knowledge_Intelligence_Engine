from datetime import datetime

from sqlalchemy.orm import Session

from app.core.models import Document
from app.rag.intelligence import detect_duplicates

# --- Completeness: word count -> 0-100 -------------------------------------
# Below this, a document is too thin to be useful (a stub, a placeholder).
_MIN_WORD_COUNT = 50
# At/above this, length stops helping - a long document isn't automatically
# a better one, it just no longer looks incomplete.
_FULL_WORD_COUNT = 400

# --- Freshness: days since last update -> 0-100 -----------------------------
# Mirrors app.rag.intelligence.detect_outdated()'s spirit but as a continuous
# score rather than a binary flag, and with its own (shorter) full-credit
# window - a doc doesn't need to sit untouched for the full detect_outdated()
# staleness_days (default 180) before freshness starts to taper.
_FRESH_DAYS = 90
_STALE_DAYS = 365

# --- Originality: near-duplicate penalty ------------------------------------
# Max points deducted at similarity=1.0 (an exact duplicate). Scaled linearly
# by the document's highest similarity to any other document.
_DUPLICATE_MAX_PENALTY = 40

_WEIGHTS = {"completeness": 0.5, "freshness": 0.3, "originality": 0.2}


def _completeness_score(raw_text: str | None) -> float:
    word_count = len((raw_text or "").split())
    if word_count <= _MIN_WORD_COUNT:
        return 0.0
    if word_count >= _FULL_WORD_COUNT:
        return 100.0
    return round(100 * (word_count - _MIN_WORD_COUNT) / (_FULL_WORD_COUNT - _MIN_WORD_COUNT), 1)


def _freshness_score(updated_at: datetime | None) -> float:
    if updated_at is None:
        return 50.0  # unknown age isn't evidence of staleness - stay neutral
    age_days = (datetime.utcnow() - updated_at).total_seconds() / 86400
    if age_days <= _FRESH_DAYS:
        return 100.0
    if age_days >= _STALE_DAYS:
        return 0.0
    return round(100 * (1 - (age_days - _FRESH_DAYS) / (_STALE_DAYS - _FRESH_DAYS)), 1)


def _originality_score(document_id: str, duplicate_pairs: list[dict]) -> float:
    """100 minus a penalty scaled by the highest similarity this document
    shares with any other document (from detect_duplicates()). A document
    that's a near-exact duplicate of another contributes little unique
    knowledge to the corpus, whatever its own length or freshness look like."""
    max_similarity = 0.0
    for pair in duplicate_pairs:
        if document_id in (pair["document_a"], pair["document_b"]):
            max_similarity = max(max_similarity, pair["similarity"])
    if max_similarity == 0.0:
        return 100.0
    return round(100 - _DUPLICATE_MAX_PENALTY * max_similarity, 1)


def score_document(document: Document, duplicate_pairs: list[dict]) -> dict:
    """
    Deterministic quality breakdown for one document: completeness (length),
    freshness (recency of updated_at), and originality (inverse of its
    worst near-duplicate match). No LLM call - cheap enough to run over the
    whole corpus on demand, and reuses signals the platform already computes
    elsewhere (detect_duplicates) rather than re-deriving them.
    """
    completeness = _completeness_score(document.raw_text)
    freshness = _freshness_score(document.updated_at)
    originality = _originality_score(document.id, duplicate_pairs)

    overall = round(
        _WEIGHTS["completeness"] * completeness
        + _WEIGHTS["freshness"] * freshness
        + _WEIGHTS["originality"] * originality,
        1,
    )
    return {
        "document_id": document.id,
        "title": document.title,
        "overall_score": overall,
        "completeness_score": completeness,
        "freshness_score": freshness,
        "originality_score": originality,
        "word_count": len((document.raw_text or "").split()),
    }


def score_document_quality(db: Session, document_id: str) -> dict | None:
    """Scores one document and persists the result to Document.quality_score.
    Returns None if the document doesn't exist."""
    document = db.get(Document, document_id)
    if not document:
        return None

    duplicate_pairs = detect_duplicates(db)
    breakdown = score_document(document, duplicate_pairs)

    document.quality_score = breakdown["overall_score"]
    db.add(document)
    db.commit()
    return breakdown


def score_all_documents(db: Session) -> list[dict]:
    """
    Scores every document in the corpus, persists each Document.quality_score,
    and returns the breakdowns worst-first (lowest overall_score first) so
    the documents needing attention surface at the top of the admin view.
    """
    duplicate_pairs = detect_duplicates(db)
    documents = db.query(Document).all()

    breakdowns = []
    for document in documents:
        breakdown = score_document(document, duplicate_pairs)
        document.quality_score = breakdown["overall_score"]
        db.add(document)
        breakdowns.append(breakdown)

    db.commit()
    breakdowns.sort(key=lambda b: b["overall_score"])
    return breakdowns
