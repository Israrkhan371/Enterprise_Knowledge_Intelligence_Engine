import logging

from sqlalchemy.orm import Session

from app.core.models import Document
from app.embeddings.embedder import embed_texts

logger = logging.getLogger(__name__)

# Below this, two documents aren't similar enough to plausibly be different
# versions of the same underlying content.
_CANDIDATE_SIMILARITY_FLOOR = 0.75

# At or above this, two documents are better explained as exact/near-exact
# duplicates of each other (app.rag.intelligence.detect_duplicates() owns
# that band, at the same default threshold) than as distinct versions -
# a real new version usually has *some* substantive edit, not none.
_DUPLICATE_CEILING = 0.92


class VersionLinkError(ValueError):
    """Raised by link_version() for an invalid version link: a missing
    document, a self-reference, or a cycle."""


def _pairwise_similarity(text_a: str, text_b: str) -> float | None:
    """
    Cosine similarity between two full-document embeddings (embed_texts()
    returns normalized vectors, so a plain dot product is the cosine
    similarity - same approach as app.rag.intelligence._embedding_similarity()).
    Returns None (rather than a misleading score) when either text is blank
    or the embedding call fails - this is a suggestion signal, not a
    critical path, so a hiccup on one pair shouldn't fail the whole scan.
    """
    if not text_a.strip() or not text_b.strip():
        return None
    try:
        vec_a, vec_b = embed_texts([text_a, text_b])
    except Exception:
        logger.exception("Embedding failed while computing version-candidate similarity.")
        return None
    return round(sum(a * b for a, b in zip(vec_a, vec_b)), 3)


def _linked_ids(db: Session, document: Document) -> set[str]:
    """IDs already directly linked to `document` in either direction (it
    supersedes one, or one supersedes it) - excluded from candidate
    suggestions since those pairs are already resolved, not merely proposed."""
    linked = set()
    if document.supersedes_id:
        linked.add(document.supersedes_id)
    successor = db.query(Document).filter(Document.supersedes_id == document.id).first()
    if successor:
        linked.add(successor.id)
    return linked


def detect_version_candidates(
    db: Session,
    document_id: str,
    similarity_threshold: float = _CANDIDATE_SIMILARITY_FLOOR,
    duplicate_ceiling: float = _DUPLICATE_CEILING,
) -> list[dict]:
    """
    Bonus: Document Version Intelligence. Suggests other documents that are
    plausibly an earlier/later version of `document_id` - close enough in
    content to be the same underlying document, but not so close they're
    better explained as an exact duplicate.

    This only suggests candidates for an admin to confirm via link_version();
    it never links anything itself. Already-linked documents (in either
    direction) are excluded, since those pairs are already resolved.
    Returns [] if `document_id` doesn't exist.
    """
    target = db.get(Document, document_id)
    if not target:
        return []

    exclude = _linked_ids(db, target)
    exclude.add(document_id)

    candidates = []
    for other in db.query(Document).filter(Document.id != document_id).all():
        if other.id in exclude:
            continue
        similarity = _pairwise_similarity(target.raw_text or "", other.raw_text or "")
        if similarity is None:
            continue
        if similarity_threshold <= similarity < duplicate_ceiling:
            candidates.append({
                "document_id": other.id,
                "title": other.title,
                "similarity": similarity,
                "status": other.status,
                "version": other.version,
            })

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates


def link_version(db: Session, document_id: str, supersedes_id: str) -> Document:
    """
    Records that `document_id` is a newer version of `supersedes_id`:
    - sets document.supersedes_id
    - sets document.version = supersedes_doc.version + 1
    - marks the superseded document status="stale" (confirmed out of date -
      distinct from app.rag.intelligence.detect_outdated()'s time-based
      staleness signal, this one reflects an admin-confirmed replacement)

    Raises VersionLinkError for a missing document, a self-reference, or a
    cycle (supersedes_id already sits downstream of document_id).
    """
    if document_id == supersedes_id:
        raise VersionLinkError("a document cannot supersede itself")

    document = db.get(Document, document_id)
    supersedes = db.get(Document, supersedes_id)
    if not document or not supersedes:
        missing = [
            doc_id for doc_id, doc in ((document_id, document), (supersedes_id, supersedes))
            if doc is None
        ]
        raise VersionLinkError(f"document(s) not found: {', '.join(missing)}")

    # Cycle guard: walk supersedes's own ancestor chain. If document_id
    # appears anywhere in it, linking would close a loop.
    seen = {supersedes_id}
    cursor = supersedes
    while cursor.supersedes_id:
        if cursor.supersedes_id == document_id:
            raise VersionLinkError("linking these documents would create a cycle")
        if cursor.supersedes_id in seen:
            break  # already-broken chain elsewhere; don't loop forever here
        seen.add(cursor.supersedes_id)
        cursor = db.get(Document, cursor.supersedes_id)
        if cursor is None:
            break

    document.supersedes_id = supersedes_id
    document.version = (supersedes.version or 1) + 1
    supersedes.status = "stale"

    db.add(document)
    db.add(supersedes)
    db.commit()
    db.refresh(document)
    return document


def get_version_history(db: Session, document_id: str) -> list[dict]:
    """
    Full version chain containing `document_id`, oldest first, regardless of
    where in the chain `document_id` itself sits. Walks backward via
    supersedes_id to the root, then forward from `document_id` to the newest
    version by repeatedly looking up whoever supersedes the current document.

    Returns [] if `document_id` doesn't exist.
    """
    doc = db.get(Document, document_id)
    if not doc:
        return []

    seen = {doc.id}

    ancestors = []
    cursor = doc
    while cursor.supersedes_id and cursor.supersedes_id not in seen:
        parent = db.get(Document, cursor.supersedes_id)
        if parent is None:
            break
        ancestors.append(parent)
        seen.add(parent.id)
        cursor = parent
    ancestors.reverse()  # oldest first

    successors = []
    cursor = doc
    while True:
        successor = db.query(Document).filter(Document.supersedes_id == cursor.id).first()
        if not successor or successor.id in seen:
            break
        successors.append(successor)
        seen.add(successor.id)
        cursor = successor

    chain = ancestors + [doc] + successors
    newest_id = chain[-1].id

    return [
        {
            "document_id": d.id,
            "title": d.title,
            "version": d.version,
            "status": d.status,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "is_current": d.id == newest_id,
        }
        for d in chain
    ]
