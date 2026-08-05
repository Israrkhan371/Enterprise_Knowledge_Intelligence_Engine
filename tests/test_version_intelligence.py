"""
Tests for app/rag/version_intelligence.py: detect_version_candidates(),
link_version(), and get_version_history().

embed_texts() is mocked throughout (no live model calls). The DB is a
MagicMock with a small in-memory Document registry, following the same
mocking approach as tests/test_intelligence_detect_duplicates.py.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.rag import version_intelligence as vi


def _doc(id_, title="doc", raw_text="some text", status="approved", version=1, supersedes_id=None,
         created_at=None):
    d = MagicMock()
    d.id = id_
    d.title = title
    d.raw_text = raw_text
    d.status = status
    d.version = version
    d.supersedes_id = supersedes_id
    d.created_at = created_at
    return d


def _mock_db(docs_by_id, supersedes_lookup=None):
    """docs_by_id: dict id -> Document mock, used by db.get().
    supersedes_lookup: dict supersedes_id -> Document mock, used by
    db.query(Document).filter(Document.supersedes_id == X).first()."""
    db = MagicMock()
    db.get.side_effect = lambda model, id_: docs_by_id.get(id_)

    def query_side_effect(model):
        query = MagicMock()

        def filter_side_effect(*args, **kwargs):
            filtered = MagicMock()
            # .all() -> every doc except the one referenced in the
            # Document.id != document_id filter (approximated via closures below)
            filtered.all.return_value = list(docs_by_id.values())
            # .first() -> supersedes_lookup result, set per-test via closure
            filtered.first.return_value = None
            return filtered

        query.filter.side_effect = filter_side_effect
        return query

    db.query.side_effect = query_side_effect
    return db


# ---------------------------------------------------------------------------
# detect_version_candidates
# ---------------------------------------------------------------------------

def test_returns_empty_when_target_document_missing():
    db = MagicMock()
    db.get.return_value = None
    result = vi.detect_version_candidates(db, "missing-id")
    assert result == []


def test_flags_candidate_in_similarity_band():
    target = _doc("d1", raw_text="the quick brown fox")
    other = _doc("d2", title="fox v2", raw_text="the quick brown fox jumps")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None  # _linked_ids lookups
    db.query.return_value.filter.return_value.all.return_value = [other]

    with patch("app.rag.version_intelligence.embed_texts", return_value=[[1.0, 0.0], [0.8, 0.6]]):
        result = vi.detect_version_candidates(db, "d1", similarity_threshold=0.75, duplicate_ceiling=0.92)

    assert len(result) == 1
    assert result[0]["document_id"] == "d2"
    assert result[0]["similarity"] == 0.8


def test_excludes_similarity_at_or_above_duplicate_ceiling():
    target = _doc("d1", raw_text="text a")
    other = _doc("d2", raw_text="text a exact")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = [other]

    with patch("app.rag.version_intelligence.embed_texts", return_value=[[1.0, 0.0], [1.0, 0.0]]):
        result = vi.detect_version_candidates(db, "d1", similarity_threshold=0.75, duplicate_ceiling=0.92)

    assert result == []


def test_excludes_similarity_below_floor():
    target = _doc("d1", raw_text="text a")
    other = _doc("d2", raw_text="unrelated content")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = [other]

    with patch("app.rag.version_intelligence.embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]):
        result = vi.detect_version_candidates(db, "d1", similarity_threshold=0.75, duplicate_ceiling=0.92)

    assert result == []


def test_already_linked_document_is_excluded_from_candidates():
    """d1.supersedes_id == d2 -> already resolved, shouldn't be re-suggested."""
    target = _doc("d1", raw_text="text a", supersedes_id="d2")
    other = _doc("d2", raw_text="text a")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target, "d2": other}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = [other]

    with patch("app.rag.version_intelligence.embed_texts", return_value=[[1.0, 0.0], [0.9, 0.436]]):
        result = vi.detect_version_candidates(db, "d1")

    assert result == []


def test_embedding_failure_is_skipped_not_raised():
    target = _doc("d1", raw_text="text a")
    other = _doc("d2", raw_text="text b")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = [other]

    with patch("app.rag.version_intelligence.embed_texts", side_effect=RuntimeError("model down")):
        result = vi.detect_version_candidates(db, "d1")

    assert result == []


def test_candidates_sorted_by_similarity_descending():
    target = _doc("d1", raw_text="text a")
    other_low = _doc("d2", raw_text="text b")
    other_high = _doc("d3", raw_text="text c")
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": target}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.all.return_value = [other_low, other_high]

    # First embed_texts call -> (d1, d2) similarity ~0.78; second -> (d1, d3) ~0.85
    with patch(
        "app.rag.version_intelligence.embed_texts",
        side_effect=[[[1.0, 0.0], [0.78, 0.6258]], [[1.0, 0.0], [0.85, 0.5268]]],
    ):
        result = vi.detect_version_candidates(db, "d1", similarity_threshold=0.75, duplicate_ceiling=0.92)

    assert [c["document_id"] for c in result] == ["d3", "d2"]


# ---------------------------------------------------------------------------
# link_version
# ---------------------------------------------------------------------------

def test_link_version_rejects_self_reference():
    db = MagicMock()
    with pytest.raises(vi.VersionLinkError):
        vi.link_version(db, "d1", "d1")


def test_link_version_raises_when_document_missing():
    db = MagicMock()
    db.get.side_effect = lambda model, id_: None
    with pytest.raises(vi.VersionLinkError, match="not found"):
        vi.link_version(db, "d1", "d2")


def test_link_version_sets_supersedes_and_bumps_version():
    document = _doc("d2", version=1)
    supersedes = _doc("d1", version=1)
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": supersedes, "d2": document}.get(id_)

    result = vi.link_version(db, "d2", "d1")

    assert result.supersedes_id == "d1"
    assert result.version == 2
    assert supersedes.status == "stale"
    assert db.commit.called


def test_link_version_rejects_cycle():
    """d1 already supersedes d2 (d1.supersedes_id == d2). Linking d2 to now
    supersede d1 would close a loop."""
    d1 = _doc("d1", version=2, supersedes_id="d2")
    d2 = _doc("d2", version=1)
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": d1, "d2": d2}.get(id_)

    with pytest.raises(vi.VersionLinkError, match="cycle"):
        vi.link_version(db, "d2", "d1")


# ---------------------------------------------------------------------------
# get_version_history
# ---------------------------------------------------------------------------

def test_version_history_empty_when_missing():
    db = MagicMock()
    db.get.return_value = None
    assert vi.get_version_history(db, "missing") == []


def test_version_history_single_document_no_links():
    doc = _doc("d1", title="Only Version", version=1)
    db = MagicMock()
    db.get.side_effect = lambda model, id_: {"d1": doc}.get(id_)
    db.query.return_value.filter.return_value.first.return_value = None

    result = vi.get_version_history(db, "d1")

    assert len(result) == 1
    assert result[0]["document_id"] == "d1"
    assert result[0]["is_current"] is True


def test_version_history_orders_oldest_to_newest_and_marks_current():
    v1 = _doc("d1", title="Onboarding Guide", version=1)
    v2 = _doc("d2", title="Onboarding Guide", version=2, supersedes_id="d1")
    v3 = _doc("d3", title="Onboarding Guide", version=3, supersedes_id="d2")
    docs = {"d1": v1, "d2": v2, "d3": v3}

    db = MagicMock()
    db.get.side_effect = lambda model, id_: docs.get(id_)

    successor_by_supersedes = {"d1": v2, "d2": v3, "d3": None}

    def query_side_effect(model):
        q = MagicMock()

        def filter_side_effect(clause):
            f = MagicMock()
            # clause is Document.supersedes_id == <id>; recover the id via
            # the right-hand comparator SQLAlchemy attaches to the clause.
            target_id = clause.right.value
            f.first.return_value = successor_by_supersedes.get(target_id)
            return f

        q.filter.side_effect = filter_side_effect
        return q

    db.query.side_effect = query_side_effect

    # Query from the middle version - history should still span the full chain.
    result = vi.get_version_history(db, "d2")

    assert [r["document_id"] for r in result] == ["d1", "d2", "d3"]
    assert result[0]["is_current"] is False
    assert result[-1]["is_current"] is True
