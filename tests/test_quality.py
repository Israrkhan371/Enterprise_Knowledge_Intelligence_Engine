"""
Tests for app/rag/quality.py: score_document(), score_document_quality(),
and score_all_documents(). detect_duplicates() is mocked throughout so
these don't touch a real DB or embedding model.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.rag import quality


def _doc(id_="d1", title="Doc", raw_text=None, updated_at=None, word_count=None):
    d = MagicMock()
    d.id = id_
    d.title = title
    if raw_text is not None:
        d.raw_text = raw_text
    else:
        d.raw_text = " ".join(["word"] * (word_count or 0))
    d.updated_at = updated_at
    return d


# ---------------------------------------------------------------------------
# _completeness_score
# ---------------------------------------------------------------------------

def test_completeness_zero_below_min_word_count():
    assert quality._completeness_score("short text here") == 0.0


def test_completeness_full_above_full_word_count():
    text = " ".join(["word"] * 500)
    assert quality._completeness_score(text) == 100.0


def test_completeness_scales_between_thresholds():
    text = " ".join(["word"] * 225)  # halfway between 50 and 400
    score = quality._completeness_score(text)
    assert 40 < score < 60


def test_completeness_handles_none():
    assert quality._completeness_score(None) == 0.0


# ---------------------------------------------------------------------------
# _freshness_score
# ---------------------------------------------------------------------------

def test_freshness_full_when_recently_updated():
    assert quality._freshness_score(datetime.utcnow() - timedelta(days=10)) == 100.0


def test_freshness_zero_when_very_stale():
    assert quality._freshness_score(datetime.utcnow() - timedelta(days=500)) == 0.0


def test_freshness_neutral_when_unknown():
    assert quality._freshness_score(None) == 50.0


def test_freshness_scales_between_thresholds():
    score = quality._freshness_score(datetime.utcnow() - timedelta(days=227))  # midpoint of 90-365
    assert 40 < score < 60


# ---------------------------------------------------------------------------
# _originality_score
# ---------------------------------------------------------------------------

def test_originality_full_when_no_duplicate_pairs():
    assert quality._originality_score("d1", []) == 100.0


def test_originality_penalized_by_duplicate_similarity():
    pairs = [{"document_a": "d1", "document_b": "d2", "similarity": 0.95}]
    score = quality._originality_score("d1", pairs)
    assert score == round(100 - 40 * 0.95, 1)


def test_originality_unaffected_by_unrelated_pairs():
    pairs = [{"document_a": "d3", "document_b": "d4", "similarity": 0.99}]
    assert quality._originality_score("d1", pairs) == 100.0


def test_originality_uses_worst_match_among_several():
    pairs = [
        {"document_a": "d1", "document_b": "d2", "similarity": 0.80},
        {"document_a": "d1", "document_b": "d3", "similarity": 0.93},
    ]
    score = quality._originality_score("d1", pairs)
    assert score == round(100 - 40 * 0.93, 1)


# ---------------------------------------------------------------------------
# score_document
# ---------------------------------------------------------------------------

def test_score_document_combines_all_three_components():
    document = _doc(word_count=500, updated_at=datetime.utcnow())
    breakdown = quality.score_document(document, [])

    assert breakdown["completeness_score"] == 100.0
    assert breakdown["freshness_score"] == 100.0
    assert breakdown["originality_score"] == 100.0
    assert breakdown["overall_score"] == 100.0
    assert breakdown["word_count"] == 500


def test_score_document_weighted_average_with_mixed_signals():
    document = _doc(id_="d1", word_count=500, updated_at=datetime.utcnow() - timedelta(days=500))
    breakdown = quality.score_document(document, [])
    # completeness=100 (0.5) + freshness=0 (0.3) + originality=100 (0.2)
    assert breakdown["overall_score"] == 70.0


# ---------------------------------------------------------------------------
# score_document_quality
# ---------------------------------------------------------------------------

def test_score_document_quality_returns_none_for_missing_document():
    db = MagicMock()
    db.get.return_value = None
    assert quality.score_document_quality(db, "missing") is None


def test_score_document_quality_persists_overall_score():
    document = _doc(id_="d1", word_count=500, updated_at=datetime.utcnow())
    db = MagicMock()
    db.get.return_value = document

    with patch("app.rag.quality.detect_duplicates", return_value=[]):
        breakdown = quality.score_document_quality(db, "d1")

    assert document.quality_score == breakdown["overall_score"] == 100.0
    assert db.commit.called


# ---------------------------------------------------------------------------
# score_all_documents
# ---------------------------------------------------------------------------

def test_score_all_documents_sorts_worst_first():
    good = _doc(id_="good", word_count=500, updated_at=datetime.utcnow())
    bad = _doc(id_="bad", word_count=10, updated_at=datetime.utcnow() - timedelta(days=500))
    db = MagicMock()
    db.query.return_value.all.return_value = [good, bad]

    with patch("app.rag.quality.detect_duplicates", return_value=[]):
        result = quality.score_all_documents(db)

    assert [r["document_id"] for r in result] == ["bad", "good"]
    assert good.quality_score == result[1]["overall_score"]
    assert bad.quality_score == result[0]["overall_score"]
    assert db.commit.called
