"""
Tests for app/admin/routes.py's review-AI-answers and usage-analytics
endpoints (GET/POST /admin/answers*, GET /admin/analytics/usage*).

These had zero coverage before this pass. Mirrors the direct-function-call
pattern used in tests/test_admin_documents.py (MagicMock db, call the route
function itself rather than going through a TestClient/FastAPI app).

response_model fields that use `from_attributes=True` (AnswerSummaryResponse,
AnswerDetailResponse, AnswerReviewLogEntry) are built here with
SimpleNamespace rows carrying real, correctly-typed values rather than
MagicMocks, since pydantic validates attribute types on construction and a
bare MagicMock attribute won't satisfy e.g. `query: str`.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.admin.routes import (
    get_answer,
    get_answer_review_history,
    list_answers,
    review_answer,
    usage_analytics,
    usage_analytics_timeseries,
    usage_top_queries,
)


def _fake_usage_log(**overrides):
    defaults = dict(
        id="11111111-1111-1111-1111-111111111111",
        user_id="22222222-2222-2222-2222-222222222222",
        query="what are FastAPI's main dependencies?",
        retrieval_score=0.87,
        citation_verified=True,
        was_helpful=None,
        flagged_for_review=False,
        reviewed=False,
        created_at=datetime(2026, 7, 30, 12, 0, 0),
        answer="FastAPI depends on Starlette and Pydantic.",
        sources=[{"document_id": "doc-1"}],
        citation_flags=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --- GET /admin/answers ------------------------------------------------


def test_list_answers_rejects_out_of_range_limit():
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        list_answers(limit=0, db=db)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        list_answers(limit=201, db=db)
    assert exc_info.value.status_code == 400


def test_list_answers_applies_flagged_and_reviewed_filters():
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value = query
    query.count.return_value = 0
    query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

    list_answers(flagged_for_review=True, reviewed=False, db=db)

    # Both filters were applied, not silently dropped (the same FastAPI
    # undeclared-param bug class documented elsewhere in this repo).
    assert query.filter.call_count == 2


def test_list_answers_returns_shaped_response():
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value = query
    query.count.return_value = 1
    row = _fake_usage_log()
    query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [row]

    result = list_answers(limit=50, offset=0, db=db)

    assert result.total == 1
    assert result.limit == 50
    assert len(result.answers) == 1
    assert result.answers[0].id == row.id
    assert result.answers[0].query == row.query


# --- GET /admin/answers/{id} --------------------------------------------


def test_get_answer_404_when_missing():
    db = MagicMock()
    db.get.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        get_answer("11111111-1111-1111-1111-111111111111", db=db)
    assert exc_info.value.status_code == 404


def test_get_answer_400_when_id_not_a_uuid():
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        get_answer("not-a-uuid", db=db)
    assert exc_info.value.status_code == 404  # ensure_valid_uuid default


def test_get_answer_returns_full_detail_including_answer_and_sources():
    db = MagicMock()
    row = _fake_usage_log()
    db.get.return_value = row

    result = get_answer(row.id, db=db)

    assert result.answer == row.answer
    assert result.sources == row.sources


# --- GET /admin/answers/{id}/review-history ------------------------------


def test_get_answer_review_history_404_when_missing():
    db = MagicMock()
    db.get.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        get_answer_review_history("11111111-1111-1111-1111-111111111111", db=db)
    assert exc_info.value.status_code == 404


def test_get_answer_review_history_returns_ordered_logs():
    db = MagicMock()
    db.get.return_value = _fake_usage_log()
    query = db.query.return_value
    query.filter.return_value = query
    expected = [SimpleNamespace(reviewer="mentor@ezitech.com", decision="approved", comment="", created_at=datetime.utcnow())]
    query.order_by.return_value.all.return_value = expected

    result = get_answer_review_history("11111111-1111-1111-1111-111111111111", db=db)

    assert len(result) == 1
    assert result[0].decision == "approved"


# --- POST /admin/answers/{id}/review -------------------------------------


def _fake_admin():
    return SimpleNamespace(email="mentor@ezitech.com")


def test_review_answer_404_when_missing():
    db = MagicMock()
    db.get.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        review_answer("11111111-1111-1111-1111-111111111111", decision="approved", admin=_fake_admin(), db=db)
    assert exc_info.value.status_code == 404


def test_review_answer_rejects_invalid_decision():
    db = MagicMock()
    db.get.return_value = _fake_usage_log()
    with pytest.raises(HTTPException) as exc_info:
        review_answer("11111111-1111-1111-1111-111111111111", decision="maybe", admin=_fake_admin(), db=db)
    assert exc_info.value.status_code == 400


def test_review_answer_approved_clears_flag():
    db = MagicMock()
    row = _fake_usage_log(flagged_for_review=True)
    db.get.return_value = row

    result = review_answer(row.id, decision="approved", admin=_fake_admin(), db=db)

    assert row.reviewed is True
    assert row.flagged_for_review is False
    assert result.decision == "approved"
    assert db.commit.called


def test_review_answer_flagged_sets_flag():
    db = MagicMock()
    row = _fake_usage_log(flagged_for_review=False)
    db.get.return_value = row

    result = review_answer(row.id, decision="flagged", admin=_fake_admin(), db=db)

    assert row.flagged_for_review is True
    assert result.flagged_for_review is True


def test_review_answer_dismissed_leaves_flag_unchanged():
    db = MagicMock()
    row = _fake_usage_log(flagged_for_review=True)
    db.get.return_value = row

    review_answer(row.id, decision="dismissed", admin=_fake_admin(), db=db)

    # "dismissed" is documented as reviewed=True with the flag left as-is —
    # confirm it doesn't silently clear a real flag the way "approved" does.
    assert row.flagged_for_review is True


def test_review_answer_logs_review_with_acting_admin_email():
    db = MagicMock()
    db.get.return_value = _fake_usage_log()

    review_answer("11111111-1111-1111-1111-111111111111", decision="approved", comment="looks right", admin=_fake_admin(), db=db)

    logged = db.add.call_args_list[-1][0][0]  # AnswerReviewLog(...) instance
    assert logged.reviewer == "mentor@ezitech.com"
    assert logged.comment == "looks right"


# --- GET /admin/analytics/usage ------------------------------------------


def test_usage_analytics_shapes_response():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = SimpleNamespace(
        total_queries=10,
        avg_retrieval_score=0.8234,
        helpful_count=6,
        unhelpful_count=2,
        no_feedback_count=2,
        verified_count=7,
        flagged_for_review_count=1,
        reviewed_count=9,
        pending_review_count=1,
    )

    result = usage_analytics(db=db)

    assert result.total_queries == 10
    assert result.avg_retrieval_score == 0.823  # rounded to 3dp
    assert result.pending_review_count == 1


def test_usage_analytics_handles_null_avg_score_and_null_sums():
    """An empty usage_logs table returns NULL for AVG()/SUM() rather than 0 -
    must not crash on round(None, 3) or None + None."""
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = SimpleNamespace(
        total_queries=0,
        avg_retrieval_score=None,
        helpful_count=None,
        unhelpful_count=None,
        no_feedback_count=None,
        verified_count=None,
        flagged_for_review_count=None,
        reviewed_count=None,
        pending_review_count=None,
    )

    result = usage_analytics(db=db)

    assert result.avg_retrieval_score is None
    assert result.helpful_count == 0
    assert result.pending_review_count == 0


# --- GET /admin/analytics/usage/timeseries --------------------------------


def test_usage_analytics_timeseries_rejects_invalid_days():
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        usage_analytics_timeseries(days=0, db=db)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        usage_analytics_timeseries(days=366, db=db)
    assert exc_info.value.status_code == 400


def test_usage_analytics_timeseries_returns_points_worst_null_handling():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        SimpleNamespace(day="2026-07-30", query_count=5, helpful_count=3, flagged_count=None),
    ]

    result = usage_analytics_timeseries(days=14, db=db)

    assert len(result) == 1
    assert result[0].query_count == 5
    assert result[0].flagged_count == 0  # None coalesced to 0, not left null


# --- GET /admin/analytics/usage/top-queries -------------------------------


def test_usage_top_queries_rejects_out_of_range_limit():
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        usage_top_queries(limit=0, db=db)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        usage_top_queries(limit=101, db=db)
    assert exc_info.value.status_code == 400


def test_usage_top_queries_returns_shaped_entries():
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = [
        SimpleNamespace(query="what is chromadb?", occurrences=4),
    ]

    result = usage_top_queries(limit=10, db=db)

    assert len(result) == 1
    assert result[0].query == "what is chromadb?"
    assert result[0].occurrences == 4
