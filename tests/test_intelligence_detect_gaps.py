"""
Tests for detect_knowledge_gaps() in app/rag/intelligence.py.

detect_knowledge_gaps() groups usage_logs by query and surfaces recurring
queries with a low average retrieval_score -- questions the knowledge base
keeps failing to answer well. It's pure SQL (db.execute(text(...))), so
db.execute is mocked here the same way test_intelligence_detect_duplicates.py
mocks Postgres access -- no live DB required.
"""
from unittest.mock import MagicMock

from app.rag.intelligence import detect_knowledge_gaps


def _row(query, occurrences, avg_score):
    row = MagicMock()
    row.query = query
    row.occurrences = occurrences
    row.avg_score = avg_score
    return row


def _make_mock_db(rows):
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = rows
    return db


def test_detect_knowledge_gaps_shapes_results_correctly():
    db = _make_mock_db([_row("what is the offboarding process?", 5, 0.187)])

    results = detect_knowledge_gaps(db)

    assert results == [
        {"query": "what is the offboarding process?", "occurrences": 5, "avg_score": 0.187}
    ]


def test_detect_knowledge_gaps_returns_empty_list_when_nothing_qualifies():
    db = _make_mock_db([])

    results = detect_knowledge_gaps(db)

    assert results == []


def test_detect_knowledge_gaps_passes_threshold_and_min_occurrences_to_query():
    db = _make_mock_db([])

    detect_knowledge_gaps(db, min_score_threshold=0.5, min_occurrences=10)

    _, params = db.execute.call_args[0]
    assert params == {"threshold": 0.5, "min_occ": 10}


def test_detect_knowledge_gaps_defaults():
    db = _make_mock_db([])

    detect_knowledge_gaps(db)

    _, params = db.execute.call_args[0]
    assert params == {"threshold": 0.3, "min_occ": 3}


def test_detect_knowledge_gaps_rounds_avg_score_to_three_decimals():
    db = _make_mock_db([_row("vague query", 4, 0.123456)])

    results = detect_knowledge_gaps(db)

    assert results[0]["avg_score"] == 0.123


def test_detect_knowledge_gaps_preserves_query_order_from_db():
    """ORDER BY occurrences DESC happens in SQL; this must not re-sort."""
    rows = [_row("most common gap", 20, 0.1), _row("less common gap", 5, 0.2)]
    db = _make_mock_db(rows)

    results = detect_knowledge_gaps(db)

    assert [r["query"] for r in results] == ["most common gap", "less common gap"]
