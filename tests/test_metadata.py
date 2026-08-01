"""
Integration tests for app/search/metadata.py. metadata_search() builds a
real SQLAlchemy query (join + multiple optional filters), so these tests
run it against a real Postgres connection rather than mocking the ORM
query-builder chain — same reasoning as test_keyword_search_integration.py.

Requires a live DB; the whole module is skipped if one isn't reachable.
Each test runs inside a transaction that's rolled back in the `db`
fixture's teardown, so nothing here is ever committed to a real database.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text as sql_text

from app.core.database import SessionLocal
from app.core.models import Category, Document
from app.search.metadata import metadata_search


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        session.execute(sql_text("SELECT 1"))
    except Exception as exc:
        session.close()
        pytest.skip(f"No live Postgres connection available for integration tests: {exc}")
    yield session
    session.rollback()
    session.close()


def _make_document(db, **kwargs):
    defaults = {"title": "untitled", "source_type": "markdown"}
    defaults.update(kwargs)
    doc = Document(**defaults)
    db.add(doc)
    db.flush()
    return doc


def test_metadata_search_filters_by_source_type(db):
    _make_document(db, title="a markdown doc", source_type="markdown")
    _make_document(db, title="a pdf doc", source_type="pdf")

    results = metadata_search(db, source_type="pdf", top_k=10)

    titles = [r["title"] for r in results]
    assert "a pdf doc" in titles
    assert "a markdown doc" not in titles


def test_metadata_search_filters_by_category(db):
    cat = Category(name=f"eng-docs-{datetime.utcnow().timestamp()}")
    db.add(cat)
    db.flush()

    _make_document(db, title="categorized doc", category_id=cat.id)
    _make_document(db, title="uncategorized doc")

    results = metadata_search(db, category=cat.name, top_k=10)

    titles = [r["title"] for r in results]
    assert "categorized doc" in titles
    assert "uncategorized doc" not in titles
    assert results[0]["category"] == cat.name


def test_metadata_search_filters_by_status(db):
    _make_document(db, title="pending doc", status="pending")
    _make_document(db, title="approved doc", status="approved")

    results = metadata_search(db, status="approved", top_k=10)

    titles = [r["title"] for r in results]
    assert "approved doc" in titles
    assert "pending doc" not in titles


def test_metadata_search_filters_by_date_range(db):
    now = datetime.utcnow()
    _make_document(db, title="old doc", created_at=now - timedelta(days=30))
    _make_document(db, title="recent doc", created_at=now)

    results = metadata_search(db, date_from=now - timedelta(days=1), top_k=10)

    titles = [r["title"] for r in results]
    assert "recent doc" in titles
    assert "old doc" not in titles


def test_metadata_search_combines_multiple_filters(db):
    cat = Category(name=f"combo-{datetime.utcnow().timestamp()}")
    db.add(cat)
    db.flush()

    _make_document(db, title="matches everything", source_type="pdf", category_id=cat.id, status="approved")
    _make_document(db, title="wrong source type", source_type="docx", category_id=cat.id, status="approved")
    _make_document(db, title="wrong status", source_type="pdf", category_id=cat.id, status="pending")

    results = metadata_search(db, category=cat.name, source_type="pdf", status="approved", top_k=10)

    titles = [r["title"] for r in results]
    assert titles == ["matches everything"]


def test_metadata_search_respects_top_k(db):
    for i in range(5):
        _make_document(db, title=f"doc-{i}")

    results = metadata_search(db, top_k=2)

    assert len(results) == 2


def test_metadata_search_returns_empty_list_when_nothing_matches(db):
    results = metadata_search(db, source_type="a-source-type-that-does-not-exist", top_k=10)

    assert results == []
