"""
Tests for app/admin/routes.py's document listing/detail/approve endpoints.

Covers:
- GET /admin/documents input validation (limit bounds) and filter wiring.
- GET /admin/documents/{id} 404 behavior.
- POST /admin/documents/{id}/approve: previously returned
  {"error": "document not found"} with an HTTP 200 status for a missing
  document — a silent-failure response shape inconsistent with every other
  "not found" case in this codebase (app/api/routes.py's compare/summarize
  endpoints both raise HTTPException(404)). Now raises 404 consistently.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.admin.routes import approve_document, get_document, list_documents


def test_list_documents_rejects_out_of_range_limit():
    db = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        list_documents(limit=0, db=db)
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        list_documents(limit=201, db=db)
    assert exc_info.value.status_code == 400


def test_list_documents_applies_status_filter():
    db = MagicMock()
    query = db.query.return_value
    query.filter.return_value = query
    query.count.return_value = 0
    query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []

    list_documents(status="approved", db=db)

    # filter() was called (status filter applied) rather than skipped
    assert query.filter.called


def test_get_document_404_when_missing():
    db = MagicMock()
    db.get.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        get_document("nonexistent-id", db=db)
    assert exc_info.value.status_code == 404


def test_approve_document_404_when_missing_not_silent_200():
    db = MagicMock()
    db.get.return_value = None
    with pytest.raises(HTTPException) as exc_info:
        approve_document("nonexistent-id", reviewer="mentor@ezitech.com", decision="approved", db=db)
    assert exc_info.value.status_code == 404


def test_approve_document_sets_status_and_logs_decision():
    db = MagicMock()
    document = MagicMock(status="pending")
    db.get.return_value = document

    result = approve_document("doc-1", reviewer="mentor@ezitech.com", decision="approved", db=db)

    assert document.status == "approved"
    assert result.status == "approved"
    assert result.document_id == "doc-1"
    assert db.add.called  # ApprovalLog was added
    assert db.commit.called
