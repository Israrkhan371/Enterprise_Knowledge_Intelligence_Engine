"""
Tests for app/rag/intelligence.py::_find_newer_related_documents() and
suggest_document_updates() - the "Suggest Document Updates" case-study
requirement (LLM diff vs newer related content), which had no
implementation at all before this.

Postgres (db.execute), ChromaDB (get_collection, via _fetch_stored_embeddings),
the embedding model (embed_texts), and the LLM (_generate_content) are all
mocked. No live services required.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.rag import intelligence


def _row(**kwargs):
    row = MagicMock()
    for k, v in kwargs.items():
        setattr(row, k, v)
    return row


def test_returns_empty_list_when_document_not_found():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None

    result = intelligence._find_newer_related_documents(db, "missing-doc")

    assert result == []


def test_returns_empty_list_when_target_has_no_chunks():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = _row(id="doc-1", updated_at=datetime.utcnow())
    db.execute.return_value.fetchall.return_value = []

    result = intelligence._find_newer_related_documents(db, "doc-1")

    assert result == []


def test_returns_empty_list_when_no_newer_candidates_exist():
    now = datetime.utcnow()
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = _row(id="doc-1", updated_at=now)

    def fetchall_side_effect():
        calls = fetchall_side_effect.n
        fetchall_side_effect.n += 1
        return [_row(text="chunk", embedding_id="e1")] if calls == 0 else []
    fetchall_side_effect.n = 0
    db.execute.return_value.fetchall.side_effect = fetchall_side_effect

    with patch("app.rag.intelligence._fetch_stored_embeddings", return_value={}):
        with patch("app.rag.intelligence.embed_texts", return_value=[[0.1, 0.2]]):
            result = intelligence._find_newer_related_documents(db, "doc-1")

    assert result == []


def test_ranks_candidates_by_similarity_and_respects_threshold():
    now = datetime.utcnow()
    later = now + timedelta(days=5)
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = _row(id="doc-1", updated_at=now)

    target_chunks = [_row(text="target chunk", embedding_id="t1")]
    candidates = [
        _row(document_id="doc-2", text="close match", embedding_id="c1", title="Close Doc", updated_at=later),
        _row(document_id="doc-3", text="far match", embedding_id="c2", title="Far Doc", updated_at=later),
    ]

    fetchall_calls = [target_chunks, candidates]
    db.execute.return_value.fetchall.side_effect = lambda: fetchall_calls.pop(0)

    with patch("app.rag.intelligence._fetch_stored_embeddings", return_value={}):
        with patch(
            "app.rag.intelligence.embed_texts",
            side_effect=[[[1.0, 0.0]], [[0.99, 0.01], [0.1, 0.99]]],
        ):
            result = intelligence._find_newer_related_documents(db, "doc-1", min_similarity=0.75)

    assert len(result) == 1
    assert result[0]["document_id"] == "doc-2"
    assert result[0]["title"] == "Close Doc"


def test_returns_not_found_message_for_missing_document():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = None

    result = intelligence.suggest_document_updates(db, "missing-doc")

    assert result["message"] == "Document not found."
    assert result["suggestions"] is None
    assert result["related_documents"] == []


def test_returns_no_related_content_message_when_nothing_newer_found():
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = _row(id="doc-1", title="Old Doc", raw_text="old content")

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=[]):
        result = intelligence.suggest_document_updates(db, "doc-1")

    assert result["title"] == "Old Doc"
    assert result["suggestions"] is None
    assert result["related_documents"] == []
    assert "No newer related content" in result["message"]


def test_calls_llm_with_older_and_newer_content_when_related_docs_found():
    db = MagicMock()
    db.execute.return_value.fetchone.side_effect = [
        _row(id="doc-1", title="Old Doc", raw_text="old content"),
        _row(raw_text="newer content"),
    ]
    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", return_value="Suggestion: update X.") as mock_llm:
            result = intelligence.suggest_document_updates(db, "doc-1")

    assert result["suggestions"] == "Suggestion: update X."
    assert result["related_documents"] == related
    assert result["message"] is None
    prompt = mock_llm.call_args[0][0]
    assert "old content" in prompt
    assert "newer content" in prompt


def test_llm_timeout_propagates():
    db = MagicMock()
    db.execute.return_value.fetchone.side_effect = [
        _row(id="doc-1", title="Old Doc", raw_text="old content"),
        _row(raw_text="newer content"),
    ]
    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", side_effect=TimeoutError):
            try:
                intelligence.suggest_document_updates(db, "doc-1")
                assert False, "expected TimeoutError to propagate"
            except TimeoutError:
                pass


def test_llm_failure_falls_back_to_message_instead_of_raising():
    db = MagicMock()
    db.execute.return_value.fetchone.side_effect = [
        _row(id="doc-1", title="Old Doc", raw_text="old content"),
        _row(raw_text="newer content"),
    ]
    related = [{"document_id": "doc-2", "title": "New Doc", "similarity": 0.9, "updated_at": "2026-01-01"}]

    with patch("app.rag.intelligence._find_newer_related_documents", return_value=related):
        with patch("app.rag.intelligence._generate_content", side_effect=RuntimeError("gemini down")):
            result = intelligence.suggest_document_updates(db, "doc-1")

    assert "unavailable" in result["suggestions"].lower()