"""
Tests for app/admin/routes.py's POST /documents/ingest-github route
(ingest_github()).

This route had no dedicated coverage before this pass — it was added in
the Week 3 enhancement commit (48ae8bb) alongside other admin work, but
only ingest_github_repo() (the pipeline function it calls) was ever
exercised, and only incidentally via string references in
test_seed_corpus.py/test_loaders.py, not the HTTP-layer route itself.

Follows the same call-the-route-function-directly-with-mocks pattern as
test_admin_documents.py, since these are unit tests of the route's own
logic (argument wiring, response shaping), not of ingest_github_repo()'s
internals (already covered separately) or of a live GitHub fetch.
"""
from unittest.mock import MagicMock, patch

from app.admin.routes import GithubIngestResponse, ingest_github


def _mock_document(doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.id = doc_id
    return doc


def test_ingest_github_passes_args_through_to_pipeline():
    db = MagicMock()
    admin = MagicMock(email="mentor@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = [_mock_document("doc-1")]

        ingest_github(
            repo_url="https://github.com/ezitech/ekie",
            category_id="cat-1",
            github_token="ghp_abc123",
            admin=admin,
            db=db,
        )

        mock_ingest.assert_called_once_with(
            db,
            "https://github.com/ezitech/ekie",
            category_id="cat-1",
            uploaded_by="mentor@ezitech.com",
            github_token="ghp_abc123",
        )


def test_ingest_github_defaults_category_and_token_to_none():
    # category_id/github_token are declared as Form(None) on the route,
    # which FastAPI only resolves to a real None through the actual
    # HTTP/ASGI dependency-injection layer. Calling ingest_github()
    # directly (as every test in this file does) bypasses that layer
    # entirely, so omitting the arguments here would leave category_id
    # bound to the literal Form(None) sentinel object, not None -
    # `sentinel is None` is False, so the test would fail for a reason
    # that has nothing to do with this route's own logic. Passing
    # explicit None values instead tests what this test actually cares
    # about: that ingest_github() forwards None straight through to
    # ingest_github_repo() unchanged, not FastAPI's own (already
    # framework-tested) Form-default resolution.
    db = MagicMock()
    admin = MagicMock(email="mentor@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = []

        ingest_github(
            repo_url="https://github.com/ezitech/ekie",
            category_id=None,
            github_token=None,
            admin=admin,
            db=db,
        )

        _, kwargs = mock_ingest.call_args
        assert kwargs["category_id"] is None
        assert kwargs["github_token"] is None


def test_ingest_github_uses_admin_email_as_uploaded_by_not_a_form_field():
    # uploaded_by isn't a client-supplied field on this route (unlike
    # upload_document's title) - it always comes from the authenticated
    # admin, so a caller can't spoof who ingested the repo.
    db = MagicMock()
    admin = MagicMock(email="someone-else@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = []

        ingest_github(repo_url="https://github.com/ezitech/ekie", admin=admin, db=db)

        assert mock_ingest.call_args.kwargs["uploaded_by"] == "someone-else@ezitech.com"


def test_ingest_github_returns_document_ids_and_file_count():
    db = MagicMock()
    admin = MagicMock(email="mentor@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = [
            _mock_document("doc-1"),
            _mock_document("doc-2"),
            _mock_document("doc-3"),
        ]

        result = ingest_github(repo_url="https://github.com/ezitech/ekie", admin=admin, db=db)

        assert isinstance(result, GithubIngestResponse)
        assert result.document_ids == ["doc-1", "doc-2", "doc-3"]
        assert result.file_count == 3


def test_ingest_github_handles_empty_repo_without_erroring():
    # An empty/all-excluded repo (e.g. only .git/node_modules content)
    # is a valid outcome, not a failure - load_github_repo() can
    # legitimately return no files after directory exclusion.
    db = MagicMock()
    admin = MagicMock(email="mentor@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = []

        result = ingest_github(repo_url="https://github.com/ezitech/empty-repo", admin=admin, db=db)

        assert result.document_ids == []
        assert result.file_count == 0


def test_ingest_github_file_count_matches_document_ids_length():
    # Regression guard: file_count is derived from len(documents), not
    # tracked separately, so the two can't silently drift apart.
    db = MagicMock()
    admin = MagicMock(email="mentor@ezitech.com")

    with patch("app.admin.routes.ingest_github_repo") as mock_ingest:
        mock_ingest.return_value = [_mock_document(f"doc-{i}") for i in range(7)]

        result = ingest_github(repo_url="https://github.com/ezitech/ekie", admin=admin, db=db)

        assert result.file_count == len(result.document_ids)