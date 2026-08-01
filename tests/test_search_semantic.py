from unittest.mock import patch

from app.search.semantic import semantic_search


def _fake_chroma_results(document_id: str, chunk_uuid: str):
    chunk_id = f"{document_id}::{chunk_uuid}"
    return {
        "ids": [[chunk_id]],
        "documents": [["some chunk text"]],
        "distances": [[0.12]],
        "metadatas": [[{"document_id": document_id}]],
    }


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query", return_value=[0.1, 0.2, 0.3])
def test_semantic_search_surfaces_document_id_from_metadata(mock_embed, mock_query):
    """
    Regression test: semantic_search() hits must carry a top-level
    'document_id' distinct from the raw Chroma chunk id. Both
    app/evaluation/eval.py and app/rag/generate.py read
    hit.get('document_id', hit.get('id')) — without this field, they
    silently fell back to the composite chunk id ("{document_id}::{uuid}"),
    breaking eval precision/recall/MRR scoring and citation source IDs for
    every semantic-search-sourced hit.
    """
    mock_query.return_value = _fake_chroma_results("doc-123", "chunk-abc")

    hits = semantic_search("what is the onboarding process?", top_k=1)

    assert len(hits) == 1
    assert hits[0]["document_id"] == "doc-123"
    assert hits[0]["id"] == "doc-123::chunk-abc"
    assert hits[0]["document_id"] != hits[0]["id"]


@patch("app.search.semantic.query_similar")
@patch("app.search.semantic.embed_query", return_value=[0.1, 0.2, 0.3])
def test_semantic_search_falls_back_to_parsing_chunk_id(mock_embed, mock_query):
    """If Chroma metadata is ever missing document_id, fall back to parsing
    it off the chunk id rather than silently returning the wrong id."""
    results = _fake_chroma_results("doc-456", "chunk-xyz")
    results["metadatas"] = [[{}]]  # metadata missing document_id
    mock_query.return_value = results

    hits = semantic_search("query", top_k=1)

    assert hits[0]["document_id"] == "doc-456"
