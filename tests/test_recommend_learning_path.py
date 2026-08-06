"""
Tests for app/graph/queries.py::recommend_learning_path().

Regression test for a real bug found via code review: the function took
user_query_history as a parameter but never referenced it anywhere in its
body - it ran one static Cypher query returning up to 20 arbitrary
LMS/entity pairs regardless of what was passed in, silently ignoring the
entire point of "query history -> skill graph -> LMS content".

GraphStore's Neo4j driver is mocked throughout - no live Neo4j required.
extract_entities() (spaCy NER) is also mocked, since these tests are about
recommend_learning_path()'s own query-building logic, not NER accuracy.
"""
from unittest.mock import MagicMock, patch

from app.graph.queries import recommend_learning_path


def _mock_store(run_return_values):
    """
    run_return_values: list of return values for successive session.run()
    calls, in call order (e.g. [entity_matched_rows, generic_rows]).
    Each "row" is a plain dict, since the real code does dict(record) on
    each Neo4j record - dict(some_dict) just copies it, which matches
    real behavior far more accurately than trying to fake it with
    MagicMock (whose default __iter__/keys() don't replicate a real
    Neo4j Record's mapping behavior).
    """
    store = MagicMock()
    session = MagicMock()
    session.run.side_effect = [iter(rows) for rows in run_return_values]
    store.driver.session.return_value.__enter__.return_value = session
    return store


@patch("app.graph.queries.extract_entities")
@patch("app.graph.queries.GraphStore")
def test_uses_entities_extracted_from_query_history(mock_graphstore_cls, mock_extract_entities):
    mock_extract_entities.side_effect = [
        [{"text": "Kubernetes", "label": "TECH"}],
        [{"text": "PostgreSQL", "label": "TECH"}],
    ]
    store = _mock_store([[{"course": "Intro to Kubernetes", "related_entity": "Kubernetes"}]])
    mock_graphstore_cls.return_value = store

    result = recommend_learning_path(["how does Kubernetes scaling work", "PostgreSQL replication basics"])

    assert result == [{"course": "Intro to Kubernetes", "related_entity": "Kubernetes"}]
    assert mock_extract_entities.call_count == 2


@patch("app.graph.queries.extract_entities")
@patch("app.graph.queries.GraphStore")
def test_query_passed_to_cypher_includes_extracted_entity_names(mock_graphstore_cls, mock_extract_entities):
    mock_extract_entities.return_value = [{"text": "Neo4j", "label": "TECH"}]
    store = MagicMock()
    session = MagicMock()
    session.run.return_value = iter([])
    store.driver.session.return_value.__enter__.return_value = session
    mock_graphstore_cls.return_value = store

    recommend_learning_path(["what is Neo4j used for"])

    first_call_kwargs = session.run.call_args_list[0].kwargs
    assert first_call_kwargs["entity_names"] == ["Neo4j"]


@patch("app.graph.queries.extract_entities")
@patch("app.graph.queries.GraphStore")
def test_falls_back_to_generic_query_when_no_history(mock_graphstore_cls, mock_extract_entities):
    store = _mock_store([[{"course": "General LMS Course", "related_entity": "Python"}]])
    mock_graphstore_cls.return_value = store

    result = recommend_learning_path([])

    assert result == [{"course": "General LMS Course", "related_entity": "Python"}]
    mock_extract_entities.assert_not_called()


@patch("app.graph.queries.extract_entities")
@patch("app.graph.queries.GraphStore")
def test_falls_back_to_generic_query_when_entities_extracted_but_no_graph_match(
    mock_graphstore_cls, mock_extract_entities
):
    mock_extract_entities.return_value = [{"text": "ObscureTech", "label": "TECH"}]
    store = _mock_store([[], [{"course": "Fallback Course", "related_entity": "Python"}]])
    mock_graphstore_cls.return_value = store

    result = recommend_learning_path(["tell me about ObscureTech"])

    assert result == [{"course": "Fallback Course", "related_entity": "Python"}]


@patch("app.graph.queries.extract_entities")
@patch("app.graph.queries.GraphStore")
def test_closes_store_even_on_empty_history(mock_graphstore_cls, mock_extract_entities):
    store = _mock_store([[]])
    mock_graphstore_cls.return_value = store

    recommend_learning_path([])

    store.close.assert_called_once()