"""
Tests for app/search/rerank.py.

get_reranker() is lru_cache'd and loads a real CrossEncoder model on first
call, which needs a network fetch from huggingface.co and isn't available
in this environment — so every test here patches
app.search.rerank.get_reranker directly rather than exercising the real
model. cache_clear() is called after each test that touches the cache so
tests don't leak a mock CrossEncoder into other test modules that also
import get_reranker.
"""
from unittest.mock import MagicMock, patch

from app.search.rerank import get_reranker, rerank


def test_rerank_returns_empty_list_for_no_hits():
    assert rerank("query", [], top_k=5) == []


@patch("app.search.rerank.get_reranker")
def test_rerank_sorts_by_cross_encoder_score_descending(mock_get_reranker):
    mock_model = MagicMock()
    # Scores intentionally out of order vs. input to prove rerank() re-sorts
    # rather than trusting the input order.
    mock_model.predict.return_value = [0.1, 0.9, 0.5]
    mock_get_reranker.return_value = mock_model

    hits = [
        {"id": "a", "text": "low relevance"},
        {"id": "b", "text": "high relevance"},
        {"id": "c", "text": "medium relevance"},
    ]

    results = rerank("query", hits, top_k=3)

    assert [r["id"] for r in results] == ["b", "c", "a"]


@patch("app.search.rerank.get_reranker")
def test_rerank_truncates_to_top_k(mock_get_reranker):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3]
    mock_get_reranker.return_value = mock_model

    hits = [{"id": str(i), "text": f"chunk {i}"} for i in range(4)]

    results = rerank("query", hits, top_k=2)

    assert len(results) == 2
    assert [r["id"] for r in results] == ["1", "2"]  # scores 0.9 and 0.5


@patch("app.search.rerank.get_reranker")
def test_rerank_builds_query_text_pairs_for_every_hit(mock_get_reranker):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.0, 0.0]
    mock_get_reranker.return_value = mock_model

    hits = [{"id": "a", "text": "first chunk"}, {"id": "b", "text": "second chunk"}]
    rerank("my query", hits, top_k=2)

    passed_pairs = mock_model.predict.call_args[0][0]
    assert passed_pairs == [("my query", "first chunk"), ("my query", "second chunk")]


@patch("app.search.rerank.get_reranker")
def test_rerank_attaches_rerank_score_to_each_hit(mock_get_reranker):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.75]
    mock_get_reranker.return_value = mock_model

    hits = [{"id": "a", "text": "chunk"}]
    results = rerank("query", hits, top_k=1)

    assert results[0]["rerank_score"] == 0.75


def test_get_reranker_is_cached():
    """
    get_reranker() is lru_cache'd so repeated calls reuse the same loaded
    model instead of re-downloading/re-instantiating it on every search.
    """
    get_reranker.cache_clear()
    with patch("app.search.rerank.CrossEncoder") as mock_cls:
        mock_cls.return_value = MagicMock()
        first = get_reranker()
        second = get_reranker()

    assert first is second
    mock_cls.assert_called_once()
    get_reranker.cache_clear()


@patch("app.search.rerank.settings")
@patch("app.search.rerank.get_reranker")
def test_rerank_raises_timeout_error_if_predict_hangs(mock_get_reranker, mock_settings):
    """
    predict() is wrapped in call_with_timeout(); a predict() call that runs
    past reranker_timeout_seconds must surface as TimeoutError so
    hybrid_search() can catch it and fall back to un-reranked results,
    rather than hanging the request indefinitely.
    """
    import time

    mock_settings.reranker_timeout_seconds = 0.05

    def slow_predict(pairs):
        time.sleep(0.5)
        return [0.0] * len(pairs)

    mock_model = MagicMock()
    mock_model.predict.side_effect = slow_predict
    mock_get_reranker.return_value = mock_model

    hits = [{"id": "a", "text": "chunk"}]
    try:
        rerank("query", hits, top_k=1)
        assert False, "expected TimeoutError"
    except TimeoutError:
        pass
