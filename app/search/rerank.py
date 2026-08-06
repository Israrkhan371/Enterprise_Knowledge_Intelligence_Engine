from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.rag.gemini_utils import call_with_timeout


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Re-score `hits` with a cross-encoder and return the top_k, ordered by the
    cross-encoder's relevance score (descending).

    Why this sits on top of hybrid search rather than replacing it: a
    cross-encoder scores relevance by attending to the query and a candidate
    chunk *jointly* in one forward pass, which is far more accurate than the
    bi-encoder cosine similarity semantic_search() uses or the BM25-style
    ts_rank keyword_search() uses — both of those score the query and each
    chunk independently, so they can't capture things like word order or
    negation ("X does not cause Y" vs "X causes Y" embed almost identically).
    The cost of that accuracy is that a cross-encoder can't be pre-indexed:
    every (query, chunk) pair needs its own forward pass at query time, so it
    doesn't scale to scoring an entire corpus. Hybrid search's fused list is
    the fix — it cheaply narrows millions of chunks down to a small
    high-recall shortlist, and the cross-encoder then reorders just that
    shortlist for precision.

    `hits` must already carry a "text" field (both semantic_search() and
    keyword_search() hits do). Mutates each hit in place by adding
    "rerank_score", then returns a new sorted-and-sliced list.

    Wrapped in call_with_timeout (same helper Gemini calls use, despite
    living in app/rag/gemini_utils.py — it's a generic thread-based
    deadline, not Gemini-specific) so a stalled model load or an unusually
    large candidate pool can't hang the request indefinitely. Raises
    TimeoutError on expiry; hybrid_search() is responsible for catching
    that and falling back to the un-reranked fused list.
    """
    if not hits:
        return []

    pairs = [(query, hit["text"]) for hit in hits]
    scores = call_with_timeout(
        get_reranker().predict,
        pairs,
        timeout_seconds=settings.reranker_timeout_seconds,
    )

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]
