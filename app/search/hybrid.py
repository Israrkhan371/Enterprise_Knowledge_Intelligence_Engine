import logging

from sqlalchemy.orm import Session

from app.core.config import settings
from app.search.semantic import semantic_search
from app.search.keyword import keyword_search
from app.search.rerank import rerank as cross_encoder_rerank

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(result_lists: list[list[dict]], key: str = "id", k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            item_id = item[key]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            # Merge rather than overwrite: a chunk found by both semantic
            # and keyword search should keep fields unique to each source
            # (e.g. semantic's "distance"/"metadata", keyword's "rank"),
            # not have the second list's dict silently replace the first's.
            payload[item_id] = {**payload.get(item_id, {}), **item}

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [{**payload[i], "fused_score": scores[i]} for i in ranked_ids]


def hybrid_search(
    db: Session,
    query: str,
    top_k: int = 10,
    category_filter: str | None = None,
    use_reranker: bool = True,
    rerank_pool_size: int | None = None,
) -> list[dict]:
    # When reranking, fetch a wider candidate pool than top_k from each leg
    # so the cross-encoder has real recall to work with — reranking a list
    # that was already truncated to top_k defeats the point, since the best
    # chunk might have been sitting at rank 15 of a bi-encoder/BM25 list.
    #
    # Clamped to rerank_pool_size_max regardless of what's requested: this
    # value can come straight from a caller (e.g. /search/hybrid's
    # rerank_pool_size query param), and an unbounded pool means fetching
    # that many rows from both semantic_search() and keyword_search() and
    # then running that many synchronous cross-encoder forward passes —
    # a cheap way to stall the API if this endpoint is reachable by
    # anyone other than a trusted caller.
    requested_pool_size = rerank_pool_size or settings.rerank_pool_size
    pool_size = min(max(top_k, requested_pool_size), settings.rerank_pool_size_max) if use_reranker else top_k

    semantic_hits = semantic_search(query, top_k=pool_size, category_filter=category_filter, db=db)
    keyword_hits = keyword_search(db, query, top_k=pool_size)

    # Normalize onto a shared fusion key before running RRF. Without this,
    # fusion silently never matches anything: semantic_search()'s "id" is
    # the ChromaDB vector id (e.g. "3c5df0fd-...::07c14190-..."), while
    # keyword_search()'s "id" is the unrelated Postgres document_chunks
    # primary key for the same row — two disjoint id spaces. RRF's whole
    # point is boosting a chunk that ranks well in *both* lists; keyed on
    # "id" directly, that could never happen, so hybrid search degraded to
    # "semantic results, then keyword results, re-sorted" instead of true
    # fusion. Bug found 2026-07-30 auditing this file for the week 2
    # Wednesday task; embedding_id is the value both sides actually share
    # (see keyword.py/semantic.py). Chunks without an embedding yet
    # (embedding_id is None) fall back to a key namespaced so it can never
    # collide with a real vector id or another un-embedded chunk.
    for hit in semantic_hits:
        hit["_fusion_key"] = hit["id"]
    for hit in keyword_hits:
        hit["_fusion_key"] = hit["embedding_id"] or f"no-embedding:{hit['id']}"

    fused = reciprocal_rank_fusion([semantic_hits, keyword_hits], key="_fusion_key")
    for hit in fused:
        hit.pop("_fusion_key", None)

    if use_reranker and fused:
        # Rerank the *entire* fused list, not fused[:pool_size]. RRF can
        # return up to 2 * pool_size unique items (a chunk found by only one
        # of the two legs keeps its own slot), and truncating to pool_size
        # here would silently drop candidates before the cross-encoder ever
        # saw them — including exactly the keyword-only-or-semantic-only
        # chunk the reranker is best positioned to correctly promote or
        # demote.
        try:
            return cross_encoder_rerank(query, fused, top_k=top_k)
        except Exception:
            logger.exception(
                "Cross-encoder reranking failed for query=%r; falling back "
                "to un-reranked fused results.", query,
            )
            return fused[:top_k]

    return fused[:top_k]