from sqlalchemy.orm import Session

from app.search.semantic import semantic_search
from app.search.keyword import keyword_search


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


def hybrid_search(db: Session, query: str, top_k: int = 10, category_filter: str | None = None) -> list[dict]:
    semantic_hits = semantic_search(query, top_k=top_k, category_filter=category_filter)
    keyword_hits = keyword_search(db, query, top_k=top_k)

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

    return fused[:top_k]
