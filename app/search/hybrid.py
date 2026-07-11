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
            payload[item_id] = item

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [{**payload[i], "fused_score": scores[i]} for i in ranked_ids]


def hybrid_search(db: Session, query: str, top_k: int = 10) -> list[dict]:
    semantic_hits = semantic_search(query, top_k=top_k)
    keyword_hits = keyword_search(db, query, top_k=top_k)
    fused = reciprocal_rank_fusion([semantic_hits, keyword_hits])
    return fused[:top_k]
