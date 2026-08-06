import logging

from app.graph.build import GraphStore

logger = logging.getLogger(__name__)

# An entity needs to be mentioned by at least this many distinct documents
# before its absence as a dedicated topic is worth flagging - a one-off
# mention isn't evidence the org needs documentation for it.
_DEFAULT_MIN_MENTIONS = 3


def _entity_mention_counts(min_mentions: int, entity_label: str | None) -> list[dict]:
    """Entities mentioned by at least `min_mentions` distinct documents,
    with the titles of every document that mentions them (used below to
    check whether any of those documents is actually *about* the entity)."""
    store = GraphStore()
    try:
        where = "WHERE e.label = $label" if entity_label else ""
        query = f"""
            MATCH (d:Document)-[:MENTIONS]->(e:Entity)
            {where}
            WITH e, collect(DISTINCT d.title) AS titles, count(DISTINCT d) AS mention_count
            WHERE mention_count >= $min_mentions
            RETURN e.name AS entity, e.label AS label, mention_count, titles
            ORDER BY mention_count DESC
        """
        params = {"min_mentions": min_mentions}
        if entity_label:
            params["label"] = entity_label
        with store.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
    finally:
        store.close()


def _has_dedicated_document(entity_name: str, titles: list[str]) -> bool:
    """A document 'about' an entity is approximated as one whose title
    contains the entity's name (case-insensitive) - a document that
    mentions PostgreSQL in passing doesn't count, but a document titled
    'PostgreSQL Setup Guide' does."""
    name = (entity_name or "").strip().lower()
    if not name:
        return False
    return any(name in (title or "").lower() for title in titles)


def detect_missing_knowledge(
    min_mentions: int = _DEFAULT_MIN_MENTIONS,
    entity_label: str | None = None,
) -> list[dict]:
    """
    Knowledge Intelligence: Missing Knowledge Alerts. Flags entities/topics
    that come up often across the corpus (mentioned by at least
    `min_mentions` distinct documents) but have no document actually
    dedicated to them - a signal the topic clearly matters to the org but is
    under-documented.

    This is a different signal from app.rag.intelligence.detect_knowledge_gaps(),
    which flags topics *users ask about* that retrieval answers poorly; this
    one flags topics the *corpus itself* talks about a lot without ever
    being the subject of a document. Sorted by mention_count descending, so
    the most broadly-referenced gaps surface first.

    Returns [] (rather than raising) if Neo4j is unreachable - this is a
    background quality signal, not a critical path.
    """
    try:
        entities = _entity_mention_counts(min_mentions, entity_label)
    except Exception:
        logger.exception("Failed to read entity mention counts from Neo4j for missing-knowledge detection.")
        return []

    return [
        {
            "entity": e["entity"],
            "label": e["label"],
            "mentioned_in_document_count": e["mention_count"],
            "mentioning_documents": e["titles"][:5],  # sample, not the full list
        }
        for e in entities
        if not _has_dedicated_document(e["entity"], e["titles"])
    ]
