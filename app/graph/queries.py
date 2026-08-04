from app.graph.extract import extract_entities
from app.graph.build import GraphStore
from app.graph.relationships import (
    build_skill_dependency_graph,
    build_technology_map,
    format_explanation,
    infer_relationship,
)


def _inferred_relationships(entity_label: str | None = None) -> list[dict]:
    """Reads accumulated co-occurrence counters from Neo4j and scores every
    edge through infer_relationship() (case study Steps 6-10)."""
    store = GraphStore()
    try:
        edges = store.get_relationship_edges(entity_label)
    finally:
        store.close()

    return [
        infer_relationship(
            e["source"], e["target"],
            sentence_count=e["sentence_count"], paragraph_count=e["paragraph_count"],
            document_count=e["document_count"], evidence=set(e["evidence"]),
            supporting_documents=e["supporting_documents"] or 1,
            supporting_github_repos=e["supporting_github_repos"],
        )
        for e in edges
    ]


def get_technology_map(entity_label: str = "TECH", limit: int = 100) -> dict[str, list[dict]]:
    """Technology Map (Step 5): scored/typed/evidenced relationships grouped
    into ecosystems, e.g. {"Python": [{"source": "Python", "target": "FastAPI",
    "relation": "PREREQUISITE_OF", "confidence": 96, "reason": ..., "evidence": [...]}]}."""
    relationships = _inferred_relationships(entity_label)[:limit]
    return build_technology_map(relationships)


def get_skill_dependencies(skill_name: str | None = None) -> list[dict]:
    """Skill Dependency Graph (Step 6): ordered PREREQUISITE_OF chain,
    optionally filtered to edges touching one skill."""
    graph = build_skill_dependency_graph(_inferred_relationships())
    if skill_name:
        graph = [r for r in graph if skill_name in (r["source"], r["target"])]
    return graph


def explain_relationship(source: str, target: str) -> dict:
    """Full traceable explanation for one edge (Step 10 / the reviewer-facing
    "explain why" feature): relation type, confidence, reasoning, evidence -
    both structured and as a formatted human-readable block."""
    store = GraphStore()
    try:
        edges = store.get_relationship_edges()
    finally:
        store.close()

    match = next((e for e in edges if {e["source"], e["target"]} == {source, target}), None)
    if not match:
        return {"error": f"No relationship found between {source} and {target}"}

    rel = infer_relationship(
        match["source"], match["target"],
        sentence_count=match["sentence_count"], paragraph_count=match["paragraph_count"],
        document_count=match["document_count"], evidence=set(match["evidence"]),
        supporting_documents=match["supporting_documents"] or 1,
        supporting_github_repos=match["supporting_github_repos"],
    )
    rel["formatted"] = format_explanation(rel)
    return rel


def recommend_learning_path(user_query_history: list[str]) -> list[dict]:
    """
    Extracts entities actually mentioned in the user's recent queries, then
    surfaces LMS-tagged content connected to those specific entities in the
    graph - recommendations track what this user has been asking about,
    not just any LMS content that happens to exist.

    Bug fixed here: the previous implementation took user_query_history as
    a parameter but never referenced it anywhere in the function body - it
    ran one static query returning up to 20 arbitrary LMS/entity pairs
    regardless of what was passed in, silently ignoring the entire point
    of "query history -> skill graph -> LMS content".

    Falls back to that previous "any LMS content" behavior when there's no
    query history, or when none of the extracted query entities resolve to
    anything in the graph (cold start / no matching content yet) - an empty
    or unmatched history should mean a less targeted result, not an empty
    one.
    """
    store = GraphStore()
    try:
        with store.driver.session() as session:
            entity_names: set[str] = set()
            for query in user_query_history:
                for ent in extract_entities(query):
                    entity_names.add(ent["text"])

            if entity_names:
                result = session.run(
                    """
                    MATCH (d:Document)-[:MENTIONS]->(e:Entity)
                    WHERE d.source_type = 'lms' AND e.name IN $entity_names
                    RETURN DISTINCT d.title AS course, e.name AS related_entity
                    LIMIT 20
                    """,
                    entity_names=list(entity_names),
                )
                rows = [dict(record) for record in result]
                if rows:
                    return rows

            result = session.run(
                """
                MATCH (d:Document)-[:MENTIONS]->(e:Entity)
                WHERE d.source_type = 'lms'
                RETURN DISTINCT d.title AS course, e.name AS related_entity
                LIMIT 20
                """
            )
            return [dict(record) for record in result]
    finally:
        store.close()
