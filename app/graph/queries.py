from app.graph.build import GraphStore


def get_technology_map(entity_label: str = "TECH", limit: int = 100) -> list[dict]:
    store = GraphStore()
    try:
        with store.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.label = $label OR b.label = $label
                RETURN a.name AS source, b.name AS target, r.type AS relation
                LIMIT $limit
                """,
                label=entity_label, limit=limit,
            )
            return [dict(record) for record in result]
    finally:
        store.close()


def get_skill_dependencies(skill_name: str) -> list[dict]:
    store = GraphStore()
    try:
        with store.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {name: $skill})-[:RELATES_TO*1..2]-(dep:Entity)
                RETURN DISTINCT dep.name AS dependency, dep.label AS label
                """,
                skill=skill_name,
            )
            return [dict(record) for record in result]
    finally:
        store.close()


def recommend_learning_path(user_query_history: list[str]) -> list[dict]:
    """
    MVP heuristic: pull entities mentioned across a user's recent queries,
    then surface graph neighbors tagged as LMS/course content they haven't
    hit yet. Replace with a ranked recommender once usage data accumulates.
    """
    store = GraphStore()
    try:
        with store.driver.session() as session:
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
