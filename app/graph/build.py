from neo4j import GraphDatabase

from app.core.config import settings


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def upsert_document_node(self, document_id: str, title: str, source_type: str):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.title = $title, d.source_type = $source_type
                """,
                id=document_id, title=title, source_type=source_type,
            )

    def upsert_entities(self, document_id: str, entities: list[dict]):
        with self.driver.session() as session:
            for e in entities:
                session.run(
                    """
                    MERGE (ent:Entity {name: $name})
                    SET ent.label = $label
                    WITH ent
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(ent)
                    """,
                    name=e["text"], label=e["label"], doc_id=document_id,
                )

    def upsert_relationships(self, relationships: list[dict]):
        with self.driver.session() as session:
            for r in relationships:
                session.run(
                    """
                    MERGE (a:Entity {name: $source})
                    MERGE (b:Entity {name: $target})
                    MERGE (a)-[rel:RELATES_TO {type: $relation}]->(b)
                    SET rel.context = $context
                    """,
                    source=r["source"], target=r["target"],
                    relation=r["relation"], context=r.get("context", ""),
                )
