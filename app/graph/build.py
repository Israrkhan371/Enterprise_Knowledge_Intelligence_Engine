from neo4j import GraphDatabase

from app.core.config import settings


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )

    def close(self):
        self.driver.close()

    def init_schema(self):
        """
        Create uniqueness constraints and supporting indexes for the graph.
        Safe to call on every startup - Neo4j no-ops if they already exist.

        Node types:
          - Document(id)      unique
          - Entity(name)      unique
          - Entity(label)     indexed (used for technology-map / skill-dependency filters)

        Relationship types (defined implicitly by usage, Neo4j has no
        relationship-level constraints):
          - (:Document)-[:MENTIONS]->(:Entity)
          - (:Entity)-[:RELATES_TO {type, context}]->(:Entity)
        """
        statements = [
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX entity_label_index IF NOT EXISTS "
            "FOR (e:Entity) ON (e.label)",
        ]
        with self.driver.session() as session:
            for stmt in statements:
                session.run(stmt)

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
