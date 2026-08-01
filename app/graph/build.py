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

    def upsert_cooccurrence(self, document_id: str, source_type: str, pair_agg: dict[tuple[str, str], dict]):
        """
        Persists one document's aggregated co-occurrence counters
        (app/graph/relationships.aggregate_cooccurrences) onto
        (:Entity)-[:RELATES_TO]->(:Entity) edges, ACCUMULATING across every
        document processed so far rather than overwriting - confidence
        scoring (relationships.infer_relationship) depends on the running
        totals, not just the latest document. Relation type/confidence/
        evidence text are deliberately NOT stored here: they're derived at
        query time in app/graph/queries.py from these counters, so scores
        stay current as more documents get ingested without a separate
        backfill step.

        Idempotency: if `document_id` is already in this pair's
        supporting_documents (i.e. this document was already processed once
        before - a retry, an admin-triggered reprocess, etc.), the
        sentence/paragraph/document counters are NOT incremented again.
        Without this, reprocessing the same document silently inflates
        confidence every time it's re-ingested, even though nothing new was
        actually observed. evidence/supporting_documents/supporting_github_repos
        are set-unions already, so they stay correct either way - only the
        raw counts needed the explicit guard.

        Two round trips (read current lists, then write the union) instead
        of one Cypher statement because list-dedup needs APOC procedures
        this deployment doesn't depend on elsewhere - plain Cypher can't
        dedupe a list append inline.
        """
        with self.driver.session() as session:
            for (source, target), data in pair_agg.items():
                session.run(
                    """
                    MERGE (a:Entity {name: $source})
                    MERGE (b:Entity {name: $target})
                    MERGE (a)-[rel:RELATES_TO]->(b)
                    ON CREATE SET rel.sentence_count = 0, rel.paragraph_count = 0,
                        rel.document_count = 0, rel.evidence = [],
                        rel.supporting_documents = [], rel.supporting_github_repos = []
                    """,
                    source=source, target=target,
                )
                current = session.run(
                    """
                    MATCH (:Entity {name: $source})-[rel:RELATES_TO]->(:Entity {name: $target})
                    RETURN rel.evidence AS evidence, rel.supporting_documents AS docs,
                           rel.supporting_github_repos AS repos
                    """,
                    source=source, target=target,
                ).single()

                already_processed = document_id in (current["docs"] or [])
                evidence = set(current["evidence"] or []) | data["evidence"]
                docs = set(current["docs"] or []) | {document_id}
                repos = set(current["repos"] or [])
                if source_type == "github":
                    repos.add(document_id)

                session.run(
                    """
                    MATCH (:Entity {name: $source})-[rel:RELATES_TO]->(:Entity {name: $target})
                    SET rel.sentence_count = coalesce(rel.sentence_count, 0) + $sentence_count,
                        rel.paragraph_count = coalesce(rel.paragraph_count, 0) + $paragraph_count,
                        rel.document_count = coalesce(rel.document_count, 0) + $document_count,
                        rel.evidence = $evidence,
                        rel.supporting_documents = $docs,
                        rel.supporting_github_repos = $repos,
                        rel.sample_context = coalesce(rel.sample_context, $sample_context)
                    """,
                    source=source, target=target,
                    sentence_count=0 if already_processed else data["sentence_count"],
                    paragraph_count=0 if already_processed else data["paragraph_count"],
                    document_count=0 if already_processed else data["document_count"],
                    evidence=list(evidence), docs=list(docs), repos=list(repos),
                    sample_context=data.get("sample_context", ""),
                )

    def get_relationship_edges(self, entity_label: str | None = None) -> list[dict]:
        """Reads back accumulated co-occurrence counters for every edge (or
        every edge touching `entity_label`), for relationships.infer_relationship
        to score at query time."""
        where_clause = "WHERE a.label = $label OR b.label = $label" if entity_label else ""
        query = f"""
            MATCH (a:Entity)-[rel:RELATES_TO]->(b:Entity)
            {where_clause}
            RETURN a.name AS source, b.name AS target,
                   coalesce(rel.sentence_count, 0) AS sentence_count,
                   coalesce(rel.paragraph_count, 0) AS paragraph_count,
                   coalesce(rel.document_count, 0) AS document_count,
                   coalesce(rel.evidence, []) AS evidence,
                   size(coalesce(rel.supporting_documents, [])) AS supporting_documents,
                   size(coalesce(rel.supporting_github_repos, [])) AS supporting_github_repos
        """
        with self.driver.session() as session:
            params = {"label": entity_label} if entity_label else {}
            result = session.run(query, **params)
            return [dict(record) for record in result]
