# Knowledge Graph Schema

This documents the Neo4j graph populated by `app/graph/build.py` /
`app/graph/extract.py` / `app/graph/relationships.py`, and read back by
`app/graph/queries.py` (technology maps, skill dependencies, learning-path
recommendations, relationship explanations).

> This file was previously referenced from the README's project-layout
> table but never actually written. Reconstructed here directly from the
> schema-creation code (`GraphStore.init_schema()`), not written
> speculatively — see the source files above if anything here and the
> code ever disagree; the code is the source of truth.

## Node types

### `Document`

One node per ingested `Document` row (same `id` as the Postgres
`documents.id`), created/updated by `GraphStore.upsert_document_node()`.

| Property      | Type   | Notes                                   |
|---------------|--------|------------------------------------------|
| `id`          | string | Postgres `documents.id`. **Unique constraint.** |
| `title`       | string | Mirrors `documents.title` at ingest time. |
| `source_type` | string | e.g. `pdf`, `docx`, `github`, `sop`, `transcript`. |

### `Entity`

One node per unique entity name extracted via spaCy NER + the technology
gazetteer (`app/graph/extract.py`), created/updated by
`GraphStore.upsert_entities()`. Names are case-canonicalized at extraction
time (e.g. `"chromadb"` and `"ChromaDB"` collapse to one node).

| Property | Type   | Notes                                                         |
|----------|--------|-----------------------------------------------------------------|
| `name`   | string | Canonicalized entity text. **Unique constraint.**                |
| `label`  | string | NER/gazetteer label (e.g. `ORG`, `TECHNOLOGY`, `PERSON`). **Indexed.** |

## Relationship types

### `(:Document)-[:MENTIONS]->(:Entity)`

Created whenever an entity is extracted from a document. No properties —
existence of the edge is the signal.

### `(:Entity)-[:RELATES_TO {...}]->(:Entity)`

Accumulated co-occurrence evidence between two entities, written by
`GraphStore.upsert_cooccurrence()` and **accumulated across every document
ingested so far** (not overwritten per-document — see the docstring in
`build.py` for the idempotency guard that prevents re-ingesting the same
document from inflating counts).

| Property                      | Type      | Meaning |
|--------------------------------|-----------|---------|
| `sentence_count`                | int       | Times the pair co-occurred in the same sentence (strongest proximity signal). |
| `paragraph_count`               | int       | Times co-occurred in the same paragraph. |
| `document_count`                | int       | Times co-occurred anywhere in the same document. |
| `evidence`                      | list[str] | Evidence tags observed, e.g. `import_statement`, `package_file`, `dependency_keyword`, `deployment_reference`, `connection_reference`. |
| `supporting_documents`          | list[str] | Document ids that contributed evidence to this pair. |
| `supporting_github_repos`       | list[str] | Subset of the above that came from a `source_type="github"` document. |
| `sample_context`                | string    | First observed sentence/paragraph context, kept for display. |

`RELATES_TO` does **not** store a relation type or confidence score directly
— those are derived at **query time**, not write time, by
`relationships.infer_relationship()` in `app/graph/queries.py`, so a
relationship's classification stays current as more documents get ingested
without needing a separate backfill/reprocessing step.

## Derived relation types (query-time, not stored)

`infer_relationship()` classifies each `RELATES_TO` edge into one of:

| Relation type      | Meaning                                                       |
|---------------------|----------------------------------------------------------------|
| `PREREQUISITE_OF`   | One technology/skill must be learned before the other (e.g. Python -> FastAPI). Curated baseline in `knowledge_base.KNOWN_RELATIONS`. |
| `DEPENDS_ON`        | Direct code dependency (import/package-file evidence). |
| `CONNECTS_TO`       | Runtime connection (driver/connection-string evidence, e.g. SQLAlchemy -> PostgreSQL). |
| `DEPLOYS_TO`        | Deployment relationship (Dockerfile/deploy-command evidence, e.g. FastAPI -> Docker). |
| `REQUIRES`          | Explicit "requires"/"depends on"/"built on" language in documentation. |
| `RELATED_TO`        | Fallback: co-occur, but without strong enough evidence for a more specific claim (case-study Step 3: never over-claim from a single loose co-occurrence). |

Confidence is a blended score (`MIN_CONFIDENCE=5` .. `MAX_CONFIDENCE=97`) of:
a curated baseline for well-known technology pairs (`KNOWN_RELATIONS`),
plus weighted contributions from sentence/paragraph/document co-occurrence
counts, evidence-tag weights, a bonus for being seen across multiple
source documents, and a bonus for being seen across multiple GitHub repos.
Below `WEAK_EVIDENCE_THRESHOLD=40` with no real evidence, the relation is
forced back down to `RELATED_TO` regardless of any baseline, per case-study
Step 3.

## Constraints & indexes

Created idempotently on every API startup by `GraphStore.init_schema()`
(safe to re-run — Neo4j no-ops if they already exist):

```cypher
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;

CREATE INDEX entity_label_index IF NOT EXISTS
FOR (e:Entity) ON (e.label);
```

`entity_label_index` backs the technology-map / skill-dependency filters
in `app/graph/queries.py` (`get_technology_map`, `get_skill_dependencies`),
which query by `Entity.label`.

## Example queries

Get everything a document mentions:
```cypher
MATCH (d:Document {id: $document_id})-[:MENTIONS]->(e:Entity)
RETURN e.name, e.label
```

Get all relationships touching a technology entity, for `get_technology_map`:
```cypher
MATCH (a:Entity)-[rel:RELATES_TO]->(b:Entity)
WHERE a.label = $label OR b.label = $label
RETURN a.name AS source, b.name AS target,
       rel.sentence_count, rel.paragraph_count, rel.document_count,
       rel.evidence, rel.supporting_documents, rel.supporting_github_repos
```
(`infer_relationship()` is then applied in Python over each returned row to
attach the derived relation type/confidence/reason before returning to
the API.)

## Known limitations (see README "Scope decisions" for full detail)

- Entity resolution does **not** span across documents: the same person
  named differently in two uploads becomes two separate `Entity` nodes.
- `RELATES_TO` accumulates forever; there is currently no decay or
  staleness handling for co-occurrence evidence from very old documents.
