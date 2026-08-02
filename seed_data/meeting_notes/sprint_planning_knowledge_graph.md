# Meeting Notes - Sprint Planning Knowledge Graph

**Date:** Week 1, Wednesday
**Attendees:** Engineer B, Mentor

## Agenda

Plan the entity extraction and graph population work for the knowledge
graph feature before Engineer B starts implementation.

## Discussion

- Reviewed spaCy NER as the extraction approach (vs. an LLM-based
  extractor) — chosen for speed and cost, since it runs on every
  ingested document and an LLM call per document would be slow and
  expensive at scale.
- Anticipated that raw spaCy NER output would be noisy (DATE/MONEY/TIME
  entities, merged multi-word tech terms) and agreed this should be
  treated as an expected first pass to harden through manual testing
  against real documents, not a one-shot implementation.
- Discussed the Document/Entity/MENTIONS/RELATES_TO graph shape:
  Document nodes mirror Postgres documents.id, Entity nodes are
  canonicalized by name, MENTIONS records what a document references,
  RELATES_TO accumulates co-occurrence evidence across all documents.
- Agreed relation type and confidence should be computed at query time
  (not stored per-edge), so classifications stay current as more
  documents get ingested without a backfill step.

## Decisions

1. Entity extraction pipeline ships in Week 1, wired into ingestion
   immediately (not built standalone and integrated later).
2. A curated gazetteer of known tech terms and known-relation baselines
   supplements raw spaCy NER, since generic NER misses or merges
   important technology names.
3. Cross-document entity resolution (same entity named differently
   across two separate documents) is explicitly out of scope for now —
   flagged as a known limitation, not a Week 1/2 blocker.

## Action items

- Engineer B: implement extract_entities/extract_relationships, wire
  into pipeline._populate_graph, and iterate on entity quality against
  real uploaded documents across every source type.
- Engineer B: derive technology maps and skill dependencies from
  accumulated co-occurrence evidence once extraction is stable.

## Related documents

- Meeting Notes - Week 2 Architecture Review
- Knowledge Graph Schema
