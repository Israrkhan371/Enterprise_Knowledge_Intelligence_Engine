# Meeting Notes - Week 2 Architecture Review

**Date:** Week 2, Monday
**Attendees:** Engineer A, Engineer B, Mentor

## Agenda

Review Week 1 checkpoint results and align on Week 2 search/RAG
architecture before starting implementation.

## Discussion

- Week 1 checkpoint confirmed: documents flow end-to-end through
  Postgres, ChromaDB, and Neo4j on upload. Graph entity quality needed
  several rounds of fixing (noisy spaCy labels, missed tech terms) but
  is now stable.
- Agreed to split Week 2 by layer rather than by feature: Engineer A
  owns the search stack (semantic, keyword, hybrid, metadata,
  context-aware rewriting), Engineer B owns document intelligence
  (comparison, duplicate detection, summarization) and graph
  intelligence (technology maps, skill dependencies).
- Decided hybrid search will use Reciprocal Rank Fusion rather than a
  weighted-score blend, since semantic and keyword scores aren't on
  comparable scales (see Dense vs Sparse Retrieval - Research Notes).
- Flagged that RAG answer generation and citation checking are
  interdependent — citation_check.py needs the same source list
  generate.py used, so they were scoped as one connected piece of work
  rather than two independent ones.

## Decisions

1. Hybrid search ships before RAG generation, since /ask depends on it
   for retrieval.
2. Evaluation framework (query set + precision/recall/MRR) gets a real
   pass this week, not just a placeholder, so Week 2's checkpoint can
   report actual retrieval quality numbers.
3. Metadata search and context-aware query rewriting are scoped as
   Week 2 Thursday work, after the rest of the search stack is stable.

## Action items

- Engineer A: semantic -> keyword -> hybrid -> metadata/context-aware,
  in that order.
- Engineer B: document comparison -> duplicate detection ->
  summarization -> technology maps/skill dependencies -> evaluation
  query set.
- Both: Week 2 checkpoint Friday — full search stack live, RAG cites
  sources, graph has real relationships.

## Related documents

- Meeting Notes - Sprint Planning Knowledge Graph
