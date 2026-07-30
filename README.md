# Enterprise Knowledge Intelligence Engine (EKIE)

Ezitech AI-007 case study — central AI intelligence layer over organizational knowledge.

## Architecture

```
Ingestion (loaders + OCR + chunking)
        v
Embedding pipeline (sentence-transformers) -> ChromaDB (vector store)
        v
Entity/relationship extraction (spaCy) -> Neo4j (knowledge graph)
        v
Search layer: semantic | keyword (Postgres FTS) | hybrid (RRF) | metadata | context-aware
        v
RAG answer generation (Google Gemini, free tier) + citation verification
        v
Intelligence layer: duplicate detection, staleness detection, gap detection,
                     comparison, summarization, ranking
        v
Admin layer: upload/approve/categories/analytics/quality  (FastAPI + Postgres)
        v
Evaluation: precision/recall/MRR against a labeled query set, tracked in MLflow
Monitoring: Prometheus metrics at /metrics
```

## Quickstart

```bash
cp .env.example .env          # fill in GOOGLE_API_KEY (free at https://aistudio.google.com/apikey)
docker compose up --build
```

- API: http://localhost:8000/docs (Swagger UI, auto-generated)
- Neo4j browser: http://localhost:7474
- MLflow UI: http://localhost:5000
- Prometheus: http://localhost:9090

## Project layout

| Path | Responsibility |
|---|---|
| `app/ingestion/` | Source loaders, OCR fallback, chunking |
| `app/embeddings/` | Embedding model wrapper, Chroma vector store |
| `app/search/` | Semantic, keyword, hybrid, context-aware search |
| `app/graph/` | Entity/relationship extraction, Neo4j read/write |
| `app/rag/` | Answer generation, citation verification, doc intelligence (duplicates, staleness, gaps, comparison, summarization) |
| `app/admin/` | Upload, approval workflow, categories, analytics, quality endpoints |
| `app/evaluation/` | Retrieval evaluation harness (precision/recall/MRR), MLflow logging |
| `app/core/` | Config, DB session, ORM models |
| `docs/knowledge_graph_schema.md` | Neo4j node/relationship types, constraints, indexes |

## Scope decisions (documented per case-study evaluation criteria)

This scaffold wires up **every** functional requirement in the spec as a working
endpoint, prioritizing breadth first (Week 1-3) then hardening via evaluation
(Week 4) — see the accompanying `EKIE_4Week_Tracker.xlsx` for the day-by-day plan
mapped to each requirement.

- **Source loaders**: PDF and docx are implemented and manually verified
  against real files (`app/ingestion/loaders.py`) — verified 2026-07-16.
  Both exclude `Title` elements from the text fed to embeddings/NER, since
  section headings glued directly onto body text were getting misread as
  ORG entities (e.g. "Business Problem" tagged as an organization);
  `load_docx` also excludes `Table` elements for the same reason (table
  cell values like "Business Value" from an evaluation-criteria table were
  showing up as fake entities), with a fallback to unfiltered text if
  filtering would otherwise leave a document empty (protects short
  documents where `unstructured` misclassifies the entire body as a
  Title).
  Markdown and code (`load_code()`) are implemented and covered by
  automated tests, along with GitHub, meeting notes, transcripts, and blog
  loaders (`tests/test_loaders.py`, 25 passed/1 skipped, verified
  2026-07-18). GitHub-repo loading (`load_github_repo`) returns `list[dict]`
  (one per file) instead of `str` like every other loader, so it is
  **deliberately excluded** from `SOURCE_LOADERS` and routed instead through
  `ingest_github_repo()` in `pipeline.py`, which fans each file out into its
  own `Document` row. It also recurses into subfolders (depth-guarded, with
  a directory exclusion list for `.git`/`node_modules`/`__pycache__`/etc.) —
  an earlier version only pulled root-level files, caught via manual testing
  against the real EKIE repo and fixed 2026-07-18 (re-verified: 38 files
  across every subfolder, no excluded dirs leaked in). Add new single-file
  formats by registering a function in `SOURCE_LOADERS`; multi-file sources
  like GitHub should follow the `ingest_github_repo()` fan-out pattern
  instead.
- **Keyword search** uses Postgres full-text search rather than standing up
  Elasticsearch/OpenSearch, to keep infra light; swap in `app/search/keyword.py`
  if you need BM25-grade ranking at larger scale.
- **Entity/relationship extraction and graph population** are implemented
  and confirmed wired end-to-end, not just present as standalone modules:
  `_populate_graph()` in `app/ingestion/pipeline.py` runs after every
  ingestion's Postgres/ChromaDB commit, calling `extract_entities()` /
  `extract_relationships()` (`app/graph/extract.py`, spaCy NER +
  sentence-level co-occurrence) and writing to Neo4j via `GraphStore`.
  Verified live via real document uploads across 8 source types (pdf,
  docx, markdown, code, meeting_notes, transcript, db_schema, blog/lms) —
  confirmed with before/after Postgres and Neo4j row counts, not just
  passing tests.
  Initial entity extraction was noisy; found and fixed via manual
  end-to-end testing rather than assumed correct: spaCy's default labels
  included non-entity categories (DATE/MONEY/TIME/CARDINAL) polluting the
  graph, now filtered to a relevant-label allowlist; a technology
  gazetteer (`entity_ruler`) was added to catch terms the base model
  misses or merges (e.g. "Rust", "PostgreSQL", "Python and Kubernetes"
  as one span); leading articles and bare adjectival demonyms (e.g. "the
  United States", "European") are stripped; a stoplist removes SQL/DDL
  syntax fragments and generic role labels (TABLE, DEFAULT, CTO,
  Attendees) that NER mistags as entities; gazetteer matches are
  case-canonicalized so "chromadb" and "ChromaDB" collapse to one node;
  and partial person names within a single document (e.g. "Priya
  Chandrasekaran" then later just "Chandrasekaran") are merged into one
  entity. All fixes are covered by the existing test suite plus manual
  verification against real documents.
  Known, deliberately out-of-scope limitations: entity resolution does
  **not** span across separate documents — the same person named
  differently in two different uploads currently becomes two graph
  nodes; fixing this needs a fuzzy/LLM-based matching step against
  existing graph entities and carries real false-positive risk (merging
  two different people who share a surname), so it's left as a Week 2+
  item rather than rushed. NER also still produces some noise on
  spec/requirement-style documents where bullet-list feature names
  (e.g. "API Documentation", "Hybrid Search") get misread as named
  entities — a general limitation of off-the-shelf NER on this document
  *type*, not a bug in the extraction code, noted here rather than
  silently left undocumented.
  The upgrade path for relationship extraction described below (multi-
  granularity co-occurrence + confidence scoring) replaces the sentence-level
  `co_occurs_with` MVP mentioned above; see the next bullet.
- **Technology maps & skill dependencies from entity co-occurrence**
  (`app/graph/relationships.py`, `app/graph/knowledge_base.py`) turn the raw
  co-occurrence edges above into scored, typed, explainable relationships —
  the actual Week 2 Monday deliverable, not just an endpoint that returns
  unscored graph edges (which is what existed before this pass).
  `extract_cooccurrences()` records co-occurrence at three granularities
  (sentence/paragraph/document, closest = highest confidence) and tags each
  with textual evidence signals (import statements, package files, deployment
  references, connection references, explicit dependency language).
  `infer_relationship()` scores every edge from two sources: a small curated
  baseline of well-known technology facts (`KNOWN_RELATIONS` — e.g. Python is
  a prerequisite of FastAPI, independent of what any given document says) and
  the actual observed frequency/evidence, which adjusts confidence up or down
  from that baseline (or is the only signal for pairs with no curated entry).
  A relationship with only one weak co-occurrence and no real evidence is
  deliberately downgraded to `RELATED_TO` rather than reported as a confident
  `DEPENDS_ON` — rule-based, not assumed. `GET /graph/relationships/explain`
  returns the full traceable breakdown (relation type, confidence 0-100,
  reasoning, evidence list) for one edge; `GET /graph/technology-map` groups
  edges into ecosystems, `GET /graph/skill-dependencies` orders `PREREQUISITE_OF`
  edges into a learning path, cross-checked against a curated skill chain so an
  inferred edge can't introduce a contradiction/cycle.
  Verified so far: 87 tests covering confidence-scoring bounds, direction
  canonicalization, weak-evidence demotion, ecosystem grouping, skill-chain
  ordering, evidence-window scoping, and the co-occurrence idempotency guard
  (below), all passing without needing Neo4j or the spaCy model except where
  noted (`tests/test_graph_relationships.py`, `tests/test_graph_extract_evidence.py`,
  `tests/test_graph_build.py`).
  **Not yet verified:** live against a running Neo4j instance — the
  accumulate-across-documents Cypher (`GraphStore.upsert_cooccurrence`) has
  only been reasoned through and mock-tested, not executed for real. Left
  deliberately open rather than faked; needs a real docker-compose run.

  Two correctness bugs found during review were fixed and regression-tested
  before this was considered done:
  - **Evidence misattribution (fixed).** Evidence was originally detected
    once per sentence/paragraph and applied to *every* entity pair in it —
    e.g. an import statement about entity A would get wrongly credited to an
    unrelated A-C pair. Sentence-level evidence is now scoped to a character
    window around the specific pair (`extract._pair_evidence_window`).
    Paragraph-level evidence detection was removed entirely rather than
    patched further: testing showed a pair spanning two different sentences
    in the same paragraph could still inherit evidence from an unrelated
    third sentence sandwiched between them, and there's no reliable regex-only
    way to rule that out. Paragraph/document granularity are now frequency-only
    signals with no evidence claims attached — a real scope reduction, not a
    silent gap. Properly solving evidence-for-cross-sentence-pairs needs actual
    relation extraction (the LLM-based upgrade path already noted above), not
    a better regex window.
  - **Double-counting on reprocessing (fixed).** `upsert_cooccurrence` now
    checks whether `document_id` is already in a pair's `supporting_documents`
    before incrementing sentence/paragraph/document counters, so re-running
    ingestion for the same document (retry, admin reprocess) no longer
    inflates confidence. `evidence`/`supporting_documents`/
    `supporting_github_repos` were already safe (set-union), only the raw
    counts needed the guard.

  Remaining known limitations — reviewed and deliberately left as documented
  trade-offs rather than fixed, since they affect scale/optimization on a
  sequential prototype rather than correctness of today's output:
  - `upsert_cooccurrence` does a read-then-write across two separate Cypher
    statements per entity pair, not one atomic transaction — safe under the
    current sequential ingestion path, but would silently lose updates under
    any concurrent writes to the same pair.
  - Three Neo4j round trips per unique entity pair per document (MERGE, read,
    write), looped in Python rather than batched — fine at current volume,
    will need batching before large bulk repo ingests.
  - Document-level co-occurrence (entities that never share a sentence or
    paragraph) is uncapped O(n²) per document — a document mentioning 20
    unrelated entities can generate ~190 weak pairs, adding graph noise with
    no cap or pruning.
  - For curated pairs, the relationship *type* always comes from
    `KNOWN_RELATIONS`, not from what your documents actually say — only the
    confidence score responds to observed evidence. Reasonable for
    well-established tech facts, but means "traceable to source" isn't fully
    true for that subset of edges.
  - Confidence-scoring weights (granularity/evidence/multi-source bonuses)
    are hand-tuned to produce sensible output on a couple of worked examples,
    not calibrated against any labeled ground truth — a candidate for the
    Week 3/4 evaluation framework rather than assumed correct.
  - `get_technology_map`'s `limit` is applied after scoring every edge
    returned by Neo4j, not in the Cypher query itself — wasteful once the
    graph has thousands of edges.
- **Bonus feature implemented:** AI Citation Verification (`app/rag/citation_check.py`)
  and Knowledge Gap Detection (`app/rag/intelligence.py::detect_knowledge_gaps`) —
  both reuse infrastructure already built for the core requirements.

## Known setup gotchas (already fixed in this repo — read before "fixing" them again)

The Docker build hit several real environment issues on 2026-07-16. All are
already resolved in the code/config below — documented here so nobody
re-discovers them the hard way on a fresh machine:

1. **The build installs from `requirements-lock.txt`, not `requirements.txt`.**
   `requirements.txt` is the human-readable "what depends on what and why"
   reference; `requirements-lock.txt` pins every package (not just direct
   deps) to one exact version frozen from a known-good build, so `pip` never
   backtracks/searches during install. Only edit `requirements.txt`
   directly when deliberately upgrading something — then re-freeze with
   `docker exec ekie-api pip freeze > requirements-lock.txt` once the new
   combination installs cleanly.
2. **`pdfminer.six` must stay at the version `pdfplumber` demands** (currently
   `20260107`) — it cannot be downgraded, pip will refuse the combination
   (`ResolutionImpossible`). The resulting `ImportError: cannot import name
   'PSSyntaxError'` from `unstructured`'s PDF partitioner is fixed with a
   compatibility shim at the top of `app/ingestion/loaders.py` instead of a
   version pin.
3. **NLTK's `punkt` data is pre-downloaded at Docker build time**
   (`docker/Dockerfile`). Without this, the first `load_docx`/`load_pdf` call
   at runtime tries to download it and fails with `HTTP 403` on networks that
   block that host.
4. **`unstructured`'s YOLOX layout model (~217MB) is also pre-downloaded at
   build time**, via a throwaway blank PDF built with `pypdf`. This step is
   placed in the Dockerfile *before* `COPY app ./app` on purpose, so editing
   application code never invalidates it or re-triggers the download. Do
   **not** wrap this step in `--mount=type=cache` — cache-mount contents are
   discarded when the layer commits, so the model would never actually end
   up in the final image (this was tried and had to be reverted).
5. **Neo4j takes 10-20+ seconds to start accepting Bolt connections** after
   its container reports "started" (it's a JVM app). Since `app/main.py`
   calls `GraphStore.init_schema()` at import time on every API startup, a
   plain `depends_on: - neo4j` (container-started, not service-ready) races
   this and crashes the API with
   `neo4j.exceptions.ServiceUnavailable: ... Connection refused`. Fixed two
   ways: `docker-compose.yml` gives `neo4j` a `cypher-shell`-based
   healthcheck and `api` depends on `neo4j: condition: service_healthy`;
   `app/main.py` also retries `init_schema()` with backoff as a safety net
   for running `api` outside that ordering (e.g. restarting it alone).

## Running tests

```bash
pip install -r requirements-lock.txt --break-system-packages
pytest tests/
```

Or, without touching your host Python at all, run inside the already-built
container:
```bash
docker exec -it ekie-api pytest tests/ -v
```

## Next steps

1. Fill in `app/evaluation/eval_set.json` with real Q&A pairs and their correct
   source document IDs, then hit `POST /api/v1/evaluation/run`.
2. Add a GIN index for Postgres full-text search (see comment in
   `app/search/keyword.py`) before load-testing keyword/hybrid search.
3. Wire `app/search/context_aware.py::rewrite_query` to a live LLM call —
   it's stubbed to a pass-through for now.
