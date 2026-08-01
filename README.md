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
- `GET /health` returns `{"status": "ok", "version": ...}` — check this
  first if the API's behavior doesn't match what you expect from the
  source (e.g. a field missing from `/docs`/`/openapi.json`): a version
  mismatch there means the running container is stale and needs a rebuild
  (`docker compose up --build`, or `docker compose build --no-cache api`
  if compose is caching layers you don't want).

## Document lifecycle

```
POST /admin/documents/upload
        v
loading -> OCR (if needed) -> chunking -> embedding -> vector store upsert
-> Postgres DocumentChunk rows -> knowledge graph population   [synchronous —
   all of this finishes before the upload request returns]
        v
Document.status = "pending"        <-- admin review, NOT an ingestion/processing
                                        state. The document is already fully
                                        chunked/embedded/indexed at this point.
        v
GET /admin/documents               <-- discover document ids + status
GET /admin/documents/{id}          <-- inspect one document (status, chunk_count, ...)
POST /admin/documents/{id}/approve <-- record a review decision -> "approved"/"rejected"
```

**Important:** an uploaded document is queryable via `/search/*` and `/ask`
immediately after upload, before any admin approval — `status` currently
tracks review record-keeping only, not content visibility. If you need
unapproved documents excluded from search/RAG results, that's a filter to
add in `app/search/keyword.py` / `app/search/semantic.py` (`WHERE
documents.status = 'approved'`) — intentionally not added in this pass,
since it changes what's currently a "everything uploaded is live"
behavior into a real publish gate, which is a product decision, not a bug
fix.

## Project layout

| Path | Responsibility |
|---|---|
| `app/ingestion/` | Source loaders, OCR fallback, chunking |
| `app/embeddings/` | Embedding model wrapper, Chroma vector store |
| `app/search/` | Semantic, keyword, hybrid, context-aware search |
| `app/graph/` | Entity/relationship extraction, Neo4j read/write |
| `app/rag/` | Answer generation, citation verification, doc intelligence (duplicates, staleness, gaps, comparison, summarization); `gemini_utils.py` holds the shared `call_with_timeout()` used by every Gemini call site |
| `app/admin/` | Upload, document listing/detail, approval workflow, categories, analytics, quality endpoints |
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
  own `Document` row. **There is currently no REST/admin endpoint that calls
  `ingest_github_repo()`** — it must be invoked directly in Python
  (`docker exec -it ekie-api python`) until an admin route is added; every
  other loader is reachable via `POST /admin/documents/upload`. It also
  recurses into subfolders (depth-guarded, with a directory exclusion list
  for `.git`/`node_modules`/`__pycache__`/etc.) — an earlier version only
  pulled root-level files, caught via manual testing against the real EKIE
  repo and fixed 2026-07-18 (re-verified: 38 files across every subfolder,
  no excluded dirs leaked in). Add new single-file formats by registering a
  function in `SOURCE_LOADERS`; multi-file sources like GitHub should
  follow the `ingest_github_repo()` fan-out pattern instead.
- **API docs, DB schema, and LMS loaders** (`load_api_docs`, `load_db_schema`,
  `load_lms`) were added 2026-07-21, bringing `tests/test_loaders.py` to 35
  passed/1 skipped (36 collected). `load_api_docs` flattens an
  OpenAPI/Swagger JSON spec into prose (`METHOD /path — summary` plus
  params/response descriptions) rather than embedding raw JSON syntax.
  `load_db_schema` keeps `CREATE TABLE`/`ALTER TABLE`/`CREATE INDEX`/comment
  statements from a `.sql` dump but strips `INSERT INTO` rows and
  `COPY ... FROM stdin` data blocks, so real row data can never leak into
  the vector store — only schema *shape* is ever embedded. `load_lms`
  handles both a SCORM `.zip` package (concatenates every HTML content file
  inside, skips `imsmanifest.xml` packaging metadata) and a single exported
  `.html` file. All three read with `utf-8-sig` rather than `utf-8`, since
  files saved via Windows tools (e.g. PowerShell's `Out-File -Encoding
  utf8`) write a UTF-8 BOM that `json.loads()` chokes on
  (`load_api_docs` was the loader that surfaced this, since it's the only
  one doing strict JSON parsing — the others tolerated the stray BOM
  character silently). Verified live through the real upload endpoint for
  all three, not just via automated tests, including the BOM case
  specifically (2026-07-22).
- **Keyword search** uses Postgres full-text search rather than standing up
  Elasticsearch/OpenSearch, to keep infra light; swap in `app/search/keyword.py`
  if you need BM25-grade ranking at larger scale. Backed by a GIN index
  (`chunks_fts_idx` on `document_chunks`, declared in `app/core/models.py`
  via `Index(..., postgresql_using="gin")` so `create_all()` creates it —
  confirmed live via `pg_indexes`, added 2026-07-30). Covered by
  `tests/test_keyword.py` (7 tests, mocked at the query level) and
  `tests/test_keyword_search_integration.py` (11 tests: real loader → real
  chunker → real Postgres write → `keyword_search()`, one per source type,
  GitHub mocked to avoid a live network call; PDF is `@pytest.mark.skip`ed
  pending a real `tests/fixtures/sample.pdf`, same as `test_loaders.py`'s
  equivalent skip). Manually verified against real ingested content across
  all 11 source types, including PDF and a real GitHub repo run through
  the full `ingest_github_repo()` pipeline, not just `load_github_repo()`
  (2026-07-30). Known limitation: Postgres's tokenizer treats a leading
  `/` as part of the token, so a query for a path-like term (e.g. an
  OpenAPI path segment) needs the slash included to match — confirmed via
  the `/flangeburst987` case during that verification pass.
- **Hybrid search** (`app/search/hybrid.py`) fuses semantic + keyword results
  via reciprocal rank fusion. Found and fixed a real bug 2026-07-30: RRF
  fused on `"id"`, but `semantic_search()`'s `"id"` (a ChromaDB vector id
  like `"doc-1::abc"`) and `keyword_search()`'s `"id"` (the Postgres
  `document_chunks` primary key) are two disjoint id spaces for the *same*
  chunk — so a chunk ranked well by both methods could never be recognized
  as the same item and boosted, silently defeating the entire point of RRF
  fusion (it degraded to "semantic results, then keyword results,
  re-sorted"). Fixed by having `keyword_search()` also return
  `embedding_id` (the value `semantic_search()` already calls `"id"`) and
  fusing on that shared value instead, with a collision-safe fallback key
  for chunks that haven't been embedded yet. Also fixed RRF's payload
  merge, which previously let the second list's dict silently overwrite
  the first's fields on a match (losing e.g. `distance`/`metadata` from a
  semantic hit). `category_filter` added to `hybrid_search()`/
  `/search/hybrid` for parity with `/search/semantic`. Covered by
  `tests/test_hybrid.py` (9 tests, mocked).
- **Metadata search** (`app/search/metadata.py`, `GET /search/metadata`) —
  the missing piece flagged in the Week 2 Thursday tracker row: filters
  documents by category/source_type/status/date range rather than by
  textual relevance to a query (that's what semantic/keyword/hybrid search
  are for). Covered by `tests/test_metadata.py` (7 integration tests
  against a real, rolled-back Postgres transaction).
- **Context-aware query rewriting** (`app/search/context_aware.py::
  rewrite_query`, used by `/ask`) is now wired to a real Gemini call (same
  client pattern as `app/rag/generate.py`) instead of the previous stub
  that always returned the query unchanged regardless of conversation
  history. Falls back to the original query on any API error rather than
  raising, so a rewrite failure can't break `/ask`. Covered by
  `tests/test_context_aware.py` (7 tests, mocked) and manually verified
  live (2026-07-30): `rewrite_query("what are its main dependencies?",
  [{"role": "user", "content": "What is FastAPI?"}])` returned `"What are
  FastAPI's main dependencies?"` — a genuinely rewritten, pronoun-resolved
  query, confirming the LLM call actually executes rather than silently
  falling back. That check also surfaced a separate, unrelated problem:
  `GEMINI_MODEL=gemini-2.5-flash` (the default in `.env.example`/
  `app/core/config.py`) was rejected by the live API; fixed by switching
  the default to `GEMINI_MODEL=gemini-flash-latest`, confirmed working via
  a direct `generate_content` call returning real text.
- **RAG citation mapping** (`app/rag/generate.py`, `app/rag/
  citation_check.py`) — found and fixed a related bug while auditing
  hybrid search: `build_context_block()`'s `hit.get("document_id",
  hit.get("id"))` fallback silently substituted the ChromaDB vector id for
  the real document id in citations sourced from semantic-only hits,
  because `semantic_search()` never exposed `document_id` at the top level
  (only nested in `metadata`). Fixed in `app/search/semantic.py`. Neither
  `generate.py` nor `citation_check.py` had any tests before this session;
  now covered by `tests/test_generate.py` (6 tests) and
  `tests/test_citation_check.py` (7 tests), both mocked. Citation
  *accuracy* against a real query set is still unmeasured — see Next
  steps.
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
- **Duplicate detection** (`GET /admin/quality/duplicates`, `detect_duplicates()`
  in `app/rag/intelligence.py`) flags document pairs with near-identical
  content, via cosine similarity between chunk embeddings (default threshold
  0.92, overridable per call).
  - Reuses each chunk's embedding already stored in ChromaDB at ingestion
    time (`document_chunks.embedding_id`) instead of re-embedding every
    chunk on every call (`_fetch_stored_embeddings()`); only chunks with no
    stored embedding, or a Chroma miss, fall back to a fresh `embed_texts()`
    call. If that fallback itself fails, the affected chunks are excluded
    from that run rather than raising — duplicate detection is a background
    quality signal, not a critical path, so a partial result beats a hard
    failure.
  - Similarity is computed as a single vectorized `numpy` matrix multiply
    across all chunk vectors, not a Python-level O(n²) double loop —
    matters once the corpus grows past a few hundred chunks.
  - Results are aggregated to **one row per document pair** (the max
    similarity across all of that pair's matching chunks), not one row per
    matching chunk pair — two documents sharing several near-identical
    chunks would otherwise flood the report with repeat rows for the same
    two documents. Output is sorted by similarity, descending, and
    `document_a`/`document_b` are alphabetically ordered so the same pair
    always reports the same way regardless of which document's chunk was
    read first.
  - Covered by `tests/test_intelligence_detect_duplicates.py` (17 tests:
    threshold boundaries, same-document exclusion, stored-embedding reuse,
    fallback-embedding paths, aggregation/ordering).
  - Known, deliberately out-of-scope limitations: chunks that go through
    the `embed_texts()` fallback aren't written back to ChromaDB/
    `embedding_id`, so the same chunk gets re-embedded on every future call
    until it's re-ingested normally; and the underlying query
    (`SELECT * FROM document_chunks`) loads the whole table into memory
    with no batching/pagination. Both are scale concerns for a much larger
    corpus, not correctness bugs today.
- **Document Comparison Endpoint** (`POST /documents/compare`,
  `compare_documents_full()` in `app/rag/intelligence.py`) combines three
  signals for a pair of documents in one response: embedding similarity
  (`_embedding_similarity()` — cosine similarity between whole-document
  embeddings, `None` if either document is blank or embedding fails),
  a line-level unified diff (`_diff_texts()`, capped at 500 lines with a
  truncation marker so one huge document pair can't blow up the response),
  and an LLM-generated narrative summary of differences/overlaps
  (`compare_documents()`). Short documents (≤4000 chars each) are sent to
  Gemini in a single inline call; longer documents are chunked, each chunk
  summarized individually, then combined into one final comparison call
  — so the summary covers the whole document, not just its first few
  thousand characters. All Gemini calls go through a shared
  `call_with_timeout()` utility (`app/rag/gemini_utils.py`,
  `settings.gemini_timeout_seconds`, default 30s); a timeout returns
  HTTP 504 rather than hanging the request, and any other LLM failure
  falls back to a comparison response with a fallback `summary` string
  ("Summary unavailable: ...") plus the similarity/diff fields still
  populated, rather than failing the whole endpoint. Covered by
  `tests/test_intelligence_compare.py`.
- **Summarization Endpoint** (`GET /documents/{document_id}/summary`,
  `summarize_document_full()` in `app/rag/intelligence.py`) returns an
  on-demand LLM summary of a single document. Same inline-vs-chunked split
  as document comparison: documents ≤4000 chars are summarized in one call
  with their full text; longer documents are chunked
  (`_chunk_summarize()`), each chunk summarized individually, then a final
  `_combine_summaries()` call merges those into one coherent whole-document
  summary — replacing an earlier version that silently truncated anything
  past the first 6000 characters. The per-chunk leaf (`summarize_document()`)
  deliberately never decides to chunk itself: `chunk_text()`'s default
  800-word chunks run ~4-5k chars, just over the inline threshold, so a
  chunking-aware leaf re-fed its own chunk would re-chunk into an identical
  single chunk and recurse forever — `summarize_document_full()` is the only
  place that makes that decision. Same timeout/fallback handling as
  comparison: a Gemini timeout returns HTTP 504, any other LLM failure
  degrades to a fallback summary string rather than failing the request.
  Missing document IDs return HTTP 404 (previously returned `{"error": ...}`
  with a `200 OK`). Covered by `tests/test_intelligence_summarize.py`.

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
6. **`GEMINI_MODEL` default was `gemini-2.5-flash`, which the live Gemini
   API rejects.** Found 2026-07-30 while manually verifying `rewrite_query()`
   against the live API: a direct `client.models.generate_content()` call
   with that model name failed. `gemini-flash-latest` was confirmed working
   (both a plain test call and `rewrite_query()`'s actual usage returned real
   text) and is now the default in `app/core/config.py` and `.env.example`.
   If you already have a `.env` from before this fix, update `GEMINI_MODEL`
   there too - it isn't regenerated from `.env.example` automatically.
7. **The ChromaDB Python client and server image must stay on the same
   version.** `requirements-lock.txt` pins `chromadb==0.5.23`, but the
   `chromadb` service in `docker-compose.yml` had drifted to the older
   `chromadb/chroma:0.5.15` server image. The 0.5.23 client added a
   startup auth-identity handshake (`GET /auth/identity`) that the 0.5.15
   server doesn't expose, so *every* Chroma operation (upsert, query) fails
   with a `404 Not Found` wrapped in a confusing
   `ValueError: {"detail":"Not Found"}` — the traceback points at
   `get_user_identity()`, not at anything your code actually called, which
   makes it easy to mistake for a config/networking problem. Fixed by
   bumping the compose image to `chromadb/chroma:0.5.23` to match the
   client. If you ever bump `chromadb` in `requirements-lock.txt`, bump the
   compose image tag to match in the same change.
8. **FastAPI silently drops query/form parameters that aren't declared in
   the route function's signature — it does not error.** This bit us
   twice: `ingest_github_repo()` has no admin route at all yet (see the
   GitHub loader note above), and `semantic_search()`'s `category_filter`
   parameter was fully implemented, correctly applied, and covered by
   passing unit tests — but the `/search/semantic` route handler's own
   signature never declared `category_filter`, so it was accepted and
   silently discarded from every real HTTP request. Filtered and
   unfiltered queries returned identical results with no error anywhere in
   the logs. The lesson: when a search/pipeline function gains a new
   parameter, grep for every route that calls it and confirm the parameter
   is threaded all the way from the route signature through — a passing
   unit test on the underlying function proves nothing about whether the
   HTTP layer actually exposes it.

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

## Evaluation query set

`app/evaluation/eval_set.json` now has 40 Q&A pairs (target: 30-50) spanning
every knowledge source type in the spec — internship case studies, coding
standards, SOPs, GitHub repos (file-level), API docs, DB schemas, LMS
courses, research notes, meeting notes, transcripts, blogs, and company
policies — plus a couple of multi-document comparison queries and one
gap-detection query (asks about a document that's expected not to exist yet).

**Entries are keyed by `relevant_document_titles`, not raw document ids.**
`Document.id` is a random `uuid4` assigned at ingest time
(`gen_uuid()` in `app/core/models.py`) — it can't be known ahead of
ingestion, and it changes every time the corpus is wiped/reseeded. Hardcoding
ids into a checked-in fixture meant the eval set silently stopped matching
anything after the next reset (this is also why the two placeholder entries
that shipped before this pass had empty `relevant_document_ids: []` — nobody
could fill them in without going through the full ingest -> copy generated
UUID -> paste loop by hand every time). `app/evaluation/eval.py::resolve_relevant_ids()`
now resolves titles to whatever the *current* document ids are at evaluation
time, so the fixture stays valid across reseeds.

**To make these 40 queries actually score**, ingest documents via
`POST /admin/documents/upload` with a `title` matching each entry's
`relevant_document_titles` exactly (the endpoint now accepts an optional
`title` field — see "Known flaws fixed" below; it previously always used the
raw uploaded filename, which made hitting an exact title awkward). A raw
`relevant_document_ids` list is still supported per-entry for the rare case
a real id is already known. Entries whose titles don't resolve to any
ingested document are skipped (and counted in the `skipped` field of the
`run_evaluation()` result) rather than silently scored as a 0 — see the
docstring on `run_evaluation()`.

## Known flaws fixed in this pass

- **`semantic_search()` hits were missing `document_id`.** Both
  `app/evaluation/eval.py` and `app/rag/generate.py` read
  `hit.get("document_id", hit.get("id"))` to identify which document a hit
  came from — for citations, and for eval precision/recall/MRR scoring.
  Keyword-search hits always had `document_id`; semantic-search hits didn't,
  so that `.get()` silently fell back to the raw Chroma chunk id (a
  `"{document_id}::{uuid}"` composite string) instead of the real document
  id, for every semantic-search-sourced hit. This broke citation source ids
  in `/ask` responses and made eval scoring undercount semantic hits as
  misses even when the correct document was retrieved. Fixed in
  `app/search/semantic.py` (document_id is read from Chroma metadata, with
  a fallback to parsing it off the chunk id); regression-tested in
  `tests/test_search_semantic.py`.
- **`app/admin/routes.py::upload_document()` had no way to set a document's
  title independent of the uploaded filename.** Added an optional `title`
  form field (defaults to the filename, so existing behavior is unchanged).
- **`docs/knowledge_graph_schema.md`** was referenced in this README's
  project-layout table but was never actually created (confirmed via
  `git log` — no commit ever touched `docs/`). Added, reconstructed directly
  from `GraphStore.init_schema()` / `app/graph/relationships.py` /
  `app/graph/knowledge_base.py`.
- **No way to discover document ids or ingestion status without querying
  Postgres directly.** `POST /admin/documents/{id}/approve` needs an id,
  but nothing returned one after upload except the single upload response
  (easy to lose) — there was no listing or detail endpoint. Added
  `GET /admin/documents` (filterable by `status`/`source_type`/`category_id`,
  paginated) and `GET /admin/documents/{id}` (single document + `chunk_count`).
- **`POST /admin/documents/{id}/approve` returned `{"error": "document not
  found"}` with an HTTP 200 status** for a missing document — a silent
  failure inconsistent with every other "not found" case in this codebase
  (`app/api/routes.py`'s `/documents/compare` and `/documents/{id}/summary`
  both raise `HTTPException(404)`). Now raises 404 consistently; same fix
  applied to the new `GET /admin/documents/{id}`.
- **Every admin endpoint's OpenAPI response schema was `{}`** (no
  `response_model`), so `/docs` and generated clients couldn't tell you
  what a response actually contains. Added Pydantic response models
  (`DocumentUploadResponse`, `DocumentListResponse`,
  `DocumentDetailResponse`, `ApprovalResponse`, `CategoryResponse`,
  `UsageAnalyticsResponse`) for every admin endpoint that returns
  structured data.
- **The "pending" status was undocumented and easy to misread as a stuck
  processing state.** It isn't — ingestion is fully synchronous (see
  "Document lifecycle" above) and `status` only tracks admin review.
  Documented explicitly in the upload/approve endpoint descriptions (visible
  in `/docs`) and in this README, including the fact that unapproved
  documents are currently still searchable (flagged as a deliberate
  not-changed-here item, not silently "fixed" by adding a filter that would
  change existing behavior).

## Next steps

1. Ingest a real corpus with titles matching `app/evaluation/eval_set.json`
   (see "Evaluation query set" above), then hit `POST /api/v1/evaluation/run`
   to actually measure RAG/retrieval/citation accuracy — `/ask`,
   `citation_check.py`, and the precision/recall/MRR harness are implemented
   and unit-tested, but accuracy itself is still unmeasured against a real
   query set.
2. Run `GraphStore.upsert_cooccurrence()` live against a real docker-compose
   Neo4j instance — the accumulate-across-documents Cypher has only been
   reasoned through and mock-tested so far (see "Technology maps & skill
   dependencies" above).
3. Add an admin/REST route for `ingest_github_repo()` — it currently only
   runs via a direct Python shell (`docker exec -it ekie-api python`).
4. Add `tests/fixtures/sample.pdf` so `test_load_pdf_on_real_sample_if_present`
   (`test_loaders.py`) and `test_keyword_search_finds_pdf_content`
   (`test_keyword_search_integration.py`) can run instead of skipping.
5. Batch `upsert_cooccurrence`'s per-pair Neo4j round trips before any large
   bulk repo ingest — currently three round trips per unique entity pair per
   document, fine at current volume but not at scale (see documented
   limitation above).
6. Cross-document entity resolution (the same person/tool named differently
   across two separate uploads currently creates two graph nodes) —
   deliberately deferred; needs fuzzy/LLM-based matching with real
   false-positive risk, not a quick fix.
