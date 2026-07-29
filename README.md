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
  The noted upgrade path for relationship extraction specifically is an
  LLM-based relation labeler once you have ingestion volume to justify
  the extra latency/cost — sentence-level co-occurrence is the deliberate
  MVP choice for now.
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
6. **The ChromaDB Python client and server image must stay on the same
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
7. **FastAPI silently drops query/form parameters that aren't declared in
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

## Next steps

1. Fill in `app/evaluation/eval_set.json` with real Q&A pairs and their correct
   source document IDs, then hit `POST /api/v1/evaluation/run`.
2. Add a GIN index for Postgres full-text search (see comment in
   `app/search/keyword.py`) before load-testing keyword/hybrid search.
3. Wire `app/search/context_aware.py::rewrite_query` to a live LLM call —
   it's stubbed to a pass-through for now.
