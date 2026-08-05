# Week 3 Checkpoint — Every Functional Requirement Has a Working Endpoint

Audit of every functional requirement in the case study brief against the
current API surface (`app/api/routes.py`, `app/admin/routes.py`). "Working"
here means: implemented, wired to a route, and (where the requirement is
logic-heavy rather than a thin passthrough) covered by tests — not just a
function that exists somewhere in `app/`.

Three real gaps were found during this audit and have been closed as part
of this checkpoint (see "Gaps found and fixed" below). Everything else was
already in place.

## Knowledge Sources → ingestion

All route through `POST /admin/documents/upload` (`source_type` param) or
the new `POST /admin/documents/ingest-github`, dispatched in
`app/ingestion/loaders.py::load_by_source_type()` / `SOURCE_LOADERS`.

| Requirement | Loader | Status |
|---|---|---|
| Internship Case Studies / Engineering Docs / SOPs / Company Policies | `load_docx` / `load_pdf` / `load_markdown` | ✅ |
| GitHub Repositories | `load_github_repo` via `ingest_github_repo()` | ✅ (endpoint added this checkpoint) |
| API Documentation | `load_api_docs` | ✅ |
| Database Schemas | `load_db_schema` | ✅ |
| LMS Courses | `load_lms` | ✅ |
| PDF Documents | `load_pdf` (+ OCR fallback, `app/ingestion/ocr.py`) | ✅ |
| Office Files | `load_docx` | ✅ |
| Meeting Notes | `load_meeting_notes` | ✅ |
| Recorded Sessions (transcripts) | `load_transcript` | ✅ |
| Technical Blogs | `load_blog` | ✅ |

## AI Capabilities

| Requirement | Endpoint | Status |
|---|---|---|
| Understand natural language queries | `POST /ask` | ✅ |
| Search across multiple knowledge sources | `GET /search/*` (all 5 modes) | ✅ |
| Retrieve relevant information | `GET /search/hybrid`, `POST /ask` | ✅ |
| Summarize technical documents | `GET /documents/{id}/summary` | ✅ |
| Compare multiple documents | `POST /documents/compare` | ✅ |
| Detect duplicate documentation | `GET /admin/quality/duplicates` | ✅ |
| Recommend missing documentation | `GET /admin/quality/missing-knowledge` | ✅ |
| Generate citations with every answer | `POST /ask` (`citation_check` field) | ✅ |
| Detect outdated knowledge | `GET /admin/quality/outdated` | ✅ |
| Suggest document updates | `GET /admin/documents/{id}/suggest-updates` | ⚠️→✅ **gap, fixed this checkpoint** |

## Knowledge Intelligence

| Requirement | Endpoint | Status |
|---|---|---|
| Knowledge Graph (build/populate) | populated at ingestion time, `app/graph/build.py` | ✅ |
| Engineering Relationships | `GET /graph/relationships/explain` | ✅ |
| Technology Maps | `GET /graph/technology-map` | ✅ |
| Skill Dependencies | `GET /graph/skill-dependencies` | ✅ |
| Learning Recommendations | `GET /graph/learning-recommendations` | ✅ |
| Missing Knowledge Alerts | `GET /admin/quality/missing-knowledge` | ✅ |

## AI Search

| Requirement | Endpoint | Status |
|---|---|---|
| Semantic Search | `GET /search/semantic` | ✅ |
| Hybrid Search | `GET /search/hybrid` | ✅ |
| Metadata Search | `GET /search/metadata` | ✅ |
| Keyword Search | `GET /search/keyword` | ✅ |
| Context-Aware Search | `GET /search/context-aware` | ⚠️→✅ **gap, fixed this checkpoint** |

## Administration

| Requirement | Endpoint | Status |
|---|---|---|
| Upload Documents | `POST /admin/documents/upload`, `POST /admin/documents/ingest-github` | ✅ |
| Approve Knowledge | `POST /admin/documents/{id}/approve` | ✅ |
| Manage Categories | `GET/POST /admin/categories` | ✅ (create/list only — no update/delete; acceptable for the case study's category set, flagged as a known limitation, not a blocking gap) |
| Review AI Answers | `GET /admin/answers`, `POST /admin/answers/{id}/review` | ✅ |
| Monitor Usage Analytics | `GET /admin/analytics/usage(+/timeseries,+/top-queries)` | ✅ |
| Track Knowledge Quality | `GET /admin/quality/{duplicates,outdated,gaps}`, `POST /admin/quality/score-all` | ✅ |

## Gaps found and fixed this checkpoint

1. **GitHub Repositories** were listed as a Knowledge Source, and
   `ingest_github_repo()` existed in `app/ingestion/pipeline.py`, but no
   route called it — a repo could never actually be ingested through the
   API, only a single uploaded file. Added `POST /admin/documents/ingest-github`.

2. **Context-Aware Search** was implemented (`rewrite_query()` in
   `app/search/context_aware.py`, already well-tested) but only reachable
   as a side effect of `POST /ask`. Unlike the other four AI Search modes,
   which each have their own directly callable, independently testable
   endpoint, this one had none. Added `GET /search/context-aware`, which
   rewrites the follow-up query using the supplied history and returns
   both the rewritten query and the resulting hybrid-search hits.

3. **"Suggest document updates"** had no implementation at all — the two
   adjacent features (`detect_outdated()`'s time-based staleness flag, and
   `version_intelligence.py`'s structural version-link suggestions) don't
   actually suggest *what* to update. Added
   `suggest_document_updates()` (`app/rag/intelligence.py`) and
   `GET /admin/documents/{id}/suggest-updates`: it finds other, more
   recently updated documents on the same topic via semantic search, and
   asks the LLM what the target document may be missing or have
   superseded — grounded only in those related documents, with a plain
   "nothing to compare against" result when no fresher related document
   exists, and a graceful fallback if the LLM call fails. Covered by
   `tests/test_intelligence_suggest_updates.py` (9 tests).

## Not yet run against a live stack

This audit was done by static review (route inspection + `py_compile`) in
an environment without the project's runtime dependencies or a live
Postgres/Neo4j/ChromaDB stack, so nothing here was exercised end-to-end via
`docker exec ekie-api pytest ...` or actual HTTP calls. Before treating this
checkpoint as fully closed, run the full suite (including the new
`test_intelligence_suggest_updates.py`) and smoke-test the three new
endpoints against a running stack.
