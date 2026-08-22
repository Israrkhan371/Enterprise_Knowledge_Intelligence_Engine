# EKIE — Week 4 Submission Checklist

**Task:** Week 4, Friday (Track B) — "Rehearse live demo; final bug sweep;
submit deliverables."

This checklist covers the full Week 4 deliverable set. Tasks 1-3
(end-to-end integration testing, architecture diagram, API documentation)
were completed in a separate pass before this one — see
`docs/Integration_Test_Report.md`, `docs/AI_Architecture_Diagram.md`, and
the `summary=`/`description=` annotations already on all 39 routes in
`app/api/routes.py` / `app/admin/routes.py`. This checklist focuses on
tasks 4-6 (README + Deployment Guide, presentation deck + demo script,
final bug sweep + submission), completed this pass, plus a fresh
end-to-end status check across everything.

## Static / code-quality checks (verified in this pass)

- [x] `python -m py_compile app/api/routes.py app/admin/routes.py
      scripts/e2e_integration_test.py` — 0 errors
- [x] `node --check frontend/app.js` — 0 errors
- [x] `node --check frontend/api.js` — 0 errors
- [x] No `TODO`/`FIXME` in `docs/Deployment_Guide.md` or
      `docs/Demo_Script.md`
- [x] `EKIE_Technical_Presentation.pptx` — passed
      `scripts/office/validate.py` (schema, relationships, content types,
      charts), `markitdown` content scan clean (no placeholder text), and
      a full 12-slide visual QA pass (checked for text overflow, contrast,
      overlap — none found)

## Requires the live Docker stack — NOT independently verified in this pass

**This environment has no Docker runtime**, so the following could not be
executed here:

- [ ] `docker exec ekie-api pytest tests/ -v` — expected to remain at or
      near the last recorded baseline (333 passed / 2 skipped as of the
      Week 3→4 checkpoint per README); this pass changed no test files.
- [ ] `python scripts/e2e_integration_test.py` (with `EKIE_ADMIN_ID` set)
      against a real running stack — this script and its report were
      written and validated in an earlier pass (see
      `docs/Integration_Test_Report.md` §1 "Environment constraint" for
      exactly how — logic-tested against a mock server, not yet run
      against real Postgres/Neo4j/ChromaDB). That remains the one
      outstanding step before treating end-to-end testing as fully closed.
- [ ] Live demo rehearsal per `docs/Demo_Script.md` — walk through every
      numbered step (3.1-3.8) against a running `docker compose up
      --build` stack, with a fresh upload, and confirm total runtime
      lands near the ~8 minute target. In particular, confirm the new
      Compare & Summarize UI (step 3.8) renders and calls the right
      endpoints — it was code-reviewed (`node --check` clean, element IDs
      and CSS classes confirmed present in `docs/Integration_Test_Report.md`
      §3) but not click-tested in a browser.
- [ ] Swagger UI spot-check — open `http://localhost:8000/docs` and
      confirm summaries/descriptions render correctly for a handful of
      the 39 annotated routes.

## Documentation cross-reference (verified by direct inspection)

- [x] `README.md` — "What's new in Week 4" section added, links to every
      new/updated doc, "Running the E2E integration test" and
      "Troubleshooting" sections added, `docs/Deployment_Guide.md` linked
      from Quickstart
- [x] `docs/Deployment_Guide.md` — new this pass; full `.env` reference,
      admin-user creation steps, explicit staging/production gap table
- [x] `docs/AI_Architecture_Diagram.md` — already existed from the earlier
      pass, re-read and confirmed accurate, referenced (not duplicated)
      from the new Deployment Guide and Demo Script
- [x] `docs/knowledge_graph_schema.md` — unchanged, re-read and confirmed
      still consistent with `app/graph/build.py`
- [x] `docs/Database_Schema_Diagram.md` — unchanged, no schema changes
      this pass
- [x] `docs/Evaluation_Report.md` — unchanged, referenced from the new
      Deployment Guide, Demo Script, and presentation deck
- [x] `docs/Integration_Test_Report.md` — already existed from the earlier
      pass; re-read to keep `docs/Demo_Script.md` and the presentation
      deck's script references (`scripts/e2e_integration_test.py`,
      `EKIE_ADMIN_ID`) consistent with what it actually documents
- [x] `docs/Demo_Script.md` — new this pass; timed walkthrough (3.1-3.8,
      including the new Compare & Summarize step) with talking points,
      expected output, and a fallback for every live-demo step
- [x] `EKIE_Technical_Presentation.pptx` — new this pass; 12 slides mapped
      1:1 to the demo script's sections (see the mapping table at the end
      of `docs/Demo_Script.md`)

## Outstanding known issues (carried forward, not new)

Pre-existing, already-documented trade-offs — listed here only so they
aren't mistaken for something this pass was supposed to fix:

- Ingestion is synchronous, no background job queue.
- Admin auth is a bare `X-User-Id` header, not session/token auth.
- Gemini free-tier quota (20 req/day/model) constrains `/ask`,
  `/documents/compare`, `/documents/{id}/summary`, and evaluation runs.
- No cross-document entity resolution in the knowledge graph.
- Citation-accuracy automated scoring has a documented methodological
  limit (see `docs/Evaluation_Report.md`) — closed as a negative finding
  with recommended follow-up, not something to "fix" here.

## What's genuinely new and ready to hand off (this pass)

| File | Status |
|---|---|
| `docs/Deployment_Guide.md` | New, complete |
| `docs/Demo_Script.md` | New, complete — updated to include the Compare & Summarize UI step |
| `EKIE_Technical_Presentation.pptx` | New, complete, QA'd |
| `SUBMISSION_CHECKLIST.md` | New, complete (this file) |
| `README.md` | Updated in place (Week 4 summary, E2E-test instructions, troubleshooting, links) |

**Before this ships as "submitted":** run the four unchecked items above
against your real `docker compose up --build` stack and update this file
with the actual pass/fail counts. Everything else in this checklist is
independently verifiable from the files themselves.
