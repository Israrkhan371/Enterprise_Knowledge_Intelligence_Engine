# EKIE — Integration Test Report

**Task:** Week 4 / Tue — End-to-end integration testing across all pipelines (backend + frontend)
**Date:** 2026-08-22
**Scope:** Every pipeline listed in the case study's AI Architecture Requirements, plus the frontend's use of the API.

This report has three parts: what was statically audited, the one real gap found and fixed, and how to run the live E2E check that this environment couldn't run itself (no Docker/Postgres/Neo4j available here — see "Environment constraint" below).

---

## 1. Method

Two passes were done, because they catch different classes of bug:

1. **Static wiring audit** — no live services needed, runs anywhere. Confirms every frontend call maps to a real, correctly-shaped backend route, and every Python file parses. This catches the exact class of bug the project already found twice before (FastAPI silently dropping undeclared query params — `category_filter`, `staleness_days`, `user_query_history`) by construction, since it's a direct route-signature-to-client-call comparison rather than trusting either side's docstrings.
2. **Live E2E script** (`scripts/e2e_integration_test.py`) — drives real HTTP calls through the full stack in dependency order: upload → embed → graph populate → search (5 modes) → RAG+citations → intelligence (summarize/compare/suggest-updates) → admin (approve/quality/analytics) → evaluation → monitoring. This is the only way to catch bugs in the *data flowing between* pipelines (e.g. the `embedding_id` fusion-key bug from Week 2, or the `document_id` citation bug from Week 2 — both were exactly this class of bug, and neither would show up in a route-signature check).

### Environment constraint

This sandbox has no Docker, Postgres, Neo4j, or GPU, and network access is restricted to package registries (no Docker Hub, no live Gemini). The live E2E script (#2) was therefore **written and logic-tested against a mock API server that mimics your route shapes** — every code path (full pass, missing-admin-id, Gemini-429-quota) was exercised and confirmed to report correctly and exit with the right status code. It has **not yet been run against your actual `docker compose` stack** — that's the one remaining step, and it's a single command (see §4).

---

## 2. Static audit results

| Check | Result |
|---|---|
| `python -m py_compile` on every file in `app/`, `scripts/`, `tests/` | **0 errors** |
| `node --check` on `frontend/app.js`, `frontend/api.js` | **0 errors** |
| Every `frontend/api.js` call vs. `app/api/routes.py` route table | **15/15 match** (path, method) |
| Every `frontend/api.js` admin call vs. `app/admin/routes.py` route table | **24/24 match** (path, method) |
| CSS classes referenced by new markup already defined in `app.css` | **confirmed** (`upload-form`, `inline-result`, `btn-ghost`, `panel-sub`, `meta`) |

No FastAPI-drops-undeclared-param-style bugs found in the current route set — the four earlier instances of that bug class (`category_filter`, `staleness_days`, `user_query_history`, and the semantic-search `document_id` field) all have both a route-signature match *and* a passing test already in `tests/`.

---

## 3. Gap found and fixed

**Compare Multiple Documents** and **Summarize Technical Documents** are both explicit case-study requirements ("AI Capabilities" section) with working, tested backend endpoints (`POST /documents/compare`, `GET /documents/{id}/summary`) and even pre-existing client stubs in `api.js` (`compareDocuments`, `documentSummary`) — but **no UI element ever called them**. A grader or demo audience clicking through the app would never see two of the required AI capabilities, even though they work.

**Fix applied:**
- `frontend/index.html`: added a "Compare & summarize" card to the Documents panel — one form for summarize-by-ID, one for compare-two-IDs. Deliberately placed *outside* the admin-gated corpus table (these are public endpoints — no `X-User-Id` needed), with a note saying so.
- `frontend/app.js`: wired both forms to the existing `API.documentSummary()` / `API.compareDocuments()` client methods, rendering `summary`, `diff`, and `similarity` exactly per `SummaryResponse`/`CompareResponse` in `app/api/routes.py`.
- Verified: `node --check` clean on both files; all referenced element IDs and CSS classes confirmed present.

No backend changes were needed — the endpoints were already correct and tested (`tests/test_intelligence_compare.py`, `tests/test_intelligence_summarize.py`).

---

## 4. Running the live E2E check

One command, against your running stack:

```bash
docker compose up --build          # if not already running
pip install httpx                  # one-time, on the host
python scripts/e2e_integration_test.py
```

For the admin-gated ~40% of checks (document approval, quality scoring, analytics, answer review), create an admin user once and pass its id:

```bash
docker exec -it ekie-postgres psql -U ekie -d ekie -c \
  "INSERT INTO users (id, email, name, role) VALUES (gen_random_uuid(), 'admin@ezitech.test', 'Admin', 'admin') RETURNING id;"

EKIE_ADMIN_ID=<paste-the-uuid> python scripts/e2e_integration_test.py
```

Expected shape of a clean run: all checks `PASS`, except `/ask`, `/documents/{id}/summary`, and `/documents/compare` may report `SKIP` if the Gemini free-tier daily quota (20 req/day) is already exhausted from earlier manual testing — that's a documented, expected constraint (see `docs/Evaluation_Report.md`), not a bug. Any `FAIL` line points to a real integration break and the script prints which check and why.

The script creates two small throwaway documents (`e2e_test_<random>.md`) during the run — safe to delete afterward via the Documents tab if you don't want them in your demo corpus.

---

## 5. Outstanding

- [ ] Run `scripts/e2e_integration_test.py` once against the live stack and confirm a clean (or expectedly-quota-skipped) result.
- [ ] If it surfaces any real `FAIL`, that's the next thing to fix before the demo.
