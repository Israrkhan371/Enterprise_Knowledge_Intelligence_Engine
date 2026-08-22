"""
End-to-end integration test for EKIE.

Unlike tests/ (unit tests, mocked, run inside the Docker image against no
live infra), this script proves the *whole system wired together* actually
works: it drives every pipeline through real HTTP calls against a running
`docker compose up` stack, in the order data actually flows through the
app, and checks the output of each stage feeds the next one correctly.

Pipelines exercised, in order:
  1. API + frontend liveness           (GET /health, GET /)
  2. Ingestion -> chunking -> embedding  (POST /admin/documents/upload)
  3. Vector store (semantic search)      (GET /search/semantic)
  4. Postgres FTS (keyword search)       (GET /search/keyword)
  5. Hybrid search (RRF + reranker)      (GET /search/hybrid)
  6. Metadata search                     (GET /search/metadata)
  7. Context-aware query rewriting       (GET /search/context-aware)
  8. Knowledge graph population          (GET /graph/technology-map, etc.)
  9. RAG answer generation + citations   (POST /ask, POST /ask/{id}/feedback)
 10. Document intelligence               (summary, compare, suggest-updates)
 11. Admin: review workflow              (approve, categories, versions)
 12. Admin: quality & analytics          (quality/*, analytics/*, answers/*)
 13. Evaluation framework + MLflow       (POST /evaluation/run)
 14. Monitoring                          (GET /metrics)

Usage:
    # from the host, with the stack up (docker compose up --build)
    pip install httpx
    python scripts/e2e_integration_test.py

    # with an admin user already created (see README "Frontend" section)
    EKIE_ADMIN_ID=<uuid> python scripts/e2e_integration_test.py

    # against a non-default host/port
    EKIE_BASE_URL=http://localhost:8000 python scripts/e2e_integration_test.py

Without EKIE_ADMIN_ID set, the script creates nothing for you (it never
talks to Postgres directly) - it prints the README's `INSERT INTO users`
command and skips the admin-gated section with a clear SKIP, instead of
failing the whole run.

Exit code is 0 only if every non-skipped check passed. Gemini-quota (429)
and Gemini-timeout (504) responses are treated as SKIP, not FAIL - the
20-req/day free-tier quota is a documented, expected constraint, not an
integration bug (see docs/Evaluation_Report.md).
"""

import os
import sys
import time
import uuid

import httpx

BASE_URL = os.environ.get("EKIE_BASE_URL", "http://localhost:8000").rstrip("/")
API = f"{BASE_URL}/api/v1"
ADMIN_ID = os.environ.get("EKIE_ADMIN_ID", "").strip()
TIMEOUT = httpx.Timeout(30.0, read=60.0)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []  # (name, status, detail)


def record(name, status, detail=""):
    results.append((name, status, detail))
    tag = {"PASS": "\033[32mPASS\033[0m", "FAIL": "\033[31mFAIL\033[0m", "SKIP": "\033[33mSKIP\033[0m"}[status]
    print(f"[{tag}] {name}" + (f" - {detail}" if detail else ""))


def admin_headers():
    return {"X-User-Id": ADMIN_ID} if ADMIN_ID else {}


def gemini_guard(name, fn):
    """Run an LLM-backed call; treat quota/timeout as SKIP, not FAIL."""
    try:
        return fn()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (429, 504):
            record(name, SKIP, f"LLM provider unavailable ({e.response.status_code}) - documented quota/timeout constraint, not a bug")
            return None
        raise


def main():
    client = httpx.Client(timeout=TIMEOUT)
    marker = f"zebra-quantum-marshmallow-{uuid.uuid4().hex[:8]}"

    # ---------------------------------------------------------------
    # 1. Liveness: API + frontend
    # ---------------------------------------------------------------
    try:
        r = client.get(f"{BASE_URL}/health")
        r.raise_for_status()
        body = r.json()
        assert body.get("status") == "ok", body
        record("API liveness (GET /health)", PASS, f"version={body.get('version')}")
    except Exception as e:
        record("API liveness (GET /health)", FAIL, str(e))
        print("\nAPI is not reachable - aborting the rest of the run. Is `docker compose up` running?")
        sys.exit(1)

    try:
        r = client.get(f"{BASE_URL}/")
        r.raise_for_status()
        assert "<div id=\"app\">" in r.text or 'id="app"' in r.text, "frontend markup not found"
        assert "ekie" in r.text.lower()
        record("Frontend served from API (GET /)", PASS)
    except Exception as e:
        record("Frontend served from API (GET /)", FAIL, str(e))

    try:
        r = client.get(f"{BASE_URL}/metrics")
        r.raise_for_status()
        assert "# HELP" in r.text or "# TYPE" in r.text
        record("Monitoring (GET /metrics, Prometheus)", PASS)
    except Exception as e:
        record("Monitoring (GET /metrics, Prometheus)", FAIL, str(e))

    # ---------------------------------------------------------------
    # 2. Ingestion -> chunking -> embedding -> vector store -> graph
    # ---------------------------------------------------------------
    doc_text = (
        f"# E2E Test Document ({marker})\n\n"
        f"This document exists solely for automated end-to-end testing of EKIE. "
        f"Its unique marker phrase is {marker}.\n\n"
        f"It discusses FastAPI, PostgreSQL, and Neo4j as core technologies used "
        f"to build the retrieval pipeline, and briefly compares ChromaDB against "
        f"other vector databases for storing embeddings.\n"
    )
    doc_id = None
    if not ADMIN_ID:
        record("Ingestion pipeline (POST /admin/documents/upload)", SKIP,
               "EKIE_ADMIN_ID not set - see README 'Frontend' section for the INSERT INTO users command")
    else:
        try:
            files = {"file": (f"e2e_test_{marker}.md", doc_text.encode(), "text/markdown")}
            data = {"source_type": "markdown", "title": f"E2E Test Doc {marker}"}
            r = client.post(f"{API}/admin/documents/upload", files=files, data=data, headers=admin_headers())
            r.raise_for_status()
            body = r.json()
            doc_id = body["document_id"]
            assert body["chunk_count"] > 0, "document ingested with zero chunks"
            record("Ingestion pipeline (POST /admin/documents/upload)", PASS,
                   f"document_id={doc_id}, chunks={body['chunk_count']}, status={body['status']}")
        except Exception as e:
            record("Ingestion pipeline (POST /admin/documents/upload)", FAIL, str(e))

    # Give embedding/graph population a beat in case of any async buffering
    # (current implementation is synchronous, but this keeps the script
    # robust if that ever changes).
    if doc_id:
        time.sleep(1)

    # ---------------------------------------------------------------
    # 3-7. Search layer
    # ---------------------------------------------------------------
    def check_marker_hit(name, url, params):
        try:
            r = client.get(url, params=params)
            r.raise_for_status()
            body = r.json()
            results_list = body if isinstance(body, list) else body.get("results", body.get("documents", []))
            found = any(marker in str(item) for item in results_list) if results_list else False
            if doc_id:
                assert found, f"uploaded test document not found in results: {str(body)[:300]}"
            record(name, PASS, f"{len(results_list) if hasattr(results_list, '__len__') else '?'} result(s)" + ("" if doc_id else " (no doc uploaded - ran unauthenticated, shape only)"))
        except Exception as e:
            record(name, FAIL, str(e))

    check_marker_hit("Semantic search (vector store)", f"{API}/search/semantic", {"q": marker, "top_k": 5})
    check_marker_hit("Keyword search (Postgres FTS)", f"{API}/search/keyword", {"q": marker, "top_k": 5})
    check_marker_hit("Hybrid search (RRF + cross-encoder rerank)", f"{API}/search/hybrid", {"q": marker, "top_k": 5})

    try:
        r = client.get(f"{API}/search/metadata", params={"source_type": "markdown", "top_k": 5})
        r.raise_for_status()
        record("Metadata search", PASS)
    except Exception as e:
        record("Metadata search", FAIL, str(e))

    try:
        r = client.get(f"{API}/search/context-aware", params={
            "q": "what are its main dependencies?",
            "history": ["Tell me about FastAPI"],
            "top_k": 5,
        })
        r.raise_for_status()
        record("Context-aware query rewriting", PASS)
    except Exception as e:
        record("Context-aware query rewriting", FAIL, str(e))

    # ---------------------------------------------------------------
    # 8. Knowledge graph
    # ---------------------------------------------------------------
    try:
        r = client.get(f"{API}/graph/technology-map", params={"entity_label": "TECH"})
        r.raise_for_status()
        body = r.json()
        record("Knowledge graph: technology map", PASS, f"{len(body) if hasattr(body, '__len__') else '?'} entries")
    except Exception as e:
        record("Knowledge graph: technology map", FAIL, str(e))

    try:
        r = client.get(f"{API}/graph/skill-dependencies")
        r.raise_for_status()
        record("Knowledge graph: skill dependencies", PASS)
    except Exception as e:
        record("Knowledge graph: skill dependencies", FAIL, str(e))

    try:
        r = client.get(f"{API}/graph/relationships/explain", params={"source": "FastAPI", "target": "PostgreSQL"})
        r.raise_for_status()
        record("Knowledge graph: relationship explain", PASS)
    except Exception as e:
        record("Knowledge graph: relationship explain", FAIL, str(e))

    try:
        r = client.get(f"{API}/graph/learning-recommendations", params={"user_query_history": [marker]})
        r.raise_for_status()
        record("Knowledge graph: learning recommendations", PASS)
    except Exception as e:
        record("Knowledge graph: learning recommendations", FAIL, str(e))

    # ---------------------------------------------------------------
    # 9. RAG: ask + citations + feedback
    # ---------------------------------------------------------------
    usage_log_id = None

    def do_ask():
        r = client.post(f"{API}/ask", json={"query": f"What is the marker phrase {marker} about?", "history": []})
        r.raise_for_status()
        return r.json()

    body = gemini_guard("RAG answer generation + citation check (POST /ask)", do_ask)
    if body is not None:
        try:
            assert "answer" in body and "citation_check" in body and "usage_log_id" in body
            usage_log_id = body["usage_log_id"]
            record("RAG answer generation + citation check (POST /ask)", PASS,
                   f"citation_verified={body['citation_check'].get('verified')}")
        except Exception as e:
            record("RAG answer generation + citation check (POST /ask)", FAIL, str(e))

    if usage_log_id:
        try:
            r = client.post(f"{API}/ask/{usage_log_id}/feedback", json={"was_helpful": True})
            r.raise_for_status()
            record("Answer feedback (POST /ask/{id}/feedback)", PASS)
        except Exception as e:
            record("Answer feedback (POST /ask/{id}/feedback)", FAIL, str(e))
    else:
        record("Answer feedback (POST /ask/{id}/feedback)", SKIP, "no usage_log_id from /ask")

    # ---------------------------------------------------------------
    # 10. Document intelligence: summary, compare, suggest-updates
    # ---------------------------------------------------------------
    if doc_id:
        name = "Document summarization (GET /documents/{id}/summary)"
        try:
            def _summarize():
                r = client.get(f"{API}/documents/{doc_id}/summary")
                r.raise_for_status()
                return r.json()
            body = gemini_guard(name, _summarize)
            if body is not None:
                assert body.get("summary")
                record(name, PASS)
        except Exception as e:
            record(name, FAIL, str(e))

        try:
            r = client.get(f"{API}/documents/{doc_id}/suggest-updates")
            r.raise_for_status()
            record("Suggest document updates", PASS)
        except Exception as e:
            record("Suggest document updates", FAIL, str(e))

        # Upload a second, overlapping doc so /documents/compare has two real IDs.
        if ADMIN_ID:
            try:
                doc2_text = doc_text.replace("ChromaDB", "Pinecone")
                files = {"file": (f"e2e_test_2_{marker}.md", doc2_text.encode(), "text/markdown")}
                data = {"source_type": "markdown", "title": f"E2E Test Doc 2 {marker}"}
                r = client.post(f"{API}/admin/documents/upload", files=files, data=data, headers=admin_headers())
                r.raise_for_status()
                doc_id_2 = r.json()["document_id"]

                name = "Document comparison (POST /documents/compare)"
                def _compare():
                    r = client.post(f"{API}/documents/compare",
                                     json={"document_id_a": doc_id, "document_id_b": doc_id_2})
                    r.raise_for_status()
                    return r.json()
                body = gemini_guard(name, _compare)
                if body is not None:
                    assert "summary" in body and "diff" in body
                    record(name, PASS, f"similarity={body.get('similarity')}")
            except Exception as e:
                record("Document comparison (POST /documents/compare)", FAIL, str(e))
    else:
        for name in ("Document summarization (GET /documents/{id}/summary)",
                     "Suggest document updates", "Document comparison (POST /documents/compare)"):
            record(name, SKIP, "no document_id (ingestion was skipped/failed)")

    # ---------------------------------------------------------------
    # 11 & 12. Admin: review workflow, quality, analytics
    # ---------------------------------------------------------------
    if not ADMIN_ID:
        record("Admin review/quality/analytics surface", SKIP,
               "EKIE_ADMIN_ID not set - run: docker exec -it ekie-postgres psql -U ekie -d ekie -c "
               "\"INSERT INTO users (id, email, name, role) VALUES (gen_random_uuid(), "
               "'admin@ezitech.test', 'Admin', 'admin') RETURNING id;\"")
    else:
        admin_checks = [
            ("Admin: list documents", "GET", "/admin/documents", {"limit": 5}),
            ("Admin: list categories", "GET", "/admin/categories", None),
            ("Admin: usage analytics", "GET", "/admin/analytics/usage", None),
            ("Admin: usage timeseries", "GET", "/admin/analytics/usage/timeseries", {"days": 14}),
            ("Admin: top queries", "GET", "/admin/analytics/usage/top-queries", {"limit": 5}),
            ("Admin: duplicate detection", "GET", "/admin/quality/duplicates", None),
            ("Admin: outdated detection", "GET", "/admin/quality/outdated", {"staleness_days": 180}),
            ("Admin: knowledge gap detection", "GET", "/admin/quality/gaps", None),
            ("Admin: missing-knowledge alerts", "GET", "/admin/quality/missing-knowledge", {"min_mentions": 1}),
            ("Admin: list AI answers (review queue)", "GET", "/admin/answers", {"limit": 5}),
        ]
        for name, method, path, params in admin_checks:
            try:
                r = client.request(method, f"{API}{path}", params=params, headers=admin_headers())
                r.raise_for_status()
                record(name, PASS)
            except Exception as e:
                record(name, FAIL, str(e))

        if doc_id:
            try:
                r = client.post(f"{API}/admin/documents/{doc_id}/approve",
                                 params={"decision": "approved", "comment": "e2e test"}, headers=admin_headers())
                r.raise_for_status()
                record("Admin: approve document", PASS)
            except Exception as e:
                record("Admin: approve document", FAIL, str(e))

            try:
                r = client.post(f"{API}/admin/documents/{doc_id}/score-quality", headers=admin_headers())
                r.raise_for_status()
                record("Admin: score document quality", PASS)
            except Exception as e:
                record("Admin: score document quality", FAIL, str(e))

            try:
                r = client.get(f"{API}/admin/documents/{doc_id}/version-candidates", headers=admin_headers())
                r.raise_for_status()
                record("Admin: version candidates", PASS)
            except Exception as e:
                record("Admin: version candidates", FAIL, str(e))

    # ---------------------------------------------------------------
    # 13. Evaluation framework + MLflow
    # ---------------------------------------------------------------
    try:
        r = client.post(f"{API}/evaluation/run", params={"k": 10})
        r.raise_for_status()
        body = r.json()
        record("Evaluation framework (POST /evaluation/run)", PASS,
               f"precision@10={body.get('precision_at_k')}, recall@10={body.get('recall_at_k')}, mrr={body.get('mrr')}")
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (429, 504):
            record("Evaluation framework (POST /evaluation/run)", SKIP, "LLM provider unavailable (quota/timeout)")
        else:
            record("Evaluation framework (POST /evaluation/run)", FAIL, str(e))
    except Exception as e:
        record("Evaluation framework (POST /evaluation/run)", FAIL, str(e))

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"E2E RESULT: {n_pass} passed, {n_fail} failed, {n_skip} skipped (of {len(results)} checks)")
    if n_fail:
        print("\nFailed checks:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
    print("=" * 70)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
