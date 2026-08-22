# EKIE — Live Demo Script

**Task:** Week 4, Friday (Track A) — "Build technical presentation deck.
Update demo script + presentation deck to include live UI walkthrough."
Companion deliverable: `EKIE_Technical_Presentation.pptx` (slide deck built
from this script — see the mapping table at the end).

**Total runtime target:** ~8 minutes live demo + slides, leaving room for
Q&A in a typical case-study review slot.

**Before you start:**
- `docker compose up --build`, wait for `GET /health` to return `{"status": "ok"}`.
- Create (or reuse) an admin user and have its UUID ready — see
  `docs/Deployment_Guide.md` §1, or paste it into the frontend's ⚙ Admin
  tab (it's stored in `localStorage`, so you only need to do this once per
  browser).
- Run `pip install httpx && python scripts/e2e_integration_test.py`
  (optionally with `EKIE_ADMIN_ID=<uuid>` set for the admin-gated checks)
  from the host and confirm no `FAIL` lines — this is your pre-demo smoke
  test, not something to run live (see `docs/Integration_Test_Report.md`).
- Have 1-2 documents from `seed_data/` ready to upload live (a `.md` from
  `seed_data/sop/` uploads fastest and doesn't need OCR), separate from
  whatever's already seeded, so the audience sees ingestion happen live
  rather than pointing at pre-existing data.
- Open three browser tabs ahead of time: the frontend (`localhost:8000/`),
  Swagger UI (`localhost:8000/docs`), and Neo4j Browser
  (`localhost:17474`) — switching tabs live is smoother than launching
  them mid-demo.

---

## 1. Problem statement (~30 sec)

**Say:** Ezitech's engineering knowledge — case studies, SOPs, mentor
docs, code, recordings — is scattered across platforms. Interns ask the
same questions repeatedly, mentors give inconsistent answers, and
knowledge walks out the door when people leave. EKIE is the AI layer that
indexes all of it and answers questions with traceable sources.

**Show:** Nothing yet — this is scene-setting, stay on the title slide.

---

## 2. Architecture overview (~1 min)

**Say:** Six Docker services: FastAPI for the API and frontend,
PostgreSQL for structured data, ChromaDB for vector embeddings, Neo4j for
the knowledge graph, MLflow for evaluation tracking, Prometheus for
monitoring. A document's path: load → chunk → embed → store vectors →
extract entities → populate the graph — all synchronous, so by the time
an upload responds, it's already fully searchable.

**Show:** `docs/AI_Architecture_Diagram.md`, diagram 1 (system component
view) — either the rendered Mermaid in an editor, or the corresponding
slide.

**Fallback:** If screen-sharing a `.md` file is awkward live, the
presentation deck's Architecture slide has the same diagram as an image —
switch to that instead.

---

## 3. Live demo walkthrough (~4-5 min)

Do these in order, in the frontend UI (`localhost:8000/`) unless noted.

### 3.1 Upload a document (~45 sec)

**Do:** Documents tab → Upload form → pick a file from `seed_data/sop/` →
set `source_type` to `sop` → Upload.

**Say:** This one request does loading, chunking, embedding, and graph
population — point out the response includes a real `chunk_count`, not
just a success flag.

**Expected output:** `uploadResult` shows the new `document_id` and a
non-zero `chunk_count`.

**Fallback:** If the upload form errors on a specific file type, switch to
a plain `.md` file — every format's loader is covered, but `.md` has the
fewest moving parts (no OCR, no unstructured-library partitioning) for a
live-demo fallback.

### 3.2 Semantic search (~30 sec)

**Do:** Search tab → mode: Semantic → query with a phrase from the
document you just uploaded.

**Say:** This is nearest-neighbor search over 384-dim sentence-transformer
embeddings in ChromaDB — semantically similar text ranks high even
without exact keyword overlap.

**Expected output:** The just-uploaded document appears near the top of
`searchResults`, with a distance/score shown.

### 3.3 Hybrid search, same query (~30 sec)

**Do:** Search tab → mode: Hybrid → same query.

**Say:** Hybrid fuses semantic and keyword (Postgres full-text) results
with reciprocal rank fusion, then reranks the fused pool with a
cross-encoder for better precision at the top — compare the ordering to
the semantic-only results from 3.2.

**Expected output:** Results returned, ranking may differ subtly from 3.2.

### 3.4 Ask a question, RAG + citations (~1 min)

**Do:** Ask tab → ask a question the uploaded document answers.

**Say:** This retrieves relevant chunks, sends them to Gemini for
generation, and checks the answer's citations against the actual source
text before showing it — point out the "✓ citations verified" /
"⚠ N citation flag(s)" badge.

**Expected output:** A generated answer with inline citations and a
verification badge.

**Fallback:** If this returns a `429`, say so plainly: "This is the
documented Gemini free-tier limit — 20 requests/day/model, see
`docs/Fixing_Gemini_Quota.md`. It's an environment constraint, not a
pipeline bug" — then move on to 3.5 using a pre-recorded screenshot of a
successful `/ask` response instead of re-attempting live.

### 3.5 Give feedback (~15 sec)

**Do:** Click "👍 helpful" (or "👎 not helpful") under the answer from 3.4.

**Say:** This writes to `usage_logs` and feeds the Answer Review and
Analytics tabs on the admin side.

**Expected output:** The feedback buttons are replaced with a "feedback
recorded" badge.

### 3.6 Explore the knowledge graph (~1 min)

**Do:** Knowledge Graph tab → Technology Map (load with label `TECH`), then
Skill Dependencies.

**Say:** Entities were extracted via spaCy NER plus a curated technology
gazetteer; relationships between them (`DEPENDS_ON`, `CONNECTS_TO`,
`DEPLOYS_TO`, etc.) are inferred from co-occurrence evidence — not just a
flat entity list, real typed relationships with a confidence score. Point
out this is the same graph populated automatically by every upload,
including the one from 3.1.

**Expected output:** A populated technology map / dependency list. If the
just-uploaded document mentioned specific technologies, they should
appear.

**Fallback:** For a richer visual, switch to the Neo4j Browser tab and run
`MATCH (d:Document)-[:MENTIONS]->(e:Entity) RETURN d, e LIMIT 50` for a
graph visualization instead of the flat JSON view.

### 3.7 Admin: approval, quality, analytics (~45 sec)

**Do:** ⚙ Admin tab (paste admin UUID if not already set) → Documents →
approve the document from 3.1 → Quality & Gaps tab → show a quality score
→ Analytics tab → show usage over time.

**Say:** Approval is a review record, not a publish gate — the document
was searchable immediately after upload; approval tracks human sign-off
separately. Quality score blends completeness/freshness/originality;
Analytics aggregates real `usage_logs` rows, including the feedback from
3.5.

**Expected output:** Document status flips to `approved`; a quality score
appears; the analytics chart includes today's query.

### 3.8 Compare & summarize (~30 sec)

**Do:** Documents tab → "Compare & summarize" card → summarize the
document from 3.1 by ID, then compare it against any second document ID
from the corpus.

**Say:** Both are public endpoints (no admin header needed). This card
was added this pass specifically because the endpoints were already
implemented and tested but had no UI element calling them — a demo
audience would never have seen two of the case study's required AI
capabilities (Summarize Technical Documents, Compare Multiple Documents)
even though they worked (see `docs/Integration_Test_Report.md` §3 for the
gap and fix). Point out the diff/similarity fields, not just a generated
paragraph.

**Expected output:** `summaryResult` shows a generated summary;
`compareResult` shows a similarity score and diff/comparison text.

---

## 4. Key AI capabilities called out (~30 sec, can run while switching tabs above rather than as a separate stop)

- Semantic search + vector embeddings (ChromaDB)
- Hybrid search (reciprocal rank fusion + cross-encoder reranking)
- Knowledge graph (Neo4j, spaCy NER + gazetteer, inferred relationship
  types with confidence scoring)
- RAG with citation verification (not just generation — a second pass
  checks citations against source text)
- Duplicate detection, outdated-knowledge detection, missing-documentation
  gap detection, quality scoring — the "Knowledge Intelligence" layer
  beyond plain search

---

## 5. Evaluation results (~45 sec)

**Say:** Retrieval measured against a real 40-query labeled eval set:
precision@10 0.097, recall@10 0.897, MRR 0.608 (39/40 queries scored, one
correctly skipped for having no matching ingested document). Low
precision alongside high recall is expected here, not a red flag — most
queries only have a couple of truly relevant chunks against 10 returned
slots. Citation accuracy was investigated separately with two independent
scoring approaches; both hit a real methodological limit (score tracks
source-chunk length/breadth more than whether the specific cited fact is
present) — documented as a negative finding with a full manual-validation
table, not swept under the rug.

**Show:** `docs/Evaluation_Report.md` for the full numbers and the
citation-accuracy investigation.

---

## 6. Known limitations & next steps (~30 sec)

**Say, briefly, don't dwell:**
- Single-container design, synchronous ingestion — no background job
  queue yet (documented in `docs/Deployment_Guide.md` and
  `docs/AI_Architecture_Diagram.md`).
- Admin auth is a bare UUID header, fine for a local demo, not
  production-ready.
- Gemini free-tier quota is a real, recurring constraint during
  development and demo (20 req/day/model).
- Cross-document entity resolution isn't implemented — the same
  person/tool named differently across two uploads becomes two graph
  nodes.

**Show:** README "Next steps" section has the full list with more detail
if asked.

---

## Slide deck mapping

| Script section | Slide(s) |
|---|---|
| Title | 1 |
| §1 Problem statement | 2 |
| §2 Architecture overview | 3 |
| §3.1 Upload | 4 |
| §3.2-3.3 Search (semantic vs. hybrid) | 5 |
| §3.4-3.5 Ask (RAG + citations + feedback) | 6 |
| §3.6 Knowledge graph | 7 |
| §3.7-3.8 Admin + Compare & summarize | 8 |
| §4 Key AI capabilities | 9 |
| §5 Evaluation results | 10 |
| §6 Known limitations & next steps | 11 |
| Conclusion | 12 |

These are talking-point recap slides around the live UI, not a
replacement for it.

See `EKIE_Technical_Presentation.pptx` for the built deck.
