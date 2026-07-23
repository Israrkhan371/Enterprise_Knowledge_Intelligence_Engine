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

## Scope decisions (documented per case-study evaluation criteria)

This scaffold wires up **every** functional requirement in the spec as a working
endpoint, prioritizing breadth first (Week 1-3) then hardening via evaluation
(Week 4) — see the accompanying `EKIE_4Week_Tracker.xlsx` for the day-by-day plan
mapped to each requirement.

- **Source loaders**: PDF and docx are implemented and manually verified
  against real files (`app/ingestion/loaders.py`) — verified 2026-07-16.
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
- **Relationship extraction** starts as sentence-level co-occurrence
  (`app/graph/extract.py`) — the noted upgrade path is an LLM-based relation
  labeler once you have ingestion volume to justify the extra latency/cost.
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
