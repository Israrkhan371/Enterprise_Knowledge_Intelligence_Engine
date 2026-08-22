# EKIE — AI Architecture Diagram

Case-study deliverable: "AI Architecture Diagram". Companion to
`docs/knowledge_graph_schema.md` (Knowledge Graph Design) and
`docs/Database_Schema_Diagram.md` (Database Schema).

Source of truth: `docker-compose.yml` (6 services) and `app/main.py`
(router wiring). Paste any block below into mermaid.live, GitHub/GitLab
markdown preview, or the VS Code Mermaid extension to render it.

---

## 1. System component diagram

Six Docker services, one FastAPI process holding five internal pipeline
layers plus the static frontend.

```mermaid
flowchart TB
    subgraph Client
        FE["Frontend (static HTML/JS/CSS)<br/>served by FastAPI at /"]
    end

    subgraph API["ekie-api (FastAPI, :8000)"]
        direction TB
        ROUTES["API layer<br/>app/api/routes.py (public)<br/>app/admin/routes.py (X-User-Id gated)"]
        subgraph ING["Ingestion"]
            LOAD["Loaders<br/>pdf/docx/md/code/github/<br/>transcripts/meeting notes/blogs"]
            OCR["OCR pipeline<br/>Tesseract, scanned PDFs"]
            CHUNK["Chunking<br/>app/ingestion/chunking.py"]
        end
        subgraph EMB["Embedding"]
            EMBED["sentence-transformers<br/>app/embeddings/embedder.py"]
        end
        subgraph GRAPH["Knowledge graph"]
            NER["Entity/relationship extraction<br/>spaCy NER + gazetteer<br/>app/graph/extract.py"]
        end
        subgraph SEARCH["Search layer"]
            SEM["Semantic"]
            KW["Keyword (Postgres FTS)"]
            HYB["Hybrid (RRF)"]
            RERANK["Cross-encoder rerank"]
            META["Metadata filter"]
            CTX["Context-aware rewrite"]
        end
        subgraph RAG["RAG + Intelligence"]
            GEN["Answer generation<br/>app/rag/generate.py"]
            CITE["Citation verification<br/>app/rag/citation_check.py"]
            INTEL["Compare / summarize / duplicate /<br/>outdated / gaps / suggest-updates<br/>app/rag/intelligence.py"]
            QUAL["Quality scoring<br/>app/rag/quality.py"]
        end
        METRICS["/metrics endpoint<br/>prometheus_client"]
    end

    PG[("PostgreSQL 16<br/>documents, chunks, users,<br/>usage_logs, approval_logs")]
    NEO[("Neo4j 5<br/>Document + Entity nodes<br/>MENTIONS / RELATES_TO")]
    CHROMA[("ChromaDB<br/>chunk embeddings")]
    LLM(["Google Gemini API<br/>(free tier, external)"])
    MLFLOW["MLflow<br/>eval run tracking, :5000"]
    PROM["Prometheus<br/>scrapes /metrics, :9090"]

    FE -->|"fetch() /api/v1/*"| ROUTES
    ROUTES --> ING
    LOAD --> OCR --> CHUNK --> EMB
    EMB -->|"upsert vectors"| CHROMA
    CHUNK -->|"Document + DocumentChunk rows"| PG
    CHUNK --> GRAPH
    NER -->|"upsert nodes/edges"| NEO

    ROUTES --> SEARCH
    SEM --> CHROMA
    KW --> PG
    HYB --> SEM
    HYB --> KW
    HYB --> RERANK
    META --> PG
    CTX -->|"query rewrite"| LLM

    ROUTES --> RAG
    GEN -->|"retrieve"| HYB
    GEN -->|"generate"| LLM
    GEN --> CITE
    RAG -->|"UsageLog"| PG
    INTEL --> LLM
    INTEL -->|"related-doc lookup"| CHROMA
    QUAL -->|"duplicate check"| CHROMA
    QUAL --> NEO

    ROUTES -->|"eval run"| MLFLOW
    METRICS --> PROM

    classDef store fill:#2d3548,stroke:#6b7cad,color:#fff;
    classDef ext fill:#4a3548,stroke:#c76b9c,color:#fff;
    class PG,NEO,CHROMA store;
    class LLM ext;
```

**Reading the diagram:** ingestion is synchronous end-to-end — a single
`POST /admin/documents/upload` request runs loading → OCR (if needed) →
chunking → embedding → ChromaDB upsert → Postgres row insert → graph
population, and doesn't return until all of it finishes. That's a
deliberate simplicity trade-off (see README "Scope decisions" — no queue,
no Celery/background worker) that keeps the demo predictable at the cost
of upload latency on large documents.

---

## 2. Request-flow sequence: `POST /ask` (RAG with citations)

The most architecturally interesting request — it touches every store and
the one external dependency (Gemini).

```mermaid
sequenceDiagram
    participant U as Frontend / caller
    participant API as FastAPI (/api/v1/ask)
    participant CTX as context_aware.rewrite_query
    participant HYB as hybrid_search
    participant CHROMA as ChromaDB
    participant PG as PostgreSQL
    participant LLM as Gemini API
    participant CITE as citation_check

    U->>API: POST /ask {query, history}
    API->>CTX: rewrite_query(query, history)
    CTX->>LLM: rewrite prompt
    LLM-->>CTX: rewritten query (or original on error)
    CTX-->>API: query'

    API->>HYB: hybrid_search(query')
    HYB->>CHROMA: semantic_search (vector)
    HYB->>PG: keyword_search (FTS)
    HYB->>HYB: reciprocal rank fusion + cross-encoder rerank
    HYB-->>API: ranked chunks + source doc ids

    API->>LLM: generate_answer(context = ranked chunks)
    LLM-->>API: answer text
    API->>CITE: verify_citations(answer, sources)
    CITE-->>API: {verified, flags}

    API->>PG: INSERT UsageLog (query, answer, sources,<br/>citation_verified, flagged_for_review)
    API-->>U: {answer, sources, citation_check, usage_log_id}
```

Failure modes explicitly handled at this layer (see `app/api/routes.py`):
Gemini `429` (quota exceeded) → `HTTP 429` with a clear detail message;
Gemini timeout → `HTTP 504`; `context_aware.rewrite_query` failure → falls
back to the original, unrewritten query rather than failing the whole
request.

---

## 3. Deployment topology (docker-compose.yml)

```mermaid
flowchart LR
    subgraph Host["Docker host"]
        API["ekie-api<br/>:8000"]
        PG["ekie-postgres<br/>:5432 (postgres:16-alpine)"]
        NEO["ekie-neo4j<br/>:17474 http / :7687 bolt<br/>(neo4j:5-community)"]
        CHROMA["ekie-chromadb<br/>:8001 (chromadb/chroma:0.5.23)"]
        MLFLOW["ekie-mlflow<br/>:5000 (mlflow:v2.16.2)"]
        PROM["ekie-prometheus<br/>:9090 (prom/prometheus)"]
    end
    Browser -->|":8000 (UI + API + /docs)"| API
    API -->|"depends_on: service_healthy"| PG
    API --> NEO
    API --> CHROMA
    API -.->|"eval logging"| MLFLOW
    PROM -->|"scrapes /metrics"| API
    Browser -.->|"direct browse, optional"| NEO
    Browser -.->|"direct browse, optional"| MLFLOW
    Browser -.->|"direct browse, optional"| PROM
```

All six services start with one `docker compose up --build`. `api`
declares `depends_on: postgres (service_healthy), neo4j (service_healthy),
chromadb (service_started)`; `app/main.py` additionally retries its own
Postgres schema init and Neo4j constraint setup on startup (10 attempts,
3s backoff) as a second safety net against boot-order races — see the
comments at the top of `app/main.py` for why both layers exist (Neo4j is
a JVM app and can take 10-20s past its own "started" log line before
accepting Bolt connections).

---

## 4. Why these technology choices (brief, full detail in README)

| Layer | Choice | Why |
|---|---|---|
| Vector store | ChromaDB | Simplest of the three suggested options (Chroma/Pinecone/Weaviate) to self-host with zero external account/API key, fits a 5-week case study. |
| Knowledge graph | Neo4j | Only graph DB in the suggested list; Cypher's pattern-matching fits `RELATES_TO`/`MENTIONS` traversal directly. |
| Keyword search | Postgres FTS (GIN index) | Avoids standing up a 4th datastore (Elasticsearch/OpenSearch) for search modes that overlap heavily with what Postgres already does well at this corpus size. |
| Reranking | Local cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Free, runs in-process, no added external dependency or per-call cost on top of the already-metered Gemini calls. |
| LLM | Google Gemini (free tier) | Zero-cost for a student project; trade-off is the 20-req/day/model quota that shows up repeatedly in the Week 4 evaluation work — documented, not hidden. |
| Eval tracking | MLflow | Suggested in the case study; low-overhead self-hosted run tracking, no external account. |

---

## 5. Related documents

- `docs/knowledge_graph_schema.md` — Knowledge Graph Design (node/edge schema, constraints, derived relationship inference, example Cypher).
- `docs/Database_Schema_Diagram.md` — PostgreSQL ERD (7 tables).
- `docs/Integration_Test_Report.md` — how these components were verified to actually talk to each other correctly.
- `README.md` — Quickstart, full scope-decision rationale, known limitations.
