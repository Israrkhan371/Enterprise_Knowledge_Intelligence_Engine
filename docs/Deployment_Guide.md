# EKIE — Deployment Guide

**Deliverable:** Deployment Guide (Case Study AI-007 deliverables list).
For local dev quickstart, see README "Quickstart" — this document covers
the same local path in more depth, plus what changes for a shared/staging
or production deployment. For step-by-step instructions to actually put
this on the public internet at no cost, see
`docs/FREE_DEPLOYMENT_GUIDE.md`.

## 1. Local development (fully supported, this is what's been tested)

```bash
git clone <repo-url> && cd Enterprise_Knowledge_Intelligence_Engine
cp .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY` (free at
https://aistudio.google.com/apikey — required for `/ask`, document
comparison, summarization, context-aware rewriting, and LLM cross-checks
in outdated-knowledge detection; every other endpoint works without it).

```bash
docker compose up --build
```

This starts six containers: `ekie-api` (FastAPI, builds from
`docker/Dockerfile`), `ekie-postgres`, `ekie-neo4j`, `ekie-chromadb`,
`ekie-mlflow`, `ekie-prometheus`. `api` waits on Postgres and Neo4j
healthchecks (`condition: service_healthy`) before starting, so a fresh
`docker compose up` won't race the database still initializing.

On first boot, `app/main.py` runs `Base.metadata.create_all()` (new
tables) then Alembic migrations (`command.upgrade(..., "head")`) for
schema changes to existing tables, and `GraphStore.init_schema()` for the
Neo4j constraints/indexes — all idempotent, safe on every restart.

Create an admin user (needed for every `/admin/*` endpoint — see README
"Frontend" for why this is a bare header check, not real auth, and why
that's an accepted trade-off for local use only):

```bash
docker exec -it ekie-postgres psql -U ekie -d ekie -c \
  "INSERT INTO users (id, email, name, role) VALUES (gen_random_uuid(), 'admin@ezitech.test', 'Admin', 'admin') RETURNING id;"
```

Then open:

| Service | URL |
|---|---|
| Frontend + API | http://localhost:8000/ |
| Swagger UI | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:17474 |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |

Verify the stack: `curl http://localhost:8000/health` should return
`{"status": "ok", "version": ...}`. If it doesn't match what you expect
from the source, rebuild (`docker compose up --build`, or `docker compose
build --no-cache api` if Compose is caching a layer you don't want).

Run the full test suite inside the container (not on the host — it needs
the live Postgres/Neo4j the container has):

```bash
docker exec ekie-api pytest tests/ -v
```

Run the new end-to-end integration check (see `docs/Integration_Test_Report.md`):

```bash
pip install httpx   # one-time, on the host — this script runs from outside the container
python scripts/e2e_integration_test.py
```

### Seeding demo data

`seed_data/` ships one real document per source type. To load it (and the
40-query evaluation set alongside it):

```bash
docker exec ekie-api python -m scripts.seed_eval_corpus
```

## 2. Environment variables reference

All variables are documented with defaults in `.env.example`; the ones
worth calling out specifically:

| Variable | Purpose | Notes |
|---|---|---|
| `DATABASE_URL` | Postgres connection string | Container-to-container hostname (`postgres`) baked in for Compose — change only if you're running Postgres elsewhere |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j bolt connection | Must match `docker-compose.yml`'s `NEO4J_AUTH` |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_COLLECTION` | Vector store | `CHROMA_PORT=8000` is the **container-internal** port even though it's mapped to host `8001` — don't change this to 8001 |
| `GOOGLE_API_KEY` | Gemini access | Leave blank to run without LLM-backed features; free tier is 20 req/day/model (see `docs/Fixing_Gemini_Quota.md`) |
| `GEMINI_MODEL` | Model name | `gemini-flash-latest` — an earlier default (`gemini-2.5-flash`) was rejected by the live API; keep this in sync with what Google's API currently accepts |
| `EMBEDDING_MODEL` | sentence-transformers model | `all-MiniLM-L6-v2` (384-dim); changing this requires re-embedding every existing chunk and updating the Chroma collection dimension |
| `MLFLOW_TRACKING_URI` | MLflow server | Container-to-container hostname (`mlflow`) |

## 3. What a staging/production deployment would additionally need

This case study's scope stops at a fully working local Docker Compose
stack (per the case-study brief's own suggested-technology list, which
lists Kubernetes as *optional*). The following gaps are real and should be
treated as a roadmap, not silently glossed over:

| Area | Current state | What production needs |
|---|---|---|
| **Auth** | `X-User-Id` header checked against `users.role` — no password, no session, no token expiry | Real auth (OAuth2/JWT, or an identity provider) before this touches a shared network |
| **Secrets** | Real passwords/API keys now live in a server-side `.env` (`docker-compose.yml`'s `${POSTGRES_PASSWORD:-ekie_password}`-style defaults mean nothing sensitive needs to be *committed* — see `docs/FREE_DEPLOYMENT_GUIDE.md`), but it's still a plaintext file on one server, not a managed secret store | A real secrets manager (Vault, AWS/GCP secret manager) for anything beyond a single-server deployment |
| **Ingestion** | Synchronous — upload blocks until chunk/embed/graph-populate finish (see `docs/AI_Architecture_Diagram.md` "Known architectural trade-offs") | Background job queue (Celery/RQ/Arq) with an async `processing` status |
| **Scaling** | Single `api` container, single Postgres/Neo4j/Chroma instance each | Horizontal scaling for `api` behind a load balancer (stateless — fine to scale); managed/clustered Postgres and Neo4j; ChromaDB's clustered mode or a managed vector DB (Pinecone/Weaviate, both already listed as alternatives in the case-study brief) for HA |
| **TLS** | Solved for a single-server deployment — `docker-compose.prod.yml` adds a `caddy` reverse proxy that automatically obtains and renews free Let's Encrypt certificates (see `docs/FREE_DEPLOYMENT_GUIDE.md`); backend services (Postgres, Neo4j, Chroma, MLflow, Prometheus, and `api` itself) no longer bind to the host at all, only reachable over Docker's internal network | A managed load balancer / ingress controller for anything beyond one server |
| **Backups** | Named Docker volumes only (`pg_data`, `neo4j_data`, `chroma_data`, `mlflow_data`) — durable across `docker compose down` but not across host loss | Scheduled `pg_dump`/Neo4j dump/Chroma persistence-dir backups to off-host storage |
| **Orchestration** | Docker Compose (single host) | Kubernetes (optional per the brief) for multi-host scheduling, rolling deploys, and resource limits — the existing container boundaries map directly onto Deployments/StatefulSets, so this is additive, not a redesign |
| **Rate limiting / quota** | None on the API itself; the only quota is Gemini's own free-tier limit | Request rate limiting per user/API key, and a paid Gemini tier (or a self-hosted LLM) once real traffic exceeds 20 req/day/model |
| **Observability** | Prometheus scraping `/metrics` (now blocked from public access by `docker/Caddyfile`'s `respond /metrics 404` rule — Prometheus still reaches it directly over the internal network); no alerting, no log aggregation | Alertmanager rules on top of the existing Prometheus setup; centralized log shipping (the Dockerfile already sets `PYTHONUNBUFFERED=1` so container logs are usable as-is) |

## 4. Rollback / recovery

- **Bad migration:** Alembic migrations are additive/idempotent in this
  repo (see `app/main.py`'s comments on `create_all()` vs. Alembic) — to
  roll back a specific revision, `docker exec ekie-api alembic downgrade
  -1` from inside the container.
- **Corrupted vector index / graph:** ChromaDB and Neo4j data live in
  named volumes (`chroma_data`, `neo4j_data`); `docker compose down -v`
  wipes them for a clean re-seed via `scripts/seed_eval_corpus.py`, or
  restore from a backup taken per the table above.
- **Stale running container vs. new source:** rebuild rather than
  restart — `docker compose up --build` (or `docker compose build
  --no-cache api` if layer caching is masking a change);
  `GET /health`'s `version` field is the fastest way to confirm the
  running container actually matches the source you expect.
