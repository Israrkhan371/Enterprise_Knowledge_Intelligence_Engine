from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
import time

from alembic import command
from alembic.config import Config as AlembicConfig
from neo4j.exceptions import ServiceUnavailable
from sqlalchemy.exc import OperationalError

from app.core.database import Base, engine
from app.graph.build import GraphStore
from app.api.routes import router as knowledge_router
from app.admin.routes import router as admin_router

app = FastAPI(
    title="Enterprise Knowledge Intelligence Engine (EKIE)",
    description="Central AI intelligence layer over Ezitech's organizational knowledge.",
    version="0.2.0",
)


def _init_postgres_schema():
    # Auto-create brand-new tables on startup for local dev convenience.
    #
    # IMPORTANT: create_all() only creates TABLES that don't exist yet — it
    # does NOT alter existing tables to add columns a model gained later
    # (e.g. Document.supersedes_id/version). That gap is what let a running
    # Postgres volume silently drift out of sync with app/core/models.py.
    # Schema changes to already-existing tables must go through the Alembic
    # migration below, never through create_all() or manual ALTER TABLEs.
    Base.metadata.create_all(bind=engine)

    # Apply any pending Alembic migrations (e.g. new columns on tables that
    # already existed from a previous version of the schema). Safe/idempotent
    # on every startup: no-ops once the DB is already at head, and each
    # migration guards its own column-existence checks so it's also a no-op
    # on a freshly created table where create_all() above just created the
    # column already.
    alembic_cfg = AlembicConfig(str(Path(__file__).resolve().parent.parent / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")


# Retry with backoff: docker-compose.yml's healthcheck on the postgres
# service (condition: service_healthy) normally guarantees Postgres is
# actually accepting connections before api starts, so this loop shouldn't
# be hit under Compose. It's a safety net for running the API outside that
# ordering (e.g. restarting api on its own, or Postgres briefly recycling
# connections) - without it, a connection race here would crash the app
# before a single table gets created, same failure mode this whole retry
# pattern already guards against for Neo4j below.
_pg_max_attempts = 10
_pg_delay_seconds = 3
for _pg_attempt in range(1, _pg_max_attempts + 1):
    try:
        _init_postgres_schema()
        break
    except OperationalError:
        if _pg_attempt == _pg_max_attempts:
            raise
        time.sleep(_pg_delay_seconds)

# Auto-create Neo4j constraints/indexes on startup for local dev.
# Safe to run every time - Neo4j no-ops on existing constraints/indexes.
#
# Retry with backoff: Neo4j (a JVM app) can take 10-20+ seconds to start
# accepting Bolt connections after its container reports "started", so a
# bare first attempt here can race Neo4j's own boot and crash the API on
# a fresh `docker compose up`. docker-compose.yml also has a healthcheck
# so this normally isn't hit under Compose - this loop is a safety net for
# running the API outside that ordering (e.g. restarting api on its own).
_graph_store = GraphStore()
_max_attempts = 10
_delay_seconds = 3
try:
    for _attempt in range(1, _max_attempts + 1):
        try:
            _graph_store.init_schema()
            break
        except ServiceUnavailable:
            if _attempt == _max_attempts:
                raise
            time.sleep(_delay_seconds)
finally:
    _graph_store.close()

app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

# Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


# Frontend last: a Mount only catches paths not already matched by a route
# registered above it, so this can never shadow /api/v1/*, /health, or
# /metrics regardless of mounting it at "/" — but it must still be added
# after those routes for that ordering to hold.
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
