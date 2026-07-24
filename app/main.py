from fastapi import FastAPI
from prometheus_client import make_asgi_app
import time

from neo4j.exceptions import ServiceUnavailable

from app.core.database import Base, engine
from app.graph.build import GraphStore
from app.api.routes import router as knowledge_router
from app.admin.routes import router as admin_router

app = FastAPI(
    title="Enterprise Knowledge Intelligence Engine (EKIE)",
    description="Central AI intelligence layer over Ezitech's organizational knowledge.",
    version="0.1.0",
)

# Auto-create tables on startup for local dev.
# Use Alembic migrations for anything beyond local dev.
Base.metadata.create_all(bind=engine)

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
    return {"status": "ok"}
