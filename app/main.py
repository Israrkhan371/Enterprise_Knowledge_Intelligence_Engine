from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.core.database import Base, engine
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

app.include_router(knowledge_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")

# Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok"}
