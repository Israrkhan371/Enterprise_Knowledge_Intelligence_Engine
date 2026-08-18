"""
Exports the live OpenAPI schema without needing the app fully running
(mocks out the Postgres/Neo4j startup side effects in app.main so this
works even outside the container, as long as all packages are installed).
Run inside the container where all deps are already present:
    docker exec ekie-api python /app/scripts/export_openapi.py
"""
import json
import sys

sys.path.insert(0, "/app")

from app.main import app  # noqa: E402

schema = app.openapi()
with open("/app/API_Documentation_Swagger.json", "w") as f:
    json.dump(schema, f, indent=2)

print(f"Exported {len(schema.get('paths', {}))} paths to API_Documentation_Swagger.json")
