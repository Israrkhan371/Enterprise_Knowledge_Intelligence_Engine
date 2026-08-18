"""
Calls POST /ask in-process via FastAPI's TestClient, bypassing the network
stack and docker logging entirely — any unhandled exception prints straight
to this script's own stdout/stderr with a full traceback.

Usage: docker exec ekie-api python /app/scripts/debug_ask.py "your query here"
"""
import sys
import traceback

from fastapi.testclient import TestClient

from app.main import app

query = sys.argv[1] if len(sys.argv) > 1 else "What is Ezitech remote work policy for interns?"

client = TestClient(app, raise_server_exceptions=True)
try:
    resp = client.post("/api/v1/ask", json={"query": query})
    print("STATUS:", resp.status_code)
    print("BODY:", resp.text[:2000])
except Exception:
    print("UNCAUGHT EXCEPTION:")
    traceback.print_exc()
