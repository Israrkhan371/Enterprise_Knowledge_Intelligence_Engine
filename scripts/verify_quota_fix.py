"""
Confirms the GeminiQuotaExceededError fix works end-to-end WITHOUT spending
real Gemini quota: monkeypatches the client to raise the same 429 ClientError
Google actually returned, then checks /ask responds 429 with a clean message
instead of crashing to 500.

Usage: docker exec ekie-api python -m scripts.verify_quota_fix
"""
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.main import app
import app.rag.generate as generate_module


def _raise_quota_exceeded(*args, **kwargs):
    # Bypass ClientError.__init__ entirely (it expects a real httpx/requests
    # Response object with .body_segments, not a plain dict — constructing
    # one just to trigger our except-clause isn't worth replicating the
    # SDK's real HTTP response shape). We only need an instance that IS-A
    # genai_errors.ClientError with .code == 429, since that's all
    # app/rag/generate.py's except clause actually checks.
    exc = genai_errors.ClientError.__new__(genai_errors.ClientError)
    exc.code = 429
    exc.message = "You exceeded your current quota, please check your plan and billing details."
    exc.status = "RESOURCE_EXHAUSTED"
    exc.details = {}
    raise exc


def main() -> None:
    client = TestClient(app, raise_server_exceptions=True)

    with patch.object(
        generate_module._client.models, "generate_content", side_effect=_raise_quota_exceeded
    ):
        resp = client.post("/api/v1/ask", json={"query": "test query"})

    print("STATUS:", resp.status_code)
    print("BODY:", resp.json())

    if resp.status_code == 429:
        print("\nPASS: quota error now returns a clean 429 instead of crashing to 500.")
    else:
        print(f"\nFAIL: expected 429, got {resp.status_code}.")


if __name__ == "__main__":
    main()
