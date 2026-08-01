"""
Rewrites a follow-up query into a standalone query using recent conversation
turns, so retrieval doesn't lose context on pronouns/references
(e.g. "what about its dependencies?" -> "what are FastAPI's dependencies?").
"""
from google import genai
from google.genai import types

from app.core.config import settings

_client = genai.Client(api_key=settings.google_api_key)


def rewrite_query(current_query: str, history: list[dict]) -> str:
    if not history:
        return current_query

    recent_turns = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
    prompt = (
        "Given this conversation history:\n"
        f"{recent_turns}\n\n"
        "Rewrite the following follow-up question as a standalone search "
        "query that makes sense without the conversation history (resolve "
        "pronouns/references to what they actually refer to). Return ONLY "
        "the rewritten query text, nothing else.\n\n"
        f"Follow-up question: {current_query}"
    )

    try:
        response = _client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=200),
        )
        rewritten = (response.text or "").strip()
        return rewritten or current_query
    except Exception:
        # A rewrite failure is a degraded-but-recoverable situation, not a
        # reason to fail the whole /ask request — fall back to the
        # original (un-rewritten) query, same "never block on a non-
        # critical enhancement" pattern as _populate_graph() in
        # app/ingestion/pipeline.py for graph population failures.
        return current_query
