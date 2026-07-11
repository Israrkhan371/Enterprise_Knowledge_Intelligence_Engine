"""
Rewrites a follow-up query into a standalone query using recent conversation
turns, so retrieval doesn't lose context on pronouns/references
(e.g. "what about its dependencies?" -> "what are FastAPI's dependencies?").
Swap the stub below for a real LLM call (Claude via the Anthropic SDK).
"""


def rewrite_query(current_query: str, history: list[dict]) -> str:
    if not history:
        return current_query

    recent_turns = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
    prompt = (
        "Given this conversation history:\n"
        f"{recent_turns}\n\n"
        f"Rewrite this follow-up question as a standalone search query: '{current_query}'"
    )
    # TODO: call the LLM (see app/rag/generate.py for the client pattern)
    return current_query  # placeholder until wired to the LLM call
