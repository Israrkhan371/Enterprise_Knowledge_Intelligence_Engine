from google import genai
from google.genai import types

from app.core.config import settings
from app.rag.gemini_utils import call_with_timeout
from app.search.hybrid import hybrid_search

_client = genai.Client(api_key=settings.google_api_key)

SYSTEM_PROMPT = """You are Ezitech's Enterprise Knowledge Intelligence assistant.
Answer only using the provided source chunks. For every claim, cite the source
number in square brackets, e.g. [1]. If the sources don't contain the answer,
say so plainly instead of guessing."""


def build_context_block(hits: list[dict]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] (doc {hit.get('document_id', hit.get('id'))}): {hit['text']}")
    return "\n\n".join(lines)


def generate_answer(db, query: str, top_k: int = 6) -> dict:
    hits = hybrid_search(db, query, top_k=top_k)
    context = build_context_block(hits)

    response = call_with_timeout(
        _client.models.generate_content,
        timeout_seconds=settings.gemini_timeout_seconds,
        model=settings.gemini_model,
        contents=f"Sources:\n{context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1000,
        ),
    )

    answer_text = response.text or ""

    return {
        "answer": answer_text,
        # Full chunk text, not a truncated preview: this list is fed straight
        # into verify_citations() (which does sentence-vs-source similarity
        # checks) and stored on UsageLog for admin answer review. Clipping
        # it here previously made citation verification compare each
        # sentence against a truncated source, silently weakening the check
        # on any chunk longer than 300 characters. Callers that want a
        # lighter payload for display can truncate client-side.
        "sources": [
            {"index": i + 1, "document_id": hit.get("document_id", hit.get("id")), "text": hit["text"]}
            for i, hit in enumerate(hits)
        ],
    }
