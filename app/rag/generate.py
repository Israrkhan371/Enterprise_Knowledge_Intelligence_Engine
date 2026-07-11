import anthropic

from app.core.config import settings
from app.search.hybrid import hybrid_search

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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

    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {query}",
        }],
    )

    answer_text = "".join(block.text for block in response.content if block.type == "text")

    return {
        "answer": answer_text,
        "sources": [
            {"index": i + 1, "document_id": hit.get("document_id", hit.get("id")), "text": hit["text"][:300]}
            for i, hit in enumerate(hits)
        ],
    }
