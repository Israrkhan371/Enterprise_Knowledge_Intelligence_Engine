import re

from app.embeddings.embedder import embed_texts


def extract_cited_indices(answer_text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})


def verify_citations(answer_text: str, sources: list[dict], similarity_threshold: float = 0.55) -> dict:
    """
    Bonus: AI Citation Verification. For each [n] citation in the answer,
    checks that the sentence containing it is actually semantically close
    to the cited source chunk, flagging unsupported claims.
    """
    cited = extract_cited_indices(answer_text)
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)

    flags = []
    for sentence in sentences:
        indices = extract_cited_indices(sentence)
        if not indices:
            continue
        for idx in indices:
            source = next((s for s in sources if s["index"] == idx), None)
            if not source:
                flags.append({"sentence": sentence, "issue": f"citation [{idx}] has no matching source"})
                continue

            sent_vec, src_vec = embed_texts([sentence, source["text"]])
            similarity = sum(a * b for a, b in zip(sent_vec, src_vec))  # cosine, vectors are normalized
            if similarity < similarity_threshold:
                flags.append({
                    "sentence": sentence,
                    "issue": f"low similarity ({similarity:.2f}) to source [{idx}]",
                })

    return {"cited_sources": cited, "flags": flags, "verified": len(flags) == 0}
