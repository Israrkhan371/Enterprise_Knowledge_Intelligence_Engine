import re

from app.search.rerank import get_reranker


def extract_cited_indices(answer_text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})


def verify_citations(answer_text: str, sources: list[dict], relevance_threshold: float = 0.0) -> dict:
    """
    Bonus: AI Citation Verification. For each [n] citation in the answer,
    checks that the sentence containing it is actually supported by the
    cited source chunk, flagging unsupported claims.

    Uses the same cross-encoder as app/search/rerank.py (jointly scores
    the sentence and source in one forward pass), not embed_texts()'s
    bi-encoder cosine similarity. Switched 2026-08-19 after manually
    reviewing all 37 flags a citation-accuracy eval run produced at
    similarity_threshold=0.55 on the old bi-encoder approach: every
    single one was a false positive, including a near-verbatim match
    ("use retrieved_ids instead of lst2" appearing almost word-for-word
    in its source, similarity 0.44) and a fully accurate one-sentence
    summary of a longer source paragraph scoring only 0.19 - lower than
    every genuinely-flaggable case we could find. A bi-encoder embeds the
    sentence and source independently and compares the resulting vectors,
    which systematically under-scores accurate *summaries* (a compressed
    paraphrase sits far from its longer source in vector space even when
    it's a correct compression) and can't use one text to help interpret
    the other. A cross-encoder attends to both texts together in a single
    pass, which is exactly why rerank.py already uses one instead of
    semantic_search()'s bi-encoder for query-vs-chunk relevance.

    IMPORTANT: relevance_threshold's default of 0.0 is an *unverified
    starting guess*, not a calibrated value - ms-marco-MiniLM-L-6-v2's
    predict() returns raw unbounded logits (no sigmoid applied), typically
    very negative for irrelevant pairs and positive for relevant ones, but
    the actual cutoff that separates "genuinely unsupported" from
    "genuinely supported" for THIS task (short claim vs. longer source
    chunk, not the query-vs-chunk relevance this model was trained for)
    has not been empirically checked the way the old bi-encoder threshold
    was. Re-run the citation-accuracy eval, then inspect_citation_flags.py
    against a fresh batch of results before trusting this number - same
    process, same reason, as the fix that replaced the 0.55 bi-encoder
    threshold in the first place.

    Each flag dict carries machine-readable "kind" ("no_matching_source"
    or "low_relevance"), "cited_source_index", and "score" (None for
    no_matching_source) fields, not just the human-readable "issue"
    string. Downstream code (flag categorization, threshold inspection)
    should read those fields directly rather than parsing "issue" -
    substring-matching free text against wording this docstring's own
    history changed once already (the old bi-encoder version said "low
    similarity"; this version says "low relevance") is exactly the bug
    class that silently broke check_citation_accuracy.py's flag_breakdown
    and inspect_citation_flags.py's filter after this rewrite.
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
                flags.append({
                    "sentence": sentence,
                    "kind": "no_matching_source",
                    "cited_source_index": idx,
                    "score": None,
                    "issue": f"citation [{idx}] has no matching source",
                })
                continue

            score = float(get_reranker().predict([(sentence, source["text"])])[0])
            if score < relevance_threshold:
                flags.append({
                    "sentence": sentence,
                    "kind": "low_relevance",
                    "cited_source_index": idx,
                    "score": score,
                    "issue": f"low relevance ({score:.2f}) to source [{idx}]",
                })

    return {"cited_sources": cited, "flags": flags, "verified": len(flags) == 0}
