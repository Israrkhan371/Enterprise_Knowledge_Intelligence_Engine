# Dense vs Sparse Retrieval - Research Notes

**Document type:** Research notes
**Relevant to:** EKIE's semantic (dense) and keyword (sparse) search,
fused via hybrid search.

## Sparse retrieval (keyword)

Represents text as a sparse vector over the vocabulary (term frequencies,
weighted by schemes like BM25 or Postgres's `ts_rank`). EKIE's
`app/search/keyword.py` uses Postgres full-text search
(`to_tsvector`/`plainto_tsquery`), backed by a GIN index.

**Strengths:**
- Exact term matches are guaranteed to be found — critical for queries
  containing specific identifiers, error codes, function names, or
  acronyms (e.g. searching for `call_with_timeout` or `SEV-2`).
- No embedding model dependency; cheap to compute and index.
- Interpretable — you can see exactly which terms matched.

**Weaknesses:**
- No notion of meaning: a query for "how do I fix a broken build" won't
  match a document that only says "resolving CI failures" unless the
  exact words overlap.
- Vulnerable to vocabulary mismatch between how a question is phrased
  and how the answer is written.

## Dense retrieval (semantic)

Represents text as a dense embedding vector (EKIE uses
sentence-transformers) and ranks by vector similarity (cosine
similarity, via ChromaDB). EKIE's `app/search/semantic.py` implements
this.

**Strengths:**
- Captures meaning, not just surface wording — synonyms, paraphrases,
  and related concepts can match even with zero shared vocabulary.
- Generally better for natural-language questions ("how do I..." /
  "what's the difference between...").

**Weaknesses:**
- Can miss exact-term queries: an embedding model may not weight a rare
  identifier or acronym heavily enough to surface the one document that
  contains it verbatim.
- Embedding quality is model-dependent; results degrade on
  domain-specific jargon the model wasn't trained on.
- More expensive to compute and index than sparse methods.

## Why EKIE fuses both (hybrid search)

Because the two methods fail in different, largely non-overlapping ways
— sparse retrieval is strong on exact terms, dense retrieval is strong
on meaning — combining their rankings (see Reciprocal Rank Fusion -
Research Notes) recovers relevant documents that either method alone
would miss, rather than picking one and accepting its blind spots.

## Related documents

- Reciprocal Rank Fusion - Research Notes
