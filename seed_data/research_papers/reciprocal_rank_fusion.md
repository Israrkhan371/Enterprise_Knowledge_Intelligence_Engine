# Reciprocal Rank Fusion - Research Notes

**Document type:** Research notes
**Relevant to:** EKIE hybrid search (app/search/hybrid.py)

## What it is

Reciprocal Rank Fusion (RRF) combines multiple ranked result lists (e.g.
a semantic-search ranking and a keyword-search ranking) into a single
ranking, without needing the two lists' raw scores to be on comparable
scales — which matters here since cosine similarity (semantic) and a
BM25/tsvector rank (keyword) aren't directly comparable numbers.

## The scoring formula

For each document `d` that appears in one or more of the input rankings:

```
RRF_score(d) = sum over each ranking r that contains d of  1 / (k + rank_r(d))
```

where `rank_r(d)` is d's 1-indexed position in ranking `r`, and `k` is a
small constant (commonly 60) that dampens the influence of very
high-ranked results so the fusion isn't dominated by a single list's #1
result.

Documents are then sorted by their summed RRF score, descending.

## Why it works well for hybrid search

- **Rank-based, not score-based** — it only needs each ranking's order,
  not the underlying similarity metric, so semantic and keyword results
  fuse cleanly even though their score distributions are unrelated.
- **Rewards agreement** — a document ranked reasonably well by *both*
  semantic and keyword search outranks a document that only one method
  loved, which tends to surface genuinely relevant results over results
  that one method scored well by coincidence.
- **Simple and parameter-light** — only one tunable constant (`k`),
  versus weighted-sum fusion which needs a per-source weight that has to
  be tuned per corpus.

## Tradeoffs

- RRF ignores the *magnitude* of the underlying scores — a semantic
  match at 0.95 cosine similarity and one at 0.55 both just contribute
  "rank 1" if they're each first in their list. This can under-weight a
  very strong single-method match relative to two mediocre matches that
  happen to agree.
- Choice of `k` does matter at the margins; very small `k` makes rank 1
  dominate almost entirely, very large `k` flattens the fusion toward a
  simple sum of appearances.

## Related documents

- Dense vs Sparse Retrieval - Research Notes
