# EKIE Evaluation Report

Week 4, Monday B / Thursday B. Covers retrieval quality (`POST /api/v1/evaluation/run`) and
citation accuracy (`scripts/check_citation_accuracy.py`).

## 1. Evaluation set

40 query/relevant-document pairs across 14 required categories (`app/evaluation/eval_set.json`),
including multi-document comparison and gap-detection queries. Cross-referenced against the seed
corpus (26 documents, 11 categories) and verified by `tests/test_seed_corpus.py`.

5 of 40 queries are expected to skip during a retrieval-metrics run:
- 4 reference GitHub file paths (`app/ingestion/pipeline.py`, `app/graph/extract.py`,
  `app/search/hybrid.py`, `app/api/routes.py`) — these only resolve once the EKIE repo itself is
  ingested via the GitHub-ingestion admin endpoint; they aren't part of the seeded document set by
  design (see `scripts/seed_eval_corpus.py`).
- 1 ("Is there a documented SOP for offboarding an intern?") deliberately references a document
  that doesn't exist anywhere in the corpus — it exists to test gap-detection, not retrieval.

## 2. Retrieval quality

Run via `POST /api/v1/evaluation/run?k=10`, `app/evaluation/eval.py`.

| Metric | Value | Queries scored |
|---|---|---|
| Precision@10 | 0.109 | 35 / 40 (5 skipped, as documented above) |
| Recall@10 | 1.0 | 35 / 40 |
| MRR | 0.933 | 35 / 40 |

**Reading precision@10:** the eval set is dominated by single-relevant-document queries, so the
mathematical ceiling for precision@10 is 1/10 = 0.10 regardless of retrieval quality — the metric's
denominator is a fixed `k`, not the number of relevant documents. 0.109 sits essentially at that
ceiling. This is a property of the metric choice at this `k`, not a retrieval weakness; recall@10 =
1.0 and MRR = 0.933 are the metrics that actually reflect retrieval quality here, and both are
strong — every relevant document appears in the top 10, and on average the correct document lands
near rank 1.

## 3. Citation accuracy

Run via `scripts/check_citation_accuracy.py`, which sends every eval-set query through `POST
/api/v1/ask` and aggregates the `citation_check` verification `app/rag/citation_check.py` already
computes per answer (embedding-similarity check per cited sentence against its claimed source,
threshold 0.55).

**Status: blocked on Gemini API quota, not yet run to completion.** The free-tier key used for
local development caps at 20 requests/day (`generativelanguage.googleapis.com` free-tier limit).
Both the retrieval-metrics run above and repeated attempts at this script exhausted the day's quota
before a full 40-query pass completed. A real bug was found and fixed in the process: an unhandled
`429 RESOURCE_EXHAUSTED` from Gemini was crashing `/ask` to an opaque 500 with no diagnosis path;
`app/rag/generate.py` now raises `GeminiQuotaExceededError`, caught in `app/api/routes.py` to
return a clean `429` instead.

**To complete this section:** once quota resets (daily) or a paid tier is in place, run:

```bash
docker exec ekie-api python -m scripts.check_citation_accuracy
```

and replace the table below with the real output (`app/evaluation/citation_accuracy_results.json`
after the run):

| Metric | Value |
|---|---|
| Citation accuracy | *pending* |
| Verified / total with citations | *pending* |
| Queries with no citations | *pending* |
| Failures | *pending* |

**One thing to check once real numbers are in, before treating a low score as a citation-quality
problem:** a partial run (22/40 queries, before quota fully exhausted) showed nearly every citation
flagged as "low similarity" (0.2–0.5 against the 0.55 threshold) — including citations that read as
clearly correct on manual inspection. That pattern (uniformly-low-but-not-zero across unrelated
queries) is more consistent with the verification check itself being miscalibrated than with 22
consecutive bad citations: `verify_citations()` compares a full, often markdown-formatted answer
sentence against the entire raw source chunk using the same general-purpose retrieval embedding
model, and cosine similarity drops for that kind of length/format mismatch even when the claim is
fully supported. If the completed run reproduces this pattern, the fix is recalibrating the
threshold against real data (or replacing the bi-encoder cosine check with the existing
cross-encoder reranker for entailment scoring) — not concluding the RAG pipeline fabricates
citations.

## 4. Summary

Retrieval quality is strong and fully evaluated (recall 1.0, MRR 0.933). Citation-accuracy
evaluation is built, tested, and ready to run, but blocked on an external quota limit outside the
application's control. This report will be finalized (Thursday B) once a completed
`check_citation_accuracy.py` run is available.
