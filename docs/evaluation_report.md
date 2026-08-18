# EKIE Evaluation Report

**Case study:** AI-007 — Enterprise Knowledge Intelligence Engine
**Maps to:** Evaluation Report (Week 4, Mon, Track B)
**Status of this document:** Retrieval accuracy section is final, from a
completed live run (Week 4 Mon, Track A). Citation accuracy section has a
first real partial result (n=8 of 40, Aug 18) — not a placeholder
anymore, but not complete either. The 12 queries that failed with `503`
in that run should be retried, and the remaining ~20 queries in the eval
set still need to run, before this section's numbers and conclusions are
treated as final for Thursday's writeup.

## 1. Scope

Two independent questions, evaluated separately because they measure
different failure modes:

1. **Retrieval accuracy** — when someone asks a question, does hybrid
   search surface the documents that actually answer it?
2. **Citation accuracy** — when the RAG pipeline generates an answer
   *from* those retrieved chunks, does every claim it cites actually say
   what the citation claims it says?

A system can score well on one and poorly on the other: perfect retrieval
with a hallucinating generation step is still untrustworthy, and accurate
citation of badly-retrieved chunks just means confidently citing the wrong
source.

Both evaluations run against the same 40-query labeled set
(`app/evaluation/eval_set.json`), spanning all 14 required knowledge
source categories plus multi-document comparison and gap-detection
queries — see [README.md § Evaluation query set](../README.md#evaluation-query-set)
for how that set is built and kept valid across corpus reseeds.

## 2. Retrieval accuracy

**Method:** `app/evaluation/eval.py::run_evaluation()`. For each of the 40
eval queries, the query's `relevant_document_titles` are resolved against
the live `documents` table, the query is run through `hybrid_search()`
(RRF fusion of semantic + keyword search, cross-encoder reranked) with
`top_k=10`, and precision/recall/reciprocal-rank are computed against the
resolved relevant-document set. Queries whose target document hasn't been
ingested are skipped and counted separately rather than scored as 0 (a
missing fixture document isn't a retrieval failure — see the function's
docstring).

**Results** (`docker compose exec api python -m app.evaluation.eval`,
logged to MLflow experiment `ekie-retrieval-eval`):

| Metric | Before corpus seeding | After corpus seeding (current) |
|---|---:|---:|
| precision@10 | 0.087 | **0.097** |
| recall@10 | 0.795 | **0.897** |
| MRR | 0.581 | **0.608** |
| queries scored | — | 39 / 40 (1 correctly skipped — see note below) |

**Interpretation:** precision@10 = 0.097 next to recall@10 = 0.897 looks
contradictory at a glance but is the expected shape here, not a red flag.
Most eval queries have only 1-2 truly relevant chunks in the corpus, but
`top_k=10` always fills all 10 slots — so even a search that finds every
relevant chunk (high recall) is structurally capped at low precision,
because 8-9 of those 10 slots are necessarily non-relevant padding. Recall
is the more meaningful of the two numbers for this corpus size; precision
would only become a fair comparison at a `k` closer to the true number of
relevant chunks per query (which varies per query and isn't currently
tracked in `eval_set.json`).

The 1 skipped query is the deliberate gap-detection entry
(`"SOP - Intern Offboarding"`, a document that must not exist — see
`tests/test_seed_corpus.py`'s `_EXPECTED_UNSEEDED_TITLES`), not a fixture
bug.

## 3. Citation accuracy

**Why this is separate from retrieval accuracy:** `POST /ask` doesn't just
retrieve chunks — it asks Gemini to generate a natural-language answer
that cites them (`app/rag/generate.py`), then independently checks each
citation (`app/rag/citation_check.py::verify_citations()`): for every
`[n]` in the answer, the sentence containing it is embedded alongside the
cited source chunk, and flagged if their cosine similarity falls below
0.55. That check runs on every live `/ask` call and populates
`UsageLog.citation_verified` — the admin answer-review queue at
`GET /admin/answers?flagged_for_review=true` surfaces whatever it flags.
This evaluation asks the same question `verify_citations()` asks, but in
aggregate across a representative query set, rather than one answer at a
time.

**Method:** `scripts/check_citation_accuracy.py` (new this pass, see
`tests/test_check_citation_accuracy.py` for coverage — 4 tests, mocked).
For each of the 40 eval queries: run `generate_answer()` (the same
function `/ask` calls) to get a real generated answer + its cited source
chunks, run `verify_citations()` on the result, and aggregate.

**A deliberate departure from the raw `verified` flag:** `verify_citations()`
defaults `verified: True` when an answer contains zero `[n]` citations —
correct for its actual job (nothing to flag as *unsupported*), but wrong
as a headline accuracy number, since it would let an uncited, unhelpful
answer count as "verified" and flatter the result. The script instead
reports:
- `num_answers_with_no_citations` — answers that cited nothing at all
  (a completeness problem, distinct from an accuracy problem)
- `citation_accuracy_rate` — computed only over answers that *did* cite
  something: of those, what fraction had every citation check out

### Running the citation accuracy check

Requires the live stack (same requirements as `/ask` itself — Postgres,
ChromaDB, Neo4j, a working `GOOGLE_API_KEY`):

```bash
docker compose exec api python scripts/check_citation_accuracy.py
```

**Gemini free-tier quota constraint:** the Gemini API free tier caps
requests at 20/day per model
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). All 40 eval queries
in one run exceed that on their own, before counting any other `/ask`
traffic that day — a full run on a free-tier key will fail most or all
queries with `429 RESOURCE_EXHAUSTED`. Use `--offset`/`--limit` to split
the set across multiple days:

```bash
docker compose exec api python scripts/check_citation_accuracy.py --limit 15
docker compose exec api python scripts/check_citation_accuracy.py --offset 15 --limit 15
docker compose exec api python scripts/check_citation_accuracy.py --offset 30
```

Each `--offset`/`--limit` combination writes its own results file
(`docs/citation_accuracy_results_offset{N}_limit{M}.json`) rather than
overwriting the full run's file, so partial runs can be combined by hand.
The summary also reports `num_errored_quota_exhausted` separately from
other errors — if that number is high, the fix is waiting for the daily
reset or requesting quota, not debugging the pipeline.

This prints a summary to stdout, writes the full per-query breakdown to
`docs/citation_accuracy_results.json`, and logs the aggregate metrics to
MLflow under a new `ekie-citation-eval` experiment (kept separate from
`ekie-retrieval-eval` — different thing being measured).

### Results

*A first partial live result landed Aug 18 — 8 of 20 queries scored (the
other 12 in that batch failed with `503 UNAVAILABLE`, Gemini's servers
being overloaded, not a quota issue — `num_errored_quota_exhausted: 0`
confirms that). The remaining ~20-32 queries (the rest of the 40-query
set, plus retrying the 12 that hit 503s — those are worth a rerun since
`503` is transient server load, not a hard failure) should be run before
this table and the report's conclusions are treated as final. The numbers
below are real, not placeholders, but n=8 is a small sample — treat the
25% accuracy rate as an early, concerning signal worth investigating now
rather than a settled result.*

| Metric | Value |
|---|---:|
| Queries attempted / scored | 20 / 8 (12 failed: `503 UNAVAILABLE`, retry pending) |
| Answers with ≥1 citation | 8 / 8 |
| Answers with no citations | 0 |
| Citation accuracy rate (of cited answers) | **0.25** (2 of 8) |
| Flag: citation number not in sources | 0 |
| Flag: claim not supported by cited source | 37 (avg. ~4.6 per scored answer) |

**Early read, pending the fuller run:** every single flag is the same
type — a cited claim's sentence not matching its cited source closely
enough (similarity < 0.55), not a malformed citation number. Two
non-exclusive explanations worth checking against
`docs/citation_accuracy_results_limit20.json`'s per-query detail before
concluding either way:
- The model is genuinely citing loosely — pointing at a source that's
  topically related but doesn't actually say what the sentence claims.
- The 0.55 similarity threshold itself may be too strict for how Gemini
  paraphrases source text — a well-supported claim phrased very
  differently from the source's wording can fail a pure cosine-similarity
  check even when a human would consider it accurately cited (see § 4's
  "citation similarity check is a proxy, not entailment").

Distinguishing these needs eyeballing a handful of the 37 flagged
sentences against their cited source text directly, not just the
aggregate rate.

## 4. Known limitations

Carried over from Week 1-3 notes plus new limitations specific to this
evaluation:

- **Cross-document entity resolution** and **NER noise on spec-style
  documents** — flagged in Week 1 (rows 6/7 of the tracker), not fixed,
  not blocking. Affects graph-derived features (technology map, skill
  dependencies), not search or citation accuracy directly.
- **Precision@10 is structurally capped** by corpus size relative to
  `k=10`, per § 2 above — not a retrieval defect, but means precision@10
  alone shouldn't be read as "90%+ of results are noise."
- **The citation similarity check is a proxy, not entailment.** A
  cosine-similarity threshold on sentence-vs-source embeddings can pass a
  claim that merely shares vocabulary with its source without actually
  being supported by it, and can flag a legitimately-supported claim that
  happens to be phrased very differently from the source text. Treat
  `citation_accuracy_rate` as a signal for where to spot-check, not a
  guarantee.
- **Only cited claims are checked.** Neither `verify_citations()` nor this
  script detects an *uncited* factual claim that should have had a
  citation and didn't — that's what `num_answers_with_no_citations`
  partially surfaces (whole answers with zero citations), but a
  partially-cited answer with one unsupported uncited sentence next to
  three well-cited ones isn't caught at the sentence level.
- **Citation accuracy results in this report reflect one partial run
  (n=8), not variance across runs.** Gemini generation is
  non-deterministic; a single run of `check_citation_accuracy.py` is a
  snapshot, not a confidence interval, and n=8 is too small to treat 0.25
  as a stable rate rather than a starting signal. Completing the
  remaining ~32 queries (and re-running on a separate day) before the
  Thursday writeup would strengthen this section considerably.
- **The free-tier Gemini quota (20 requests/day/model) limits how much
  live evaluation can happen per day**, on both this check and ordinary
  `/ask` testing sharing the same quota. This is an account/billing
  constraint, not a code defect — see § 3's "Running the citation
  accuracy check" for the `--offset`/`--limit` workaround.
- **Gemini's own server-side 503 overload errors are a real, separate
  source of missing data**, distinct from quota exhaustion — 12 of the
  20 queries in the Aug 18 partial run failed this way. These are worth
  retrying (transient, not a hard cap like quota), but until retried they
  leave real gaps in coverage, not just quota-driven ones.

## 5. Recommendations

- **Investigate the 0.25 citation accuracy rate now, in parallel with
  completing the run** — don't wait for full coverage to start looking.
  Pull a handful of the 37 flagged sentences from
  `docs/citation_accuracy_results_limit20.json` and read them against
  their cited source chunk directly: if the cited source clearly doesn't
  support the claim, that's a real grounding problem in generation. If
  the source *does* support it but in different words, that points at the
  0.55 similarity threshold being miscalibrated rather than the model
  actually citing badly — a different, cheaper fix.
- Finish scoring the rest of the 40-query set (retry the 12 `503`
  failures, then cover the untried `--offset 20 --limit 20` range) before
  treating any conclusion here as final.
- If the low rate holds up across the fuller run, use
  `docs/citation_accuracy_results*.json`'s per-query breakdown (via the
  admin answer-review queue, `GET /admin/answers?flagged_for_review=true`,
  or directly in the JSON files) to decide whether the fix belongs in
  retrieval (bad chunks reaching generation), the system prompt (citing
  loosely), or the similarity threshold itself.
- Consider tracking an approximate "relevant chunk count" per
  `eval_set.json` entry so precision can be measured at a `k` matched to
  each query, rather than a fixed `k=10` that structurally caps it.
