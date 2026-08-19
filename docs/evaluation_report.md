# EKIE Evaluation Report

**Case study:** AI-007 — Enterprise Knowledge Intelligence Engine
**Maps to:** Evaluation Report (Week 4, Mon, Track B)
**Status of this document:** Retrieval accuracy section is final, from a
completed live run (Week 4 Mon, Track A). Citation accuracy section now
covers **13 of 40 queries scored** across three runs (Aug 18–19) — real
progress, but still well short of full coverage. 27 queries have not yet
succeeded (8 in the 0–19 range, 19 in the 20–39 range), most recently
blocked by a batch of DNS resolution failures unrelated to quota or code
(see § 4). Numbers below are real and combined across runs, not
placeholders, but should still be treated as an early read pending fuller
coverage.

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

*Three runs now, Aug 18–19, across two disjoint offsets. The 0–19 range
was run twice (8 scored Aug 18, then 12 scored on a rerun Aug 19 — the
rerun's file overwrote the first, so 12 is the current count for that
range, not 8+12). The 20–39 range was attempted once and mostly hit a
batch of DNS resolution failures (`Name or service not known` — see § 4),
leaving only 1 of 20 scored there. Combined, unique coverage is now
**13 of 40 queries (33%)** — real progress, but 27 queries (8 in 0–19,
19 in 20–39) still haven't succeeded even once.*

| Metric | Value |
|---|---:|
| Queries scored (unique, combined) | 13 / 40 |
| — 0–19 range | 12 / 20 (latest rerun; 8 still unscored) |
| — 20–39 range | 1 / 20 (mostly blocked by DNS failures; 19 still unscored) |
| Answers with ≥1 citation | 12 / 13 |
| Answers with no citations | 1 |
| Citation accuracy rate (of cited answers) | **0.167** (2 of 12, combined) — see analysis below |
| Flag: citation number not in sources | 0 |
| Flag: claim not supported by cited source | 65 combined (61 from 0–19, 4 from 20–39) |

**Analysis:** the original 8-scored batch (37 flags) was inspected at the
sentence level and traced to a formatting/threshold artifact, not real
hallucination — see the detailed breakdown below, still accurate for that
data. The 20–39 range's single scored answer (4 flags) is consistent with
that same finding: its flagged similarity scores were **0.51, 0.51, 0.55,
0.55** — all clustered right at the 0.55 cutoff, none in the "clearly
wrong" range. **The newest 0–19 rerun's 61 flags have not yet been
inspected at the sentence level** (the full per-query file wasn't
available for direct analysis this round) — worth checking whether the
same pattern holds before drawing conclusions from the combined 0.167
rate, the same way `docs/citation_accuracy_results_limit20.json` was
checked previously.

**Original per-query analysis (Aug 18 batch, 8 scored / 37 flags)** — the
37 flags were not evenly spread: **26 of 37 (70%) come from a single
query**, *"How do I set up the internship onboarding checklist?"* Its
answer is a long bulleted checklist where the model tags **every bullet,
regardless of length, with both source citations** (`[1], [2]` repeated
on each line). Each short one-line bullet then gets checked against the
*entire* source chunk (which covers the whole multi-item checklist) — a
one-line item can't fully match a large multi-topic chunk, so its
similarity structurally lands just under the threshold. Sample scores
from that query: 0.46, 0.46, 0.47, 0.47.

This pattern held across all 37 of that batch's flags: none scored below
0.2 (where a citation would look genuinely unrelated to its source) and
29 of 37 sat in the 0.40–0.54 band, clustered just under the 0.55 cutoff
— consistent with the 20–39 range's 4 flags above. This reads as **the
0.55 threshold being too strict for list-formatted, multi-citation
answers**, not the model citing unsupported claims, across every batch
inspected so far.

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
- **Citation accuracy results reflect a handful of partial runs
  (n=13 combined), not variance across many runs.** Gemini generation is
  non-deterministic; each run of `check_citation_accuracy.py` is a
  snapshot, not a confidence interval, and n=13 is still too small to
  treat 0.167 as a stable rate rather than a developing signal.
  Completing the remaining 27 queries (and re-running on separate days)
  before the Thursday writeup would strengthen this section considerably.
- **The free-tier Gemini quota (20 requests/day/model) limits how much
  live evaluation can happen per day**, on both this check and ordinary
  `/ask` testing sharing the same quota. This is an account/billing
  constraint, not a code defect — see § 3's "Running the citation
  accuracy check" for the `--offset`/`--limit` workaround.
- **Gemini's own server-side 503 overload errors are a real, separate
  source of missing data**, distinct from quota exhaustion — 12 of 20
  queries in the first Aug 18 attempt failed this way. These are worth
  retrying (transient, not a hard cap like quota), but until retried they
  leave real gaps in coverage, not just quota-driven ones.
- **DNS resolution failures blocked most of the Aug 19 20–39-range run**
  (`Failed to resolve 'generativelanguage.googleapis.com'` — 19 of 20
  queries in that batch). This is a container/network-level failure, not
  a quota, code, or Gemini-side issue — the request never left the
  container. Likely causes: Docker's embedded DNS resolver having a
  transient hiccup, or network instability around the time of that run
  (possibly related to switching `GOOGLE_API_KEY`/project and recreating
  the container beforehand). If it recurs, check
  `docker compose exec api getent hosts generativelanguage.googleapis.com`
  before assuming it's a Gemini-side problem, and consider retrying after
  a plain `docker compose restart api` (no `--force-recreate` needed) to
  rule out a stuck network namespace.

## 5. Recommendations

- **The threshold/formatting-artifact finding (originally from the Aug 18
  n=8 batch) holds up against the 20–39 range's data too** — its 4 flags
  scored 0.51, 0.51, 0.55, 0.55, the same near-threshold clustering, none
  in the "clearly wrong" range. Still not confirmed against the newest
  0–19 rerun's 61 flags, which haven't been inspected at the sentence
  level yet — do that next, the same way `citation_accuracy_results_limit20.json`
  was checked before, before treating the combined 0.167 rate as settled.
- Two concrete follow-ups from the threshold finding, either or both
  worth doing:
  - Consider whether `verify_citations()` should check a bulleted list
    item against just the relevant portion of a multi-item source chunk
    rather than the whole chunk, since a one-line item can't fully match
    a large multi-topic source by design.
  - Re-run a small sensitivity check with the threshold lowered (e.g. to
    0.45) against the flagged answers collected so far to see how many
    would clear — informs whether 0.55 needs adjusting generally or just
    for list-style answers specifically.
- **27 queries still need to run at least once**: 8 remaining in the
  0–19 range (some combination of `503`s and quota hits across the two
  attempts there), and 19 in the 20–39 range (mostly blocked by the DNS
  failures in § 4 — worth investigating that before just retrying, since
  a retry into the same broken network state will likely fail the same
  way).
- If a genuinely low-similarity pattern (scores well under 0.2) shows up
  once the newest 61-flag batch and the remaining queries are inspected,
  use `docs/citation_accuracy_results*.json`'s per-query breakdown (now
  including `cited_source_text` on each flag — via the admin
  answer-review queue, `GET /admin/answers?flagged_for_review=true`, or
  directly in the JSON files) to decide whether the fix belongs in
  retrieval, the system prompt, or the threshold.
- Consider tracking an approximate "relevant chunk count" per
  `eval_set.json` entry so precision can be measured at a `k` matched to
  each query, rather than a fixed `k=10` that structurally caps it.
