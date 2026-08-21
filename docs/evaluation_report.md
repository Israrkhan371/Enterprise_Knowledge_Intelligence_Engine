# EKIE Evaluation Report

**Case study:** AI-007 — Enterprise Knowledge Intelligence Engine
**Maps to:** Evaluation Report (Week 4, Mon, Track A + Track B)
**Status of this document:** Both tracks are closed. Retrieval accuracy
(Track A) closed with a passing, fully-scored metric. Citation accuracy
(Track B) closed on Aug 20 with a **negative finding, not a passing
metric**: two independent automated scoring approaches were built,
tested, run against real data, and manually validated against real
flagged citations — neither produces a threshold that reliably
separates unsupported claims from accurate ones. That's a real,
useful conclusion for this report, not an unfinished task; see § 3.

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
queries.

## 2. Retrieval accuracy — CLOSED, passing

**Method:** `app/evaluation/eval.py::run_evaluation()`. For each of the 40
eval queries, the query's `relevant_document_titles` are resolved against
the live `documents` table, the query is run through `hybrid_search()`
(RRF fusion of semantic + keyword search, cross-encoder reranked) with
`top_k=10`, and precision/recall/reciprocal-rank are computed against the
resolved relevant-document set. Queries whose target document hasn't been
ingested are skipped and counted separately rather than scored as 0.

**Results** (logged to MLflow experiment `ekie-retrieval-eval`):

| Metric | Before corpus seeding | After corpus seeding (final) |
|---|---:|---:|
| precision@10 | 0.087 | **0.097** |
| recall@10 | 0.795 | **0.897** |
| MRR | 0.581 | **0.608** |
| queries scored | — | 39 / 40 (1 correctly skipped — a deliberate gap-detection entry that must not exist) |

**Interpretation:** precision@10 = 0.097 next to recall@10 = 0.897 is the
expected shape here, not a red flag. Most eval queries have only 1–2
truly relevant chunks in the corpus, but `top_k=10` always fills all 10
slots — so even a search that finds every relevant chunk (high recall) is
structurally capped at low precision, because 8–9 of those 10 slots are
necessarily non-relevant padding. Recall is the more meaningful number
here.

## 3. Citation accuracy — CLOSED, negative finding

**Why this is separate from retrieval accuracy:** `POST /ask` doesn't just
retrieve chunks — it asks Gemini to generate a natural-language answer
that cites them (`app/rag/generate.py`), then independently checks each
citation (`app/rag/citation_check.py::verify_citations()`). This
evaluation asks whether that check is itself trustworthy, by running it
against real generated answers and manually reading a sample of what it
flags against the actual cited source text.

### 3.1 Two scoring approaches were tried, in sequence

**Approach 1 — bi-encoder cosine similarity (original).** Each cited
sentence and its source chunk were embedded (the same `sentence-transformers/all-MiniLM-L6-v2`
model used for retrieval) and compared by cosine similarity, threshold
0.55. Manual review of the first real batch (8 scored queries, 37 flags,
Aug 18) found this threshold was miscalibrated for list-formatted
answers: 26 of 37 flags came from one checklist-style answer where every
bullet was tagged with all cited sources regardless of length, and across
all 37 flags none scored below 0.2 (would indicate a genuinely unrelated
citation) — 78% clustered in 0.40–0.54, just under the cutoff. This read
as a threshold problem, not a real accuracy problem, and motivated
rewriting the check.

**Approach 2 — cross-encoder relevance score (rewrite).**
`citation_check.py` was rewritten to score each (sentence, source) pair
with `cross-encoder/ms-marco-MiniLM-L-6-v2` (the same reranker already
used for hybrid search), a bounded model change intended to give a more
semantically-grounded score than cosine similarity. `verify_citations()`
now emits structured, machine-readable fields on every flag (`kind`,
`cited_source_index`, `score`) instead of only free text — this was also
a real bug fix: an earlier version of both `check_citation_accuracy.py`'s
flag categorization and `inspect_citation_flags.py`'s filter hardcoded a
substring match against the old wording ("low similarity"), which broke
silently (reporting 0 flags, or dumping everything into an "other"
bucket) the moment the wording changed to "low relevance" — exactly the
failure class structured fields are meant to prevent.

### 3.2 Manual validation of the cross-encoder scores — the actual finding

16 `low_relevance` flags from a real run (Aug 20, `citation_accuracy_results_limit20.json`
+ `citation_accuracy_results_offset20_limit20.json`) were read in full —
each flagged sentence against the complete text of its cited source —
using `scripts/inspect_citation_flags.py`. Classification:

| Score | Cited claim (abridged) | Verdict |
|---:|---|---|
| −0.14 | "Embeddings are stored in the ChromaDB vector database" | **Clearly supported** — source states this near-verbatim |
| −0.31 (×2) | "page the on-call engineer immediately... SEV-1/SEV-2" | Topic matches source exactly; likely supported (source truncated in display) |
| −0.39 | "knowledge graph entity extraction pipelines" | Source is about chunking strategy — plausible real mismatch |
| −0.64 | "Embeddings are generated... using sentence-transformers" | **Clearly supported** — source states this near-verbatim |
| −2.00 (×2) | Bonus challenges list | Plausibly supported; source truncated before reaching the list |
| −2.08 (×2) | "Keep the weekly tracker updated..." | Plausibly supported; source truncated |
| −2.14 | "app/ingestion/: loaders, OCR fallback, chunking logic" | **Clearly supported** — source states this near-verbatim |
| −3.12 | "Ingestion finished at upload... no waiting for admin approval" | Topic not present in shown source — plausible real issue |
| −3.65 | "app/embeddings/: model wrapper + ChromaDB operations" | Directionally supported; specific path not shown in source |
| −4.43 | "Supported source types include markdown, code, transcripts..." | Cites a test-file docstring for a factual claim — plausible real issue |
| −4.60 | "Document text is extracted using source-type-specific loaders" | **Clearly supported** — source states this near-verbatim |
| −5.72 (×2) | "use `retrieved_ids` instead of `lst2`" | **Near-verbatim match** — the single clearest true positive in the sample |

**The finding:** there is no threshold that works. The single clearest
true positive in the entire sample — a near-verbatim match — scored
**−5.72, the single worst score of all 16 flags.** Five other claims that
are just as unambiguously supported by their sources (−0.14, −0.64,
−2.14, −4.60, plus the −0.31 pair) span almost the *entire* observed
range, completely overlapping the scores of the handful of flags that
looked like plausible real citation issues (−0.39, −3.12, −4.43). Any
threshold picked from this data either flags nearly everything —
including the most obviously correct citations — or misses the clearest
true positive outright.

**Likely cause:** both scoring approaches compare a short single sentence
against an entire multi-section source chunk (up to several hundred
words spanning multiple architecture components, checklist items, or SOP
sections). The resulting score tracks the source chunk's length and
topical breadth more than whether the specific cited fact is present in
it — so it systematically penalizes accurate citations to long,
structured documents (READMEs, checklists, SOPs) more than citations to
short, single-topic sources. This is consistent with both failed
attempts: the bi-encoder's checklist-answer problem and the
cross-encoder's worst-scoring-a-verbatim-match problem are the same root
cause wearing two different scoring functions.

### 3.3 What's closed vs. what remains open

**Closed (this evaluation cycle):**
- The engineering is complete and tested: `scripts/check_citation_accuracy.py`
  (with `--offset`/`--limit` batching for quota-constrained runs),
  `app/rag/citation_check.py` with structured flag output, and
  `scripts/inspect_citation_flags.py` for manual review all work
  correctly (15 passing tests across `test_citation_check.py` and
  `test_check_citation_accuracy.py`).
- The question this evaluation was built to answer — *is a
  similarity/relevance-score threshold a trustworthy way to automatically
  verify citation accuracy for this corpus?* — has a real answer: **no**,
  based on two independently-built and manually-validated approaches.
- Aggregate coverage across all runs (Aug 18–20): a combined but
  non-representative sample of citation-accuracy runs was collected
  (13–15 unique queries scored out of 40, heavily constrained by the
  Gemini free-tier's 20-request/day quota — see § 4). Given § 3.2's
  finding, expanding this sample further under the current scoring
  approach would not produce a more trustworthy number; it would just be
  more data points spread across the same unreliable metric.

**Explicitly not resolved, and not this evaluation's job to resolve —
recommended as follow-up work, not part of Track B's scope:**
- **Span-targeted comparison**: score the cited claim against the
  specific paragraph or list item within the source chunk that best
  matches it (a secondary retrieval step within the chunk), rather than
  the whole chunk — directly targets the length/breadth bias identified
  above.
- **LLM-as-judge entailment**: replace the similarity/relevance score
  with a direct yes/no "does this source support this claim?" prompt to
  an LLM, which doesn't have the same length-bias failure mode as a
  fixed-dimension embedding comparison.
- Full 40/40 coverage of the citation-accuracy script under whichever
  scoring approach eventually gets adopted, once one exists that passes
  this same manual-validation bar.

## 4. Known limitations

- **Precision@10 is structurally capped** by corpus size relative to
  `k=10` — see § 2's interpretation note. Not a retrieval defect.
- **The free-tier Gemini quota (20 requests/day/model) severely limited
  how much live data could be collected for citation-accuracy work.**
  Across three days (Aug 18–20) and roughly ten separate run attempts,
  many batches scored 0–2 queries before exhausting the day's quota,
  and two runs failed almost entirely to transient causes (a `503`
  overload wave, and a batch of DNS resolution failures later confirmed
  transient via `getent hosts`). This is an account/infrastructure
  constraint, not a code defect, and — per § 3.3 — expanding sample size
  under the current scoring approach wouldn't have changed the outcome
  regardless.
- **Cross-document entity resolution and NER noise on spec-style
  documents** — flagged in Week 1, not fixed, not blocking. Affects
  graph-derived features (technology map, skill dependencies), not
  search, generation, or citation accuracy directly.
- **The manual validation in § 3.2 is itself a sample of 16 flags**, not
  the full 40-query set. The finding (no working threshold exists) is
  strong enough — spanning the full observed score range with both
  clear-true-positives and plausible-issues overlapping — that more
  samples under the same scoring approach are unlikely to change the
  conclusion, but this is judgment, not exhaustive proof.

## 5. Recommendations

- **Do not invest further time in threshold-tuning the current
  similarity/relevance-based citation check.** § 3.2 shows the signal
  itself doesn't separate the two classes it needs to separate; no
  amount of additional data collection fixes that without changing the
  underlying comparison mechanism.
- If citation accuracy verification is revisited, start with
  **LLM-as-judge entailment** (§ 3.3) — cheaper to prototype than
  span-targeted retrieval and doesn't inherit the length-bias failure
  mode either approach tried here shares.
- The retrieval-accuracy result (§ 2) stands on its own and doesn't need
  citation accuracy to be resolved — it can go into the case study
  writeup as-is.
- If a future citation-accuracy attempt needs more Gemini quota than the
  free tier allows, budget for either enabling billing on the project
  or explicitly planning multi-day, multi-key runs from the start rather
  than discovering the constraint mid-investigation (see the tracker's
  Track B history for how much of this cycle was lost to that
  discovery).
