# seed_data/

Backing corpus for `app/evaluation/eval_set.json`. The eval set has 40
realistic Q&A pairs referencing 31 document titles, but the documents
themselves didn't exist anywhere — so every query silently resolved to
zero relevant documents and got skipped by `run_evaluation()` rather
than scored (see `app/evaluation/eval.py`'s skip-not-zero design).

This directory has one real, content-complete file per referenced title,
in the exact format its `SOURCE_LOADERS` loader expects (`.md`, `.sql`,
`.json` OpenAPI spec, `.html` for blog/LMS, `.vtt` for transcripts).

## Usage

```
python -m scripts.seed_eval_corpus   # requires live Postgres/ChromaDB/Neo4j
```

Then ingest this repo itself via `ingest_github_repo()` (covers the 4
`github_repositories` eval titles), and run:

```
POST /api/v1/evaluation/run
```

## Keeping this in sync

`tests/test_seed_corpus.py` checks that every title in `eval_set.json`
either has a matching entry in `scripts/seed_eval_corpus.py`'s
`SEED_DOCUMENTS`, or is one of the two documented exceptions (GitHub
file titles, and the intentionally-absent "SOP - Intern Offboarding"
gap-detection title). Add new eval queries and seed documents together.
