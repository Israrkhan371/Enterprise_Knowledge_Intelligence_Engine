"""
Guards against seed_data/ and eval_set.json drifting apart.

eval_set.json's relevant_document_titles are resolved against the live
`documents` table at evaluation time (see app/evaluation/eval.py). If a
title in eval_set.json doesn't have a corresponding seed document (and
isn't one of the two known exceptions below), that query silently gets
skipped by run_evaluation() instead of scored — this test catches that
before it ships, without needing a live DB.
"""
import json
from pathlib import Path

from scripts.seed_eval_corpus import SEED_DOCUMENTS

REPO_ROOT = Path(__file__).parent.parent
EVAL_SET_PATH = REPO_ROOT / "app" / "evaluation" / "eval_set.json"
SEED_DATA_DIR = REPO_ROOT / "seed_data"

# Titles that eval_set.json references but seed_eval_corpus.py deliberately
# does NOT seed:
#   - the 4 github_repositories titles resolve once this repo is ingested
#     via ingest_github_repo(), which sets Document.title to the file path
#   - "SOP - Intern Offboarding" is the gap_detection query; it must NOT
#     exist, since the query tests that a missing-documentation gap is
#     correctly surfaced
_EXPECTED_UNSEEDED_TITLES = {
    "app/api/routes.py",
    "app/graph/extract.py",
    "app/ingestion/pipeline.py",
    "app/search/hybrid.py",
    "SOP - Intern Offboarding",
}


def _eval_set_titles() -> set[str]:
    data = json.loads(EVAL_SET_PATH.read_text())
    titles: set[str] = set()
    for item in data:
        titles.update(item.get("relevant_document_titles", []))
    return titles


def test_every_eval_set_title_is_seeded_or_a_known_exception():
    eval_titles = _eval_set_titles()
    seeded_titles = {title for _path, title, _source_type, _category in SEED_DOCUMENTS}

    unaccounted = eval_titles - seeded_titles - _EXPECTED_UNSEEDED_TITLES
    assert not unaccounted, (
        f"eval_set.json references title(s) with no seed document and no "
        f"documented exception: {sorted(unaccounted)}. Either add a seed "
        f"document in seed_data/ + SEED_DOCUMENTS, or add the title to "
        f"_EXPECTED_UNSEEDED_TITLES with a reason."
    )


def test_every_seed_document_file_exists():
    missing = [
        rel_path for rel_path, _title, _source_type, _category in SEED_DOCUMENTS
        if not (SEED_DATA_DIR / rel_path).exists()
    ]
    assert not missing, f"SEED_DOCUMENTS references missing file(s): {missing}"


def test_seed_document_titles_are_unique():
    titles = [title for _path, title, _source_type, _category in SEED_DOCUMENTS]
    duplicates = {t for t in titles if titles.count(t) > 1}
    assert not duplicates, f"Duplicate titles in SEED_DOCUMENTS: {duplicates}"
