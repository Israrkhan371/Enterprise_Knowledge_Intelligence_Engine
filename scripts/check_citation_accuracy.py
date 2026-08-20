"""
Citation accuracy checks (Week 4, Mon, Track B).

Retrieval accuracy (precision/recall/MRR — app/evaluation/eval.py) measures
whether hybrid search surfaces the *right documents*. It says nothing about
whether the answers /ask actually generates are honestly grounded in what
was retrieved. This script closes that gap: it runs every query in
app/evaluation/eval_set.json through the real answer-generation path
(generate_answer(), the same function POST /ask calls) and then through the
same citation verifier /ask uses on every live request
(app.rag.citation_check.verify_citations — sentence-vs-cited-source
embedding similarity, threshold 0.55), and aggregates the results.

This intentionally reuses eval_set.json rather than a separate fixture:
those 40 queries already span every required knowledge-source type and are
kept resolvable against whatever the current corpus looks like
(see eval.py's docstring), so citation accuracy is measured against the
same real, representative query mix as retrieval accuracy — not a
hand-picked set likely to look better than production traffic.

Requires a live stack (Postgres + ChromaDB + Neo4j + a working
GOOGLE_API_KEY for the Gemini calls inside generate_answer()) — same
requirements as POST /ask itself. Run it the same way eval.py's harness is
run, e.g.:

    docker compose exec api python scripts/check_citation_accuracy.py
    docker compose exec api python scripts/check_citation_accuracy.py --k 6
    docker compose exec api python scripts/check_citation_accuracy.py --limit 15
    docker compose exec api python scripts/check_citation_accuracy.py --offset 15 --limit 15

The Gemini free tier caps requests per day per model (20/day at time of
writing — see the ClientError's quotaId
GenerateRequestsPerDayPerProjectPerModel-FreeTier if you hit it). All 40
eval queries in one run will exceed that on its own, before counting any
other /ask traffic that day. --limit/--offset let you split the 40-query
set across multiple days (or runs) to stay under quota; each
offset/limit combination writes its own results file instead of
overwriting the full run's, so partial runs can be combined by hand
afterward.

Prints a summary to stdout, writes the full per-query breakdown to
docs/citation_accuracy_results.json for a full run (or
docs/citation_accuracy_results_offset{N}_limit{M}.json for a sliced run —
overwritten if that same slice is re-run, since the file is evidence for
the evaluation report, not a growing log), and logs the aggregate numbers
to MLflow under the "ekie-citation-eval" experiment
(kept separate from eval.py's "ekie-retrieval-eval" experiment — retrieval
and citation accuracy are different things being measured and shouldn't be
mixed into the same run's metric set).
"""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Run directly as `python scripts/check_citation_accuracy.py` (as opposed
# to `python -m app.evaluation.eval`, which is how eval.py's equivalent is
# documented to run), Python only puts this file's own directory
# (.../scripts) on sys.path — not the repo root where the `app` package
# lives — so `import app...` below would fail with ModuleNotFoundError.
# Insert the repo root explicitly so this script works with either
# invocation style.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from app.core.config import settings
from app.core.database import SessionLocal
from app.evaluation.eval import load_eval_set
from app.rag.citation_check import verify_citations
from app.rag.generate import generate_answer

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "citation_accuracy_results.json"


def _flag_category(flag: dict) -> str:
    # verify_citations() now emits a machine-readable "kind" field on every
    # flag specifically so this doesn't need to substring-match the
    # human-readable "issue" text - that's exactly what broke silently
    # after citation_check.py's wording changed from "low similarity" to
    # "low relevance" during the cross-encoder rewrite (2026-08-19): this
    # function and inspect_citation_flags.py's filter both hardcoded the
    # old string, so every flag fell through into "other" or got filtered
    # out entirely, with no error - just wrong/empty numbers. Reading
    # "kind" instead makes that class of bug impossible: any future
    # wording change to "issue" simply can't affect categorization.
    return {
        "no_matching_source": "citation_number_not_in_sources",
        "low_relevance": "claim_not_supported_by_cited_source",
    }.get(flag.get("kind"), "other")


def _enrich_flags(flags: list[dict], sources: list[dict]) -> list[dict]:
    # verify_citations() reports which source was cited (cited_source_index)
    # but not what that source actually says - so judging "does the source
    # support this claim" means separately looking the source back up.
    # Attach the full text here so the results file is self-contained for
    # exactly that check, without needing another live /ask call.
    by_index = {s["index"]: s.get("text", "") for s in sources}
    enriched = []
    for flag in flags:
        cited_index = flag.get("cited_source_index")
        enriched.append({
            **flag,
            "cited_source_text": by_index.get(cited_index, "")[:800] if cited_index is not None else "",
        })
    return enriched


def _results_path(offset: int, limit) -> Path:
    if offset == 0 and limit is None:
        return RESULTS_PATH
    suffix = f"_offset{offset}" if offset else ""
    suffix += f"_limit{limit}" if limit is not None else ""
    return RESULTS_PATH.parent / f"citation_accuracy_results{suffix}.json"


def run(k: int = 6, offset: int = 0, limit: int | None = None) -> dict:
    full_eval_set = load_eval_set()
    if not full_eval_set:
        return {"error": "no eval set found — add app/evaluation/eval_set.json"}

    eval_set = full_eval_set[offset:offset + limit] if limit is not None else full_eval_set[offset:]
    if not eval_set:
        return {"error": f"offset {offset} is past the end of the {len(full_eval_set)}-query eval set"}

    db = SessionLocal()
    per_query = []
    try:
        for item in eval_set:
            query = item["query"]
            try:
                result = generate_answer(db, query, top_k=k)
            except TimeoutError:
                per_query.append({"query": query, "error": "LLM request timed out"})
                continue
            except Exception as exc:  # noqa: BLE001 — one bad query must not abort the whole run
                logger.exception("generate_answer failed for query=%r", query)
                per_query.append({"query": query, "error": str(exc)})
                continue

            verification = verify_citations(result["answer"], result["sources"])
            per_query.append({
                "query": query,
                "answer": result["answer"],
                "num_sources_offered": len(result["sources"]),
                "num_citations_in_answer": len(verification["cited_sources"]),
                "verified": verification["verified"],
                "flags": _enrich_flags(verification["flags"], result["sources"]),
            })
    finally:
        db.close()

    scored = [q for q in per_query if "error" not in q]
    errored = [q for q in per_query if "error" in q]
    # Gemini's free tier is a *daily* per-model request cap, not a
    # per-minute one — the ClientError's message includes a short
    # "retry in Ns" hint that doesn't apply to a daily cap, so surfacing
    # this count separately from other errors (timeouts, malformed
    # responses) matters: a run full of RESOURCE_EXHAUSTED entries means
    # "come back after the daily quota resets, or use --offset/--limit to
    # run a smaller slice," not "something in the pipeline is broken."
    quota_exhausted = [q for q in errored if "RESOURCE_EXHAUSTED" in q.get("error", "")]

    uncited = [q for q in scored if q["num_citations_in_answer"] == 0]
    cited = [q for q in scored if q["num_citations_in_answer"] > 0]
    cited_and_verified = [q for q in cited if q["verified"]]

    all_flags = [f for q in scored for f in q["flags"]]
    flag_breakdown = Counter(_flag_category(f) for f in all_flags)

    summary = {
        "num_queries_in_full_eval_set": len(full_eval_set),
        "num_queries_this_run": len(eval_set),
        "offset": offset,
        "num_scored": len(scored),
        "num_errored": len(errored),
        "num_errored_quota_exhausted": len(quota_exhausted),
        "num_answers_with_no_citations": len(uncited),
        "num_answers_with_citations": len(cited),
        # The metric that actually matters: of the answers that cited
        # something, how many had every citation check out. Reporting bare
        # verify_citations()["verified"] over *all* answers would count an
        # uncited, unhelpful answer as "verified" (see citation_check.py's
        # own test: verified defaults True when there's nothing to check),
        # which would flatter the number rather than measure grounding.
        "citation_accuracy_rate": round(len(cited_and_verified) / len(cited), 3) if cited else None,
        "flag_breakdown": dict(flag_breakdown),
        "total_flags": len(all_flags),
        "k": k,
    }

    results_path = _results_path(offset, limit)
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))
    summary["_results_path"] = str(results_path)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ekie-citation-eval")
    with mlflow.start_run():
        mlflow.log_metrics({
            k_: v for k_, v in summary.items()
            if isinstance(v, (int, float)) and v is not None
        })

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=6, help="top_k passed to generate_answer (default: 6, matches /ask's default)")
    parser.add_argument("--offset", type=int, default=0, help="skip this many queries from the start of eval_set.json (default: 0)")
    parser.add_argument("--limit", type=int, default=None, help="run at most this many queries, starting at --offset (default: all remaining)")
    args = parser.parse_args()

    summary = run(k=args.k, offset=args.offset, limit=args.limit)
    print(json.dumps(summary, indent=2))
    if "error" not in summary:
        print(f"\nFull per-query breakdown written to {summary['_results_path']}")
