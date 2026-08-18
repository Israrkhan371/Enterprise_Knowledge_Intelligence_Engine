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

Prints a summary to stdout, writes the full per-query breakdown to
docs/citation_accuracy_results.json (overwritten each run — this file is
evidence for the evaluation report, not a growing log), and logs the
aggregate numbers to MLflow under the "ekie-citation-eval" experiment
(kept separate from eval.py's "ekie-retrieval-eval" experiment — retrieval
and citation accuracy are different things being measured and shouldn't be
mixed into the same run's metric set).
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import mlflow

from app.core.config import settings
from app.core.database import SessionLocal
from app.evaluation.eval import load_eval_set
from app.rag.citation_check import verify_citations
from app.rag.generate import generate_answer

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "citation_accuracy_results.json"


def _flag_category(issue: str) -> str:
    # verify_citations()'s flags are free-text (f"...{idx}...") — bucket
    # them by the two distinct failure modes it actually produces, so the
    # summary can report a breakdown instead of a wall of unique strings.
    if "no matching source" in issue:
        return "citation_number_not_in_sources"
    if "low similarity" in issue:
        return "claim_not_supported_by_cited_source"
    return "other"


def run(k: int = 6) -> dict:
    eval_set = load_eval_set()
    if not eval_set:
        return {"error": "no eval set found — add app/evaluation/eval_set.json"}

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
                "flags": verification["flags"],
            })
    finally:
        db.close()

    scored = [q for q in per_query if "error" not in q]
    errored = [q for q in per_query if "error" in q]

    uncited = [q for q in scored if q["num_citations_in_answer"] == 0]
    cited = [q for q in scored if q["num_citations_in_answer"] > 0]
    cited_and_verified = [q for q in cited if q["verified"]]

    all_flags = [f for q in scored for f in q["flags"]]
    flag_breakdown = Counter(_flag_category(f["issue"]) for f in all_flags)

    summary = {
        "num_queries": len(eval_set),
        "num_scored": len(scored),
        "num_errored": len(errored),
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

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "per_query": per_query}, indent=2))

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6, help="top_k passed to generate_answer (default: 6, matches /ask's default)")
    args = parser.parse_args()

    summary = run(k=args.k)
    print(json.dumps(summary, indent=2))
    if "error" not in summary:
        print(f"\nFull per-query breakdown written to {RESULTS_PATH}")
