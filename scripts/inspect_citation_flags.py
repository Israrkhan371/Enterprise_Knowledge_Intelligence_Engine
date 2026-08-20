"""
Dumps every flagged citation from the two citation-accuracy result files,
full and untruncated, sorted by score descending (highest scores among the
flagged set are the most likely false positives — closest to whatever
threshold was in effect when the run happened), so you can manually judge
whether each low_relevance flag is a genuine unsupported claim or a false
positive from an overly strict relevance_threshold.

Usage (inside the container):
    python scripts/inspect_citation_flags.py

Reads each flag's "kind" field directly rather than substring-matching the
human-readable "issue" text. That distinction matters here specifically:
an earlier version of this script (and of check_citation_accuracy.py's
flag_breakdown) hardcoded a match against the word "similarity", which
silently broke — printing "Total flagged citations: 0" with no error —
the moment citation_check.py's wording changed to "relevance" during the
cross-encoder rewrite. Reading "kind" makes that class of bug impossible;
a future wording change to "issue" can't affect this script's filtering.

For each low_relevance flag, prints: the score, the exact sentence that
was cited, and the exact cited source text it was compared against - side
by side, so you can eyeball whether the claim really is supported by that
source. Once you've looked at ~10-15 of these (especially the ones with
the highest scores among the flagged set), pick a threshold that sits
just below the lowest score you judge to be a genuinely accurate citation.

no_matching_source flags (a citation number with no corresponding source
at all) are reported separately, not silently dropped - they're a
different bug class from threshold calibration (a malformed citation
number, not a loosely-supported one) and mixing them into the same list
would misrepresent both.
"""
import json
from pathlib import Path

RESULT_FILES = [
    "docs/citation_accuracy_results_limit20.json",
    "docs/citation_accuracy_results_offset20_limit20.json",
]


def main():
    low_relevance_flags = []
    no_matching_source_flags = []
    unrecognized_kinds = []

    for path in RESULT_FILES:
        if not Path(path).exists():
            print(f"(skipping {path} - not found)")
            continue
        data = json.loads(Path(path).read_text())
        for entry in data.get("per_query", []):
            if entry.get("error"):
                continue
            for flag in entry.get("flags", []):
                enriched = {
                    "query": entry.get("query"),
                    "sentence": flag.get("sentence"),
                    "issue": flag.get("issue"),
                    "score": flag.get("score"),
                    "cited_source_text": flag.get("cited_source_text", "(not present in this result file)"),
                }
                kind = flag.get("kind")
                if kind == "low_relevance":
                    low_relevance_flags.append(enriched)
                elif kind == "no_matching_source":
                    no_matching_source_flags.append(enriched)
                else:
                    # Deliberately not silently dropped - a flag with a
                    # missing/unrecognized "kind" (e.g. from an older
                    # result file predating this field) is reported, not
                    # swallowed, so a stale-schema result file can't
                    # quietly show up as "0 flags" again.
                    unrecognized_kinds.append(enriched)

    low_relevance_flags.sort(key=lambda f: f["score"] if f["score"] is not None else float("-inf"), reverse=True)

    print(f"low_relevance flags: {len(low_relevance_flags)}")
    print(f"no_matching_source flags: {len(no_matching_source_flags)} (different bug class - not threshold-relevant)")
    if unrecognized_kinds:
        print(f"UNRECOGNIZED kind (likely a stale result file predating the 'kind' field): {len(unrecognized_kinds)}")
    print()
    print("=" * 100)

    for i, f in enumerate(low_relevance_flags, start=1):
        print(f"\n[{i}] score={f['score']} — {f['issue']}")
        print(f"Query: {f['query']}")
        print(f"\nCited sentence:\n  {f['sentence']}")
        print(f"\nCited source text (full):\n  {f['cited_source_text']}")
        print("\n" + "=" * 100)

    if no_matching_source_flags:
        print("\nno_matching_source flags (citation number with no corresponding source — separate issue):")
        for f in no_matching_source_flags:
            print(f"  - {f['issue']}  (query: {f['query']})")


if __name__ == "__main__":
    main()
