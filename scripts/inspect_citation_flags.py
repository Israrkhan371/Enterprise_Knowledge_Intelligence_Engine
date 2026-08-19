"""
Dumps every flagged citation from the two citation-accuracy result files,
full and untruncated, sorted by similarity score ascending, so you can
manually judge whether each flag is a genuine unsupported claim or a
false positive from an overly strict similarity_threshold.

Usage (inside the container):
    python scripts/inspect_citation_flags.py

For each flag, prints: the similarity score, the exact sentence that was
cited, and the exact cited source text it was compared against - side by
side, so you can eyeball whether the claim really is supported by that
source. Once you've looked at ~10-15 of these (especially the ones with
the highest similarity scores among the flagged set - those are the most
likely false positives), pick a threshold that sits just below the lowest
score you judge to be a genuinely accurate citation.
"""
import json
from pathlib import Path

RESULT_FILES = [
    "docs/citation_accuracy_results_limit20.json",
    "docs/citation_accuracy_results_offset20_limit20.json",
]


def main():
    all_flags = []
    for path in RESULT_FILES:
        if not Path(path).exists():
            print(f"(skipping {path} - not found)")
            continue
        data = json.loads(Path(path).read_text())
        for entry in data.get("per_query", []):
            if entry.get("error"):
                continue
            for flag in entry.get("flags", []):
                if "low similarity" not in flag.get("issue", ""):
                    continue
                all_flags.append({
                    "query": entry.get("query"),
                    "sentence": flag["sentence"],
                    "issue": flag["issue"],
                    "cited_source_text": flag.get("cited_source_text", "(not present in this result file - "
                                                                        "re-run check_citation_accuracy.py after "
                                                                        "adding cited_source_text to the flag dict "
                                                                        "if this is missing)"),
                })

    def _score(f):
        # issue looks like "low similarity (0.44) to source [1]"
        try:
            return float(f["issue"].split("(")[1].split(")")[0])
        except (IndexError, ValueError):
            return -1.0

    all_flags.sort(key=_score, reverse=True)  # highest score first = most likely false positives

    print(f"Total flagged citations: {len(all_flags)}\n")
    print("=" * 100)
    for i, f in enumerate(all_flags, start=1):
        print(f"\n[{i}] {f['issue']}")
        print(f"Query: {f['query']}")
        print(f"\nCited sentence:\n  {f['sentence']}")
        print(f"\nCited source text (full):\n  {f['cited_source_text']}")
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
