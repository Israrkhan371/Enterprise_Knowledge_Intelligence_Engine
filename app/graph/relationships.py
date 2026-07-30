"""
Turns raw multi-granularity co-occurrence records (app/graph/extract.py)
into scored, typed, evidenced, explainable relationships — the actual
"technology maps & skill dependencies from entity co-occurrence"
deliverable (Week 2 Monday, Engineer B), implementing case study AI-007
Steps 3-10.

Two things happen here, deliberately kept in separate functions:
  1. aggregate_cooccurrences() — a pure per-document fold, no I/O. Called
     once per document during ingestion; the result is merged into the
     graph's running totals by GraphStore.upsert_cooccurrence.
  2. infer_relationship() — takes the *running totals* (accumulated
     across every document ingested so far, not just one) and produces
     the scored/typed/explained edge. Called at query time in
     app/graph/queries.py, so confidence reflects everything ingested up
     to that point rather than being frozen at first-sight.
"""
from collections import defaultdict

from app.graph.knowledge_base import (
    KNOWN_RELATIONS,
    SKILL_PREREQUISITE_CHAIN,
    TECHNOLOGY_ECOSYSTEMS,
)

GRANULARITY_WEIGHT = {"sentence": 10, "paragraph": 5, "document": 1}
EVIDENCE_WEIGHT = {
    "import_statement": 22,
    "package_file": 16,
    "deployment_reference": 14,
    "connection_reference": 12,
    "dependency_keyword": 18,
}
MULTI_SOURCE_BONUS_PER_DOC = 3
MULTI_SOURCE_BONUS_CAP = 24
GITHUB_BONUS_PER_REPO = 2
GITHUB_BONUS_CAP = 12

MIN_CONFIDENCE = 5
MAX_CONFIDENCE = 97
# Below this, and with no real evidence, a relationship is too weak to
# report as anything but RELATED_TO (case study Step 3/9 examples: the
# Python -> React "appeared in same project only, no direct dependency"
# case scores 18, well under this line).
WEAK_EVIDENCE_THRESHOLD = 40


def aggregate_cooccurrences(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Folds one document's raw co-occurrence records into per-pair counters."""
    agg: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["source"], r["target"])
        bucket = agg.setdefault(key, {
            "sentence_count": 0, "paragraph_count": 0, "document_count": 0,
            "evidence": set(), "sample_context": "",
        })
        bucket[f"{r['granularity']}_count"] += 1
        bucket["evidence"] |= r.get("evidence", set())
        if not bucket["sample_context"] and r.get("context"):
            bucket["sample_context"] = r["context"]
    return agg


def _canonical_known_relation(source: str, target: str):
    """Known pairs are looked up in both directions but always returned
    oriented the way KNOWN_RELATIONS defines them, so the semantics of
    e.g. PREREQUISITE_OF stay correct regardless of which order the two
    entities happened to be extracted in."""
    if (source, target) in KNOWN_RELATIONS:
        return source, target, KNOWN_RELATIONS[(source, target)]
    if (target, source) in KNOWN_RELATIONS:
        return target, source, KNOWN_RELATIONS[(target, source)]
    return None


def _default_reason(relation: str, source: str, target: str) -> str:
    if relation == "DEPENDS_ON":
        return f"{source} and {target} co-occur alongside import or package-dependency evidence, indicating a direct dependency."
    if relation == "CONNECTS_TO":
        return f"{source} and {target} co-occur alongside connection/driver references, indicating {source} connects to {target}."
    if relation == "DEPLOYS_TO":
        return f"{source} and {target} co-occur alongside deployment references (Dockerfiles, deploy commands), indicating {source} deploys to {target}."
    if relation == "REQUIRES":
        return f"Documentation explicitly describes {source} as requiring, or being built on, {target}."
    return f"{source} and {target} appear together in the knowledge base, but without strong dependency evidence."


def _build_evidence_list(
    evidence: set[str], sentence_count: int, paragraph_count: int,
    supporting_documents: int, supporting_github_repos: int, has_known_baseline: bool,
) -> list[str]:
    items = []
    if supporting_documents:
        items.append(f"Mentioned together in {supporting_documents} document{'s' if supporting_documents != 1 else ''}")
    if supporting_github_repos:
        items.append(f"Found in {supporting_github_repos} GitHub repositor{'ies' if supporting_github_repos != 1 else 'y'}")
    if sentence_count:
        items.append(f"Co-occur in the same sentence {sentence_count} time{'s' if sentence_count != 1 else ''} (strongest proximity signal)")
    if paragraph_count:
        items.append(f"Co-occur in the same paragraph {paragraph_count} time{'s' if paragraph_count != 1 else ''}")
    if "import_statement" in evidence:
        items.append("Import/require statement found linking the two entities")
    if "package_file" in evidence:
        items.append("Referenced together near package/dependency file mentions (requirements.txt, package.json, etc.)")
    if "dependency_keyword" in evidence:
        items.append('Explicit dependency language found ("depends on", "requires", "built on")')
    if "deployment_reference" in evidence:
        items.append("Deployment reference found (Dockerfile, docker-compose, kubectl, helm)")
    if "connection_reference" in evidence:
        items.append("Database/driver connection reference found")
    if has_known_baseline:
        items.append("Confirmed by known technology documentation / runtime relationship")
    return items


def infer_relationship(
    source: str, target: str, *,
    sentence_count: int = 0, paragraph_count: int = 0, document_count: int = 0,
    evidence: set[str] | None = None, supporting_documents: int = 1,
    supporting_github_repos: int = 0,
) -> dict:
    """
    Case study AI-007 Steps 3-10: turn raw co-occurrence counters into one
    scored, typed, evidenced relationship edge.

    Confidence blends two tiers of evidence:
      1. A curated baseline for well-known technology pairs
         (knowledge_base.KNOWN_RELATIONS) — e.g. we don't need to
         *observe* that FastAPI depends on Python, that's a fact about
         the technologies.
      2. Frequency / proximity / textual evidence actually observed in
         the ingested documents, which raises or lowers confidence from
         that baseline (or, for unknown pairs, is the only signal).

    Step 3 is enforced directly: with no real evidence beyond a single
    loose co-occurrence, the relation is forced to RELATED_TO rather than
    a stronger, unsupported claim like DEPENDS_ON.
    """
    evidence = evidence or set()
    known = _canonical_known_relation(source, target)

    if known:
        canon_source, canon_target, (relation, base_conf, reason) = known
        score = base_conf
    else:
        canon_source, canon_target = source, target
        relation, reason, score = None, None, 0

    score += min(sentence_count, 6) * GRANULARITY_WEIGHT["sentence"]
    score += min(paragraph_count, 6) * GRANULARITY_WEIGHT["paragraph"]
    score += min(document_count, 6) * GRANULARITY_WEIGHT["document"]
    score += sum(EVIDENCE_WEIGHT.get(ev, 0) for ev in evidence)
    score += min(max(supporting_documents - 1, 0) * MULTI_SOURCE_BONUS_PER_DOC, MULTI_SOURCE_BONUS_CAP)
    score += min(supporting_github_repos * GITHUB_BONUS_PER_REPO, GITHUB_BONUS_CAP)

    total_cooccurrences = sentence_count + paragraph_count + document_count
    has_strong_evidence = bool(evidence) or known is not None

    if relation is None:
        if "import_statement" in evidence or "package_file" in evidence:
            relation = "DEPENDS_ON"
        elif "connection_reference" in evidence:
            relation = "CONNECTS_TO"
        elif "deployment_reference" in evidence:
            relation = "DEPLOYS_TO"
        elif "dependency_keyword" in evidence:
            relation = "REQUIRES"
        else:
            relation = "RELATED_TO"
        reason = _default_reason(relation, canon_source, canon_target)

    if relation != "RELATED_TO" and not has_strong_evidence and total_cooccurrences <= 1:
        relation = "RELATED_TO"
        reason = (
            f"{canon_source} and {canon_target} were only observed together once, with no "
            f"import, dependency, or deployment evidence — too weak to claim a specific "
            f"relationship type."
        )
        score = min(score, WEAK_EVIDENCE_THRESHOLD)

    confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, round(score)))

    return {
        "source": canon_source,
        "target": canon_target,
        "relation": relation,
        "confidence": confidence,
        "reason": reason,
        "evidence": _build_evidence_list(
            evidence, sentence_count, paragraph_count,
            supporting_documents, supporting_github_repos, known is not None,
        ),
    }


def format_explanation(rel: dict) -> str:
    """Human-readable explanation block, matching the reviewer-facing format:

        Relationship: PREREQUISITE_OF
        Confidence: 96%

        Reason:
        FastAPI is a Python framework.

        Evidence:
        • Mentioned together in 18 documents
        • Found in 6 GitHub repositories
        • ...
    """
    lines = [
        f"{rel['source']} \u2192 {rel['target']}",
        "",
        f"Relationship: {rel['relation']}",
        f"Confidence: {rel['confidence']}%",
        "",
        "Reason:",
        rel["reason"],
        "",
        "Evidence:",
    ]
    lines += [f"\u2022 {e}" for e in rel["evidence"]] or ["\u2022 No supporting evidence recorded"]
    return "\n".join(lines)


def build_technology_map(relationships: list[dict]) -> dict[str, list[dict]]:
    """Groups relationship edges into technology ecosystems (Step 5)."""
    ecosystem_of: dict[str, str] = {}
    for eco, members in TECHNOLOGY_ECOSYSTEMS.items():
        for member in members:
            ecosystem_of[member] = eco

    grouped: dict[str, list[dict]] = defaultdict(list)
    for rel in relationships:
        eco = ecosystem_of.get(rel["source"]) or ecosystem_of.get(rel["target"]) or "Other"
        grouped[eco].append(rel)
    return dict(grouped)


def build_skill_dependency_graph(relationships: list[dict]) -> list[dict]:
    """
    Orders PREREQUISITE_OF edges into a learning path, cross-checked
    against the curated SKILL_PREREQUISITE_CHAIN baseline so an inferred
    edge can't introduce a contradiction (Step 6: "avoid creating
    circular dependencies").
    """
    chain_index = {name: i for i, name in enumerate(SKILL_PREREQUISITE_CHAIN)}
    prereq_edges = [r for r in relationships if r["relation"] == "PREREQUISITE_OF"]

    valid = []
    for r in prereq_edges:
        si, ti = chain_index.get(r["source"]), chain_index.get(r["target"])
        if si is not None and ti is not None and si > ti:
            continue  # contradicts the known learning order — drop as a likely-wrong inference
        valid.append(r)

    valid.sort(key=lambda r: (chain_index.get(r["source"], 999), -r["confidence"]))
    return valid
