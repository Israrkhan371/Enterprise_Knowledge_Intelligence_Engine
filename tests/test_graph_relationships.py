"""
Tests for app/graph/relationships.py — confidence scoring, relation-type
inference, ecosystem grouping, skill-chain ordering, and the explainability
formatter. All pure Python: no Neo4j, no spaCy, no live services required.
"""
from app.graph.relationships import (
    aggregate_cooccurrences,
    build_skill_dependency_graph,
    build_technology_map,
    format_explanation,
    infer_relationship,
)


def test_aggregate_cooccurrences_counts_by_granularity():
    records = [
        {"source": "Python", "target": "FastAPI", "granularity": "sentence", "context": "a", "evidence": {"import_statement"}},
        {"source": "Python", "target": "FastAPI", "granularity": "sentence", "context": "b", "evidence": set()},
        {"source": "Python", "target": "FastAPI", "granularity": "paragraph", "context": "c", "evidence": set()},
    ]
    agg = aggregate_cooccurrences(records)
    bucket = agg[("Python", "FastAPI")]
    assert bucket["sentence_count"] == 2
    assert bucket["paragraph_count"] == 1
    assert bucket["document_count"] == 0
    assert bucket["evidence"] == {"import_statement"}
    assert bucket["sample_context"] == "a"


def test_known_pair_gets_curated_relation_and_high_confidence():
    rel = infer_relationship(
        "Python", "FastAPI",
        sentence_count=5, paragraph_count=2, document_count=1,
        evidence={"import_statement"}, supporting_documents=6, supporting_github_repos=2,
    )
    assert rel["relation"] == "PREREQUISITE_OF"
    assert rel["confidence"] >= 90
    assert "Mentioned together in 6 documents" in rel["evidence"]
    assert "Found in 2 GitHub repositories" in rel["evidence"]


def test_known_pair_direction_is_canonicalized_regardless_of_input_order():
    """FastAPI/Python (reversed order) should still resolve to the
    Python -> FastAPI PREREQUISITE_OF relation, not the reverse."""
    rel = infer_relationship("FastAPI", "Python", sentence_count=3, supporting_documents=3)
    assert rel["source"] == "Python"
    assert rel["target"] == "FastAPI"
    assert rel["relation"] == "PREREQUISITE_OF"


def test_single_weak_cooccurrence_is_demoted_to_related_to():
    """Case study example: Python -> React, appeared in the same project
    only, no direct dependency evidence -> should end up weak/RELATED_TO,
    not a confidently-wrong DEPENDS_ON."""
    rel = infer_relationship(
        "Python", "React",
        sentence_count=0, paragraph_count=0, document_count=1,
        evidence=set(), supporting_documents=1,
    )
    assert rel["relation"] == "RELATED_TO"
    assert rel["confidence"] < 40


def test_unknown_pair_with_import_evidence_infers_depends_on():
    rel = infer_relationship(
        "ServiceA", "ServiceB",
        sentence_count=3, paragraph_count=1, evidence={"import_statement"},
        supporting_documents=4,
    )
    assert rel["relation"] == "DEPENDS_ON"
    assert 0 <= rel["confidence"] <= 97


def test_confidence_is_bounded():
    rel = infer_relationship(
        "X", "Y",
        sentence_count=999, paragraph_count=999, document_count=999,
        evidence={"import_statement", "package_file", "deployment_reference",
                  "connection_reference", "dependency_keyword"},
        supporting_documents=999, supporting_github_repos=999,
    )
    assert rel["confidence"] <= 97


def test_format_explanation_matches_expected_shape():
    rel = infer_relationship(
        "Python", "FastAPI",
        sentence_count=4, evidence={"import_statement"}, supporting_documents=18,
        supporting_github_repos=6,
    )
    text = format_explanation(rel)
    assert "Relationship: PREREQUISITE_OF" in text
    assert f"Confidence: {rel['confidence']}%" in text
    assert "Reason:" in text
    assert "Evidence:" in text
    assert "\u2022 Mentioned together in 18 documents" in text


def test_build_technology_map_groups_by_ecosystem():
    relationships = [
        infer_relationship("Python", "FastAPI", sentence_count=2, supporting_documents=2),
        infer_relationship("JavaScript", "React", sentence_count=2, supporting_documents=2),
    ]
    tech_map = build_technology_map(relationships)
    assert "Python" in tech_map
    assert "JavaScript" in tech_map
    assert tech_map["Python"][0]["target"] == "FastAPI"


def test_build_skill_dependency_graph_orders_by_curated_chain():
    relationships = [
        infer_relationship("Docker", "Kubernetes", sentence_count=3, supporting_documents=3),
        infer_relationship("Python", "FastAPI", sentence_count=3, supporting_documents=3),
    ]
    graph = build_skill_dependency_graph(relationships)
    sources = [r["source"] for r in graph]
    assert sources.index("Python") < sources.index("Docker")


def test_build_skill_dependency_graph_drops_edges_contradicting_known_order():
    """A (hypothetically mis-inferred) Kubernetes -> Python PREREQUISITE_OF
    edge would contradict the curated learning order and must be dropped."""
    contradicting_edge = {
        "source": "Kubernetes", "target": "Python", "relation": "PREREQUISITE_OF",
        "confidence": 50, "reason": "test", "evidence": [],
    }
    graph = build_skill_dependency_graph([contradicting_edge])
    assert graph == []
