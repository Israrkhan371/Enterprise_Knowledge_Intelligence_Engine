import spacy

_nlp = None

RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "LANGUAGE", "NORP", "TECH"}

# Known technology/tool names that en_core_web_sm frequently misses entirely
# (e.g. "Rust", "PostgreSQL") or merges incorrectly with neighboring words
# (e.g. "Python and Kubernetes" as one span). This EntityRuler runs after
# the statistical NER pass and overwrites conflicting spans, so it both
# adds missed terms and splits/corrects merged ones.
TECH_TERMS = [
    "Python", "Kubernetes", "Rust", "PostgreSQL", "Docker", "Grafana",
    "PyTorch", "TensorFlow", "React", "FastAPI", "Neo4j", "ChromaDB",
    "Prometheus", "NVIDIA Jetson", "Go", "Java", "JavaScript", "TypeScript",
    "Redis", "MongoDB", "AWS", "Azure", "GCP", "Terraform", "Node.js",
]


def get_nlp():
    global _nlp
    if _nlp is None:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe(
            "entity_ruler", after="ner", config={"overwrite_ents": True}
        )
        # Match case-insensitively on token text (LOWER), token-by-token,
        # so multi-word terms like "NVIDIA Jetson" still match correctly.
        patterns = [
            {"label": "TECH", "pattern": [{"LOWER": w.lower()} for w in term.split()]}
            for term in TECH_TERMS
        ]
        ruler.add_patterns(patterns)
        _nlp = nlp
    return _nlp


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in RELEVANT_LABELS
    ]


def extract_relationships(text: str, entities: list[dict]) -> list[dict]:
    """
    MVP: co-occurrence within the same sentence implies a relationship edge.
    Swap for an LLM-based relation extractor for higher precision — prompt
    Claude with the sentence and entity pair, ask for a relation label.
    """
    nlp = get_nlp()
    doc = nlp(text)
    relationships = []
    ent_texts = {e["text"] for e in entities}

    for sent in doc.sents:
        sent_entities = [e.text for e in sent.ents if e.text in ent_texts]
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                relationships.append({
                    "source": sent_entities[i],
                    "target": sent_entities[j],
                    "relation": "co_occurs_with",
                    "context": sent.text[:200],
                })
    return relationships