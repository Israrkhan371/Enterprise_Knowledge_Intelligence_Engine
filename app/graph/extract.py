import re
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

# Leading articles that occasionally get included in GPE/NORP spans by spaCy
# (e.g. "the United States"). Stripped for cleaner graph node names.
_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def get_nlp():
    global _nlp
    if _nlp is None:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe(
            "entity_ruler", after="ner", config={"overwrite_ents": True}
        )
        patterns = [
            {"label": "TECH", "pattern": [{"LOWER": w.lower()} for w in term.split()]}
            for term in TECH_TERMS
        ]
        ruler.add_patterns(patterns)
        _nlp = nlp
    return _nlp


def _clean_entity_text(text: str) -> str:
    return _LEADING_ARTICLE.sub("", text).strip()


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    results = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_LABELS:
            continue

        # Skip bare adjectival demonyms (e.g. "European" used as an adjective
        # in "European sales") — these are technically valid NORP spans per
        # spaCy's label definition, but aren't useful standalone graph nodes.
        # A single-token NORP whose root is tagged ADJ is this pattern;
        # multi-word or noun-headed NORP entities (e.g. "the French") still
        # pass through fine.
        if ent.label_ == "NORP" and len(ent) == 1 and ent.root.pos_ == "ADJ":
            continue

        cleaned = _clean_entity_text(ent.text)
        if not cleaned:
            continue

        # Dedup after cleaning, in case stripping created a duplicate
        # (e.g. "the United States" and "United States" both appearing)
        key = (cleaned, ent.label_)
        if key in seen:
            continue
        seen.add(key)

        results.append({"text": cleaned, "label": ent.label_})

    return results


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