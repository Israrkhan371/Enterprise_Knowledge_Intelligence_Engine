import re
import spacy

_nlp = None

RELEVANT_LABELS = {"ORG", "PERSON", "GPE", "PRODUCT", "LANGUAGE", "NORP", "TECH"}

TECH_TERMS = [
    "Python", "Kubernetes", "Rust", "PostgreSQL", "Docker", "Grafana",
    "PyTorch", "TensorFlow", "React", "FastAPI", "Neo4j", "ChromaDB",
    "Prometheus", "NVIDIA Jetson", "Go", "Java", "JavaScript", "TypeScript",
    "Redis", "MongoDB", "AWS", "Azure", "GCP", "Terraform", "Node.js",
]
# Lowercased lookup so gazetteer matches always emit the same canonical
# casing regardless of how the source text wrote it (e.g. "chromadb" in
# a Python import statement vs "ChromaDB" in prose — same real entity,
# was previously becoming two separate graph nodes).
_TECH_CANONICAL = {t.lower(): t for t in TECH_TERMS}

# SQL/code syntax and generic role labels that are technically valid
# PERSON/ORG/PRODUCT spans by spaCy's own label definitions, but aren't
# real-world named entities. These show up when NER runs over raw DDL
# (CREATE TABLE ... DEFAULT ...) or short role annotations in meeting
# notes ("Marcus Webb (CTO)", "Attendees: ..."). A stoplist is the
# pragmatic fix here — the underlying problem (spaCy has no notion of
# "this text is SQL, not English") isn't solvable by relabeling.
_STOPLIST = {
    "table", "default", "uuid primary key", "primary key", "attendees",
    "cto", "ceo", "cfo", "vp", "devops", "lead devops", "ai", "api",
}

_LEADING_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_LEADING_SYMBOLS = re.compile(r"^[-#*>\s]+")


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
    text = _LEADING_SYMBOLS.sub("", text)
    text = _LEADING_ARTICLE.sub("", text)
    return text.strip()


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    results = []
    seen = set()

    for ent in doc.ents:
        if ent.label_ not in RELEVANT_LABELS:
            continue

        if ent.label_ == "NORP" and len(ent) == 1 and ent.root.pos_ == "ADJ":
            continue

        cleaned = _clean_entity_text(ent.text)
        if not cleaned:
            continue

        if cleaned.lower() in _STOPLIST:
            continue

        canonical = _TECH_CANONICAL.get(cleaned.lower())
        if canonical:
            cleaned = canonical
            label = "TECH"
        else:
            label = ent.label_

        key = (cleaned, label)
        if key in seen:
            continue
        seen.add(key)

        results.append({"text": cleaned, "label": label})

    return _merge_partial_person_names(results)


def _merge_partial_person_names(entities: list[dict]) -> list[dict]:
    """
    Within a single document, the same person is often introduced with a
    full name once ("Priya Chandrasekaran founded...") and referred to by
    surname alone later ("Chandrasekaran confirmed..."). Without this,
    those become two separate graph nodes for one real person. This merges
    any PERSON entity that is a proper word-boundary substring of a longer
    PERSON entity into the longer (fuller) form.

    Deliberately scoped to within one document's extraction pass — matching
    partial names across *different* documents would need a much bigger
    entity-resolution step against existing graph entities, and carries a
    real risk of wrongly merging two different people who happen to share
    a surname. That's out of scope here.
    """
    persons = [e for e in entities if e["label"] == "PERSON"]
    others = [e for e in entities if e["label"] != "PERSON"]

    # Longest names first, so shorter partial mentions merge into the
    # fullest form seen in this document.
    persons_sorted = sorted(persons, key=lambda e: len(e["text"]), reverse=True)

    canonical_map: dict[str, str] = {}
    kept: list[dict] = []
    for ent in persons_sorted:
        name = ent["text"]
        name_words = set(name.split())
        merged_into = None
        for kept_ent in kept:
            kept_words = set(kept_ent["text"].split())
            # Merge only if every word in the shorter name appears as a
            # whole word in the longer name (word-boundary safe — "Chan"
            # won't wrongly merge into "Chandrasekaran").
            if name_words and name_words.issubset(kept_words):
                merged_into = kept_ent["text"]
                break
        if merged_into:
            canonical_map[name] = merged_into
        else:
            kept.append(ent)
            canonical_map[name] = name

    deduped_persons = list({e["text"]: e for e in kept}.values())
    return others + deduped_persons


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