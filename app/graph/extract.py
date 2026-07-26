import spacy

_nlp = None

# Entity types worth keeping for a technology/skill/people knowledge graph.
# Excludes noisy spaCy categories that aren't real graph entities: DATE,
# TIME, MONEY, CARDINAL, ORDINAL, PERCENT, QUANTITY, DURATION.
RELEVANT_LABELS = {
    "ORG",
    "PERSON",
    "PRODUCT",
    "GPE",
    "LANGUAGE",
    "NORP",
}


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
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