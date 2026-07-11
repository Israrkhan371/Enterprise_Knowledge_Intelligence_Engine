import spacy

_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> list[dict]:
    nlp = get_nlp()
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


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
