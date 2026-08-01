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


# Evidence signals used to distinguish a real dependency from an
# accidental co-occurrence (case study Step 3/4: "dependency statements,
# import statements, package files, architecture documentation, code
# snippets, deployment files"). Matched against the sentence/paragraph
# text surrounding a co-occurring entity pair — the label, not the
# matched text, is what gets stored, so this never leaks raw snippets
# into the evidence set.
EVIDENCE_PATTERNS: dict[str, re.Pattern] = {
    "import_statement": re.compile(
        r"\b(import\s+\w|from\s+\S+\s+import\b|require\(|package\s+\w+;)", re.IGNORECASE
    ),
    "package_file": re.compile(
        r"\b(requirements\.txt|package\.json|pyproject\.toml|pom\.xml|"
        r"cargo\.toml|go\.mod|pip install|npm install|yarn add)\b", re.IGNORECASE
    ),
    "deployment_reference": re.compile(
        r"\b(dockerfile|docker-compose|docker compose|kubectl apply|helm install|"
        r"deploy(?:ed|s)?\s+(?:to|on)|CI/CD pipeline)\b", re.IGNORECASE
    ),
    "connection_reference": re.compile(
        r"\b(connects? to|connection string|database url|db_url|driver|client library)\b",
        re.IGNORECASE,
    ),
    "dependency_keyword": re.compile(
        r"\b(depends on|dependency of|requires|required by|built on|runs on|"
        r"powered by|based on|relies on|prerequisite)\b", re.IGNORECASE
    ),
}

_PARAGRAPH_SPLIT = re.compile(r"\r?\n\s*\r?\n")


def _detect_evidence(chunk: str) -> set[str]:
    return {label for label, pattern in EVIDENCE_PATTERNS.items() if pattern.search(chunk)}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    return paragraphs or ([text] if text.strip() else [])


# Characters of context captured on each side of the nearest mention pair
# when scoping evidence detection to one specific entity pair (see
# _pair_evidence_window). Wide enough to catch "FastAPI (built on Python)"
# or "import fastapi  # requires Python 3.8+" style phrasing without also
# pulling in unrelated sentences.
EVIDENCE_WINDOW_MARGIN = 60


def _all_positions(chunk: str, term: str) -> list[tuple[int, int]]:
    positions = []
    start = 0
    while True:
        idx = chunk.find(term, start)
        if idx == -1:
            break
        positions.append((idx, idx + len(term)))
        start = idx + 1
    return positions


def _pair_evidence_window(chunk: str, a: str, b: str, margin: int = EVIDENCE_WINDOW_MARGIN) -> str:
    """
    Returns the smallest slice of `chunk` spanning one mention of `a` and one
    mention of `b` (the closest pair of mentions, if either appears more than
    once), padded by `margin` characters on each side.

    Without this, evidence was detected once per sentence/paragraph and
    applied to *every* entity pair in it — so "FastAPI imports Starlette;
    Docker was also mentioned" would wrongly credit Docker with
    import_statement evidence. Scoping detection to a window around the
    specific pair fixes that misattribution. Falls back to the whole chunk
    only if a position can't be found (shouldn't happen — callers only
    invoke this once both entities are already confirmed present).
    """
    positions_a = _all_positions(chunk, a)
    positions_b = _all_positions(chunk, b)
    if not positions_a or not positions_b:
        return chunk

    best_span = None
    best_gap = None
    for sa, ea in positions_a:
        for sb, eb in positions_b:
            gap = max(sa, sb) - min(ea, eb)  # <= 0 if the two mentions overlap
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_span = (min(sa, sb), max(ea, eb))

    start, end = best_span
    return chunk[max(0, start - margin): min(len(chunk), end + margin)]


def extract_cooccurrences(text: str, entities: list[dict]) -> list[dict]:
    """
    Multi-granularity co-occurrence extraction feeding the technology-map /
    skill-dependency pipeline (app/graph/relationships.py). Replaces the
    old single-granularity "co_occurs_with" edge, which had no confidence,
    no evidence and no relationship typing (case study AI-007 Steps 2-4).

    Granularities, closest (highest-confidence) first — case study Step 2:
    "Assign higher confidence to closer co-occurrences":
      - sentence:  both entities in the same sentence
      - paragraph: both entities in the same paragraph, not already
                   counted at sentence granularity
      - document:  both entities appear in the document, not already
                   counted at a closer granularity (catches pairs spread
                   across the doc — weak but non-zero signal)

    Each record also carries whatever dependency-evidence signals
    (imports, package files, deploy/connection references, explicit
    dependency language — see EVIDENCE_PATTERNS) appear in the text
    immediately around that specific pair (_pair_evidence_window) —
    scoped per-pair, not per-sentence/paragraph, so an import statement
    elsewhere in a multi-entity sentence doesn't get credited to a pair
    it has nothing to do with. This is what lets
    relationships.infer_relationship() tell a real DEPENDS_ON from an
    accidental RELATED_TO instead of trusting proximity alone (Step 3:
    "Do NOT assume that every co-occurrence represents a meaningful
    relationship").
    """
    ent_texts = {e["text"] for e in entities}
    nlp = get_nlp()
    doc = nlp(text)

    records: list[dict] = []
    at_sentence: set[tuple[str, str]] = set()
    at_paragraph: set[tuple[str, str]] = set()

    for sent in doc.sents:
        sent_entities = sorted({e.text for e in sent.ents if e.text in ent_texts})
        if len(sent_entities) < 2:
            continue
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                a, b = sent_entities[i], sent_entities[j]
                window = _pair_evidence_window(sent.text, a, b)
                records.append({
                    "source": a, "target": b, "granularity": "sentence",
                    "context": window.strip()[:200], "evidence": _detect_evidence(window),
                })
                at_sentence.add(_pair_key(a, b))

    for para in _split_paragraphs(text):
        para_entities = sorted(t for t in ent_texts if t in para)
        if len(para_entities) < 2:
            continue
        for i in range(len(para_entities)):
            for j in range(i + 1, len(para_entities)):
                a, b = para_entities[i], para_entities[j]
                key = _pair_key(a, b)
                if key in at_sentence or key in at_paragraph:
                    continue
                # Evidence is deliberately NOT detected at this granularity.
                # Two entities sharing a paragraph but not a sentence can have
                # an entirely unrelated sentence's dependency language sitting
                # between them (e.g. "...FastAPI requires Python. Separately,
                # Kubernetes handles deployment.") - windowing by character
                # distance alone can't tell that evidence apart from evidence
                # that's actually about this pair, so paragraph/document
                # granularity stay frequency-only signals. Sentence granularity
                # is the only one tight enough to trust for evidence (see
                # _pair_evidence_window above).
                records.append({
                    "source": a, "target": b, "granularity": "paragraph",
                    "context": para.strip()[:200], "evidence": set(),
                })
                at_paragraph.add(key)

    doc_entities = sorted(ent_texts)
    for i in range(len(doc_entities)):
        for j in range(i + 1, len(doc_entities)):
            a, b = doc_entities[i], doc_entities[j]
            key = _pair_key(a, b)
            if key in at_sentence or key in at_paragraph:
                continue
            records.append({
                "source": a, "target": b, "granularity": "document",
                "context": "", "evidence": set(),
            })

    return records