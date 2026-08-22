import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.deps import ensure_valid_uuid
from app.core.models import Document, UsageLog
from app.search.semantic import semantic_search
from app.search.keyword import keyword_search
from app.search.hybrid import hybrid_search
from app.search.metadata import metadata_search
from app.search.context_aware import rewrite_query
from app.rag.generate import generate_answer, GeminiQuotaExceededError
from app.rag.citation_check import verify_citations
from app.rag.intelligence import compare_documents, compare_documents_full, summarize_document_full, suggest_document_updates
from app.graph.queries import (
    explain_relationship,
    get_skill_dependencies,
    get_technology_map,
    recommend_learning_path,
)
from app.evaluation.eval import run_evaluation

router = APIRouter(tags=["knowledge"])

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    query: str
    history: list[dict] = []
    user_id: str | None = None


class FeedbackRequest(BaseModel):
    was_helpful: bool


class FeedbackResponse(BaseModel):
    usage_log_id: str
    was_helpful: bool


class CompareRequest(BaseModel):
    document_id_a: str = Field(min_length=1)
    document_id_b: str = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_must_differ(self):
        if self.document_id_a == self.document_id_b:
            raise ValueError("document_id_a and document_id_b must be different documents")
        return self


class CompareResponse(BaseModel):
    similarity: float | None
    diff: list[str]
    summary: str


class SummaryResponse(BaseModel):
    summary: str


@router.get(
    "/search/semantic",
    tags=["search"],
    summary="Semantic search over the vector store",
    description="AI Search: Semantic Search. Embeds the query and ranks chunks by cosine distance in ChromaDB. category_filter restricts to one document category.",
)
def search_semantic(q: str, top_k: int = 10, category_filter: str | None = None, db: Session = Depends(get_db)):
    return semantic_search(q, top_k=top_k, category_filter=category_filter, db=db)


@router.get(
    "/search/keyword",
    tags=["search"],
    summary="Keyword search over Postgres full-text index",
    description="AI Search: Keyword Search. Exact/lexical matching via Postgres FTS (GIN-indexed) - complements semantic search for queries where exact terms matter more than meaning (error codes, identifiers, filenames).",
)
def search_keyword(q: str, top_k: int = 10, db: Session = Depends(get_db)):
    return keyword_search(db, q, top_k=top_k)


@router.get(
    "/search/context-aware",
    tags=["search"],
    summary="Context-aware search (query rewriting + hybrid search)",
)
def search_context_aware(
    q: str,
    history: list[str] = Query(
        default=[],
        description=(
            "Prior turns as alternating role:content strings, oldest first, "
            "e.g. ['user: what is FastAPI', 'assistant: a Python web framework']. "
            "Only the last 4 are used (same window POST /ask uses)."
        ),
    ),
    top_k: int = 10,
    db: Session = Depends(get_db),
):
    """
    AI Search: Context-Aware Search. The other four search types
    (semantic/keyword/hybrid/metadata) are directly callable and stateless;
    this one is inherently stateful - it needs conversation history to do
    anything a stateless search doesn't already do. POST /ask uses this same
    rewrite_query() + hybrid_search() combination internally, but that
    endpoint bundles it into full answer generation. This is the standalone
    equivalent: rewrite the follow-up in isolation and run the resulting
    query through hybrid search, so context-aware retrieval can be exercised
    and tested on its own rather than only as a side effect of asking a
    question.
    """
    turns = []
    for entry in history:
        role, _, content = entry.partition(":")
        turns.append({"role": role.strip() or "user", "content": content.strip() or entry})

    rewritten_query = rewrite_query(q, turns)
    results = hybrid_search(db, rewritten_query, top_k=top_k)
    return {"original_query": q, "rewritten_query": rewritten_query, "results": results}


@router.get(
    "/search/hybrid",
    tags=["search"],
    summary="Hybrid search (semantic + keyword, reciprocal rank fusion)",
    description="AI Search: Hybrid Search. Fuses semantic and keyword result lists via reciprocal rank fusion, then reranks the fused pool with a cross-encoder (ms-marco-MiniLM-L-6-v2) unless use_reranker=False. This is what /ask uses internally for retrieval.",
)
def search_hybrid(
    q: str,
    top_k: int = 10,
    category_filter: str | None = None,
    use_reranker: bool = True,
    db: Session = Depends(get_db),
):
    return hybrid_search(db, q, top_k=top_k, category_filter=category_filter, use_reranker=use_reranker)


@router.get(
    "/search/metadata",
    tags=["search"],
    summary="Filter documents by metadata (category/source/status/date)",
    description="AI Search: Metadata Search. No embedding or ranking involved - a direct Postgres filter over document metadata, for when the caller already knows what they're looking for (e.g. \"all SOPs updated this month\").",
)
def search_metadata(
    category: str | None = None,
    source_type: str | None = None,
    status: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    top_k: int = 10,
    db: Session = Depends(get_db),
):
    return metadata_search(
        db,
        category=category,
        source_type=source_type,
        status=status,
        date_from=date_from,
        date_to=date_to,
        top_k=top_k,
    )


@router.post(
    "/ask",
    tags=["qa"],
    summary="Ask a natural-language question (RAG with citations)",
    description="RAG answer generation. Rewrites the query using conversation history, retrieves via hybrid search, generates an answer with Gemini, verifies each citation against its source chunk, and logs the interaction (UsageLog) for analytics/review. Returns 429 if the Gemini free-tier quota is exhausted, 504 on an LLM timeout.",
)
def ask(payload: AskRequest, db: Session = Depends(get_db)):
    if payload.user_id is not None:
        ensure_valid_uuid(payload.user_id, status_code=400, detail="user_id must be a valid UUID")

    query = rewrite_query(payload.query, payload.history)
    try:
        result = generate_answer(db, query)
    except TimeoutError:
        logger.error("Answer generation timed out for query=%r.", payload.query)
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except GeminiQuotaExceededError as exc:
        logger.error("Gemini quota exceeded for query=%r: %s", payload.query, exc)
        raise HTTPException(
            status_code=429,
            detail="LLM provider quota exceeded — try again later or check your Gemini API plan/billing.",
        )

    verification = verify_citations(result["answer"], result["sources"])

    usage_log = UsageLog(
        user_id=payload.user_id,
        query=payload.query,
        answer=result["answer"],
        sources=result["sources"],
        retrieval_score=1.0 if verification["verified"] else 0.5,
        citation_verified=verification["verified"],
        citation_flags=verification["flags"],
        # Auto-flag unverified answers for the admin review queue; admins
        # can still clear or re-flag this via POST /admin/answers/{id}/review.
        flagged_for_review=not verification["verified"],
    )
    db.add(usage_log)
    db.commit()
    db.refresh(usage_log)

    return {**result, "citation_check": verification, "usage_log_id": usage_log.id}


@router.post(
    "/ask/{usage_log_id}/feedback",
    response_model=FeedbackResponse,
    tags=["qa"],
    summary="Record helpful/unhelpful feedback on a previous answer",
)
def submit_feedback(usage_log_id: str, payload: FeedbackRequest, db: Session = Depends(get_db)):
    """Lets a caller mark a previously generated answer (by the id returned
    from POST /ask) as helpful/unhelpful. This is what populates
    UsageLog.was_helpful, which GET /admin/analytics/usage reports on."""
    ensure_valid_uuid(usage_log_id, detail=f"usage log not found: {usage_log_id}")
    usage_log = db.get(UsageLog, usage_log_id)
    if not usage_log:
        raise HTTPException(status_code=404, detail=f"usage log not found: {usage_log_id}")

    usage_log.was_helpful = payload.was_helpful
    db.add(usage_log)
    db.commit()
    return FeedbackResponse(usage_log_id=usage_log_id, was_helpful=payload.was_helpful)


@router.post(
    "/documents/compare",
    response_model=CompareResponse,
    tags=["documents"],
    summary="Compare two documents",
    description="AI Capability: Compare Multiple Documents. Embedding-similarity score, a line-level diff, and an LLM narrative summary of what changed/differs between the two texts. 404 if either document_id doesn't exist, 502 if the LLM comparison step fails unexpectedly.",
)
def compare(payload: CompareRequest, db: Session = Depends(get_db)):
    doc_a = db.get(Document, payload.document_id_a)
    doc_b = db.get(Document, payload.document_id_b)

    missing_ids = [
        doc_id for doc_id, doc in (
            (payload.document_id_a, doc_a),
            (payload.document_id_b, doc_b),
        ) if doc is None
    ]
    if missing_ids:
        raise HTTPException(status_code=404, detail=f"document(s) not found: {', '.join(missing_ids)}")

    try:
        return compare_documents_full(doc_a.raw_text or "", doc_b.raw_text or "")
    except TimeoutError:
        logger.error(
            "Document comparison timed out for document_id_a=%s, document_id_b=%s.",
            payload.document_id_a, payload.document_id_b,
        )
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception:
        logger.exception(
            "Document comparison failed unexpectedly for document_id_a=%s, document_id_b=%s.",
            payload.document_id_a, payload.document_id_b,
        )
        raise HTTPException(status_code=502, detail="Document comparison failed. Please try again later.")


@router.get(
    "/documents/{document_id}/summary",
    response_model=SummaryResponse,
    tags=["documents"],
    summary="On-demand LLM summary of a document",
    description="AI Capability: Summarize Technical Documents. Generates a fresh summary from the document's full raw_text on every call (not cached). 404 if the document doesn't exist, 504 on an LLM timeout.",
)
def summarize(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    try:
        return {"summary": summarize_document_full(doc.raw_text or "")}
    except TimeoutError:
        logger.error("Document summary timed out for document_id=%s.", document_id)
        raise HTTPException(status_code=504, detail="LLM request timed out")

@router.get(
    "/documents/{document_id}/suggest-updates",
    tags=["documents"],
    summary="Suggest updates to a document from newer related content",
    description="AI Capability: Suggest Document Updates. Finds newer documents covering similar ground (via chunk-embedding similarity) and asks the LLM to name concrete facts in this document that look contradicted or superseded. Returns a 'nothing found' message, not an error, when there's no related newer content. 404 if the document doesn't exist, 502 if the LLM step fails unexpectedly.",
)
def suggest_updates(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    try:
        return suggest_document_updates(db, document_id)
    except TimeoutError:
        logger.error("Suggest-updates timed out for document_id=%s.", document_id)
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except Exception:
        logger.exception("Suggest-updates failed unexpectedly for document_id=%s.", document_id)
        raise HTTPException(status_code=502, detail="Suggest-updates failed. Please try again later.")

@router.get(
    "/graph/technology-map",
    tags=["knowledge-graph"],
    summary="Technology map for one entity label",
    description="Knowledge Intelligence: Technology Maps. Every RELATES_TO edge touching an entity of the given label (default TECH), with the derived relation type (DEPENDS_ON/CONNECTS_TO/DEPLOYS_TO/etc.) and confidence - see docs/knowledge_graph_schema.md.",
)
def technology_map(entity_label: str = "TECH"):
    return get_technology_map(entity_label)


@router.get(
    "/graph/skill-dependencies",
    tags=["knowledge-graph"],
    summary="Skill/technology dependency graph",
    description="Knowledge Intelligence: Skill Dependencies. Same underlying relationship data as technology-map, filtered/framed for prerequisite-style queries (e.g. what must be learned before this skill). Omit `skill` for the full dependency graph.",
)
def skill_dependencies(skill: str | None = None):
    return get_skill_dependencies(skill)


@router.get(
    "/graph/relationships/explain",
    tags=["knowledge-graph"],
    summary="Explain the relationship between two entities",
)
def relationship_explanation(source: str, target: str):
    """Traceable-source explanation for one edge: relation type, confidence,
    reasoning and evidence — see app/graph/relationships.py."""
    return explain_relationship(source, target)


@router.get(
    "/graph/learning-recommendations",
    tags=["knowledge-graph"],
    summary="LMS content recommendations from query history",
    description="Knowledge Intelligence: Learning Recommendations. Extracts entities from user_query_history and scopes the graph query to them (falls back to a generic recommendation list if history is empty or matches nothing).",
)
def learning_recommendations(user_query_history: list[str] = Query(default=[])):
    return recommend_learning_path(user_query_history)


@router.post(
    "/evaluation/run",
    tags=["evaluation"],
    summary="Run the retrieval evaluation set and log results to MLflow",
    description="Evaluation Framework. Runs every Q&A pair in app/evaluation/eval_set.json through hybrid search, computes precision@k/recall@k/MRR, and logs the run to MLflow (experiment 'ekie-retrieval-eval'). See docs/Evaluation_Report.md for the latest results and methodology.",
)
def evaluation_run(k: int = 10, db: Session = Depends(get_db)):
    return run_evaluation(db, k=k)
