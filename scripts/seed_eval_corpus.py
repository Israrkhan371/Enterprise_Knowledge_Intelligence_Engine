"""
Ingests the seed_data/ corpus so app/evaluation/eval_set.json's 40 Q&A
pairs have real documents to resolve against and score.

Why this exists: eval_set.json was written with 40 realistic queries and
target document titles, but the referenced documents (e.g. "SOP - Mentor
Onboarding", "EKIE API Documentation") never actually existed anywhere in
the repo. run_evaluation() resolves relevant_document_titles against the
live `documents` table (see app/evaluation/eval.py::resolve_relevant_ids),
so with nothing ingested under those titles every query was silently
skipped rather than scored — the eval set looked complete but had nothing
to evaluate against.

This script ingests one real, content-complete document per title (see
seed_data/) through the actual ingestion pipeline (app.ingestion.pipeline.
ingest_document) — same code path as a real upload, so chunking, embedding,
vector-store writes, and graph population all happen exactly as they would
for any other document.

Requires live Postgres, ChromaDB, and Neo4j (the same services EKIE's
tests skip against when unavailable — see tests/test_metadata.py). Run
after `docker compose up`:

    python -m scripts.seed_eval_corpus

Idempotent-ish: re-running creates duplicate Document rows per title
(there's no upsert-by-title here, deliberately — see the docstring below
for why). If you need a clean slate, wipe the documents/document_chunks
tables (and the matching ChromaDB collection) first.

Note: the four "github_repositories" eval titles (app/api/routes.py,
app/graph/extract.py, app/ingestion/pipeline.py, app/search/hybrid.py)
are NOT seeded here — they resolve automatically once this repo itself is
ingested via ingest_github_repo(), since that path sets Document.title to
the file's path, matching those eval titles exactly.

The "SOP - Intern Offboarding" eval title (category: gap_detection) is
also deliberately NOT seeded — that query exists to test that the system
correctly reports missing documentation, so no such document should exist.
"""
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.models import Category, Document
from app.ingestion.pipeline import ingest_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED_DATA_DIR = Path(__file__).parent.parent / "seed_data"

# (relative path under seed_data/, title, source_type, category name)
# Title must match eval_set.json's relevant_document_titles exactly.
SEED_DOCUMENTS = [
    # sop
    ("sop/mentor_onboarding.md", "SOP - Mentor Onboarding", "sop", "sop"),
    ("sop/escalation_procedures.md", "SOP - Escalation Procedures", "sop", "sop"),
    ("sop/incident_response.md", "SOP - Incident Response", "sop", "sop"),
    ("sop/access_request_process.md", "SOP - Access Request Process", "sop", "sop"),
    # coding_standards
    ("coding_standards/code_review_process.md", "SOP - Code Review Process", "sop", "coding_standards"),
    ("coding_standards/python_coding_standards.md", "Ezitech Python Coding Standards", "markdown", "coding_standards"),
    # internship_case_studies
    ("internship_case_studies/ezitech_engineering_framework_overview.md", "Ezitech Engineering Framework Overview", "markdown", "internship_case_studies"),
    ("internship_case_studies/intern_onboarding_checklist.md", "Intern Onboarding Checklist", "markdown", "internship_case_studies"),
    ("internship_case_studies/case_study_ai007.md", "Case Study - AI-007 Enterprise Knowledge Intelligence Engine", "markdown", "internship_case_studies"),
    # company_policies
    ("company_policies/data_retention.md", "Company Policy - Data Retention", "markdown", "company_policies"),
    ("company_policies/third_party_ai_usage.md", "Company Policy - Third-Party AI Usage", "markdown", "company_policies"),
    ("company_policies/remote_work.md", "Company Policy - Remote Work", "markdown", "company_policies"),
    # database_schemas
    ("database_schemas/ekie_database_schema.sql", "EKIE Database Schema", "db_schema", "database_schemas"),
    ("database_schemas/knowledge_graph_schema.md", "Knowledge Graph Schema", "markdown", "database_schemas"),
    # api_documentation
    ("api_documentation/ekie_api_openapi.json", "EKIE API Documentation", "api_docs", "api_documentation"),
    # lms_courses
    ("lms_courses/intro_to_rag_systems.html", "LMS - Introduction to RAG Systems", "lms", "lms_courses"),
    ("lms_courses/vector_databases_and_embeddings.html", "LMS - Vector Databases and Embeddings", "lms", "lms_courses"),
    ("lms_courses/fastapi_fundamentals.html", "LMS - FastAPI Fundamentals", "lms", "lms_courses"),
    # research_papers
    ("research_papers/reciprocal_rank_fusion.md", "Reciprocal Rank Fusion - Research Notes", "markdown", "research_papers"),
    ("research_papers/dense_vs_sparse_retrieval.md", "Dense vs Sparse Retrieval - Research Notes", "markdown", "research_papers"),
    # meeting_notes
    ("meeting_notes/week2_architecture_review.md", "Meeting Notes - Week 2 Architecture Review", "meeting_notes", "meeting_notes"),
    ("meeting_notes/sprint_planning_knowledge_graph.md", "Meeting Notes - Sprint Planning Knowledge Graph", "meeting_notes", "meeting_notes"),
    # transcripts
    ("transcripts/mentor_session_chunking_strategy.vtt", "Transcript - Mentor Session on Chunking Strategy", "transcript", "transcripts"),
    ("transcripts/retrieval_evaluation_walkthrough.vtt", "Transcript - Retrieval Evaluation Walkthrough", "transcript", "transcripts"),
    # technical_blogs
    ("technical_blogs/choosing_a_vector_database.html", "Blog - Choosing a Vector Database", "blog", "technical_blogs"),
    ("technical_blogs/prompt_engineering_best_practices.html", "Blog - Prompt Engineering Best Practices", "blog", "technical_blogs"),
]


def _get_or_create_category(db: Session, name: str) -> Category:
    category = db.query(Category).filter(Category.name == name).first()
    if category is None:
        category = Category(name=name)
        db.add(category)
        db.commit()
        db.refresh(category)
    return category


def seed() -> None:
    db = SessionLocal()
    ingested, failed = 0, 0
    try:
        for rel_path, title, source_type, category_name in SEED_DOCUMENTS:
            file_path = SEED_DATA_DIR / rel_path
            if not file_path.exists():
                logger.error("Missing seed file, skipping: %s", file_path)
                failed += 1
                continue

            category = _get_or_create_category(db, category_name)
            document = Document(
                title=title,
                source_type=source_type,
                source_uri=str(file_path),
                category_id=category.id,
                uploaded_by="seed_eval_corpus",
            )
            try:
                ingest_document(db, document, str(file_path))
                logger.info("Ingested: %s", title)
                ingested += 1
            except Exception:
                logger.exception("Failed to ingest %r", title)
                db.rollback()
                failed += 1

        logger.info("Done. Ingested %d document(s), %d failed.", ingested, failed)
        logger.info(
            "Next: run POST /api/v1/evaluation/run to score retrieval against "
            "eval_set.json (semantic/keyword/hybrid search don't filter on "
            "Document.status by default, so no approval step is required first "
            "— only /search/metadata does, if you pass status= explicitly)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed()
