import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Float, Integer, Index, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


def gen_uuid():
    return str(uuid.uuid4())


class Category(Base):
    __tablename__ = "categories"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=True)

    documents = relationship("Document", back_populates="category")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    title = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # pdf, docx, github, transcript, sop, etc.
    source_uri = Column(String, nullable=True)
    category_id = Column(UUID(as_uuid=False), ForeignKey("categories.id"), nullable=True)
    raw_text = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, approved, rejected, stale
    quality_score = Column(Float, nullable=True)
    uploaded_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # --- Bonus: Document Version Intelligence (app/rag/version_intelligence.py) ---
    # Self-referential link set by an admin via link_version(), not inferred
    # automatically - detect_version_candidates() only *suggests* pairs.
    # A document with supersedes_id set is a newer version of that document;
    # version increments by 1 down the chain from the root (version=1).
    supersedes_id = Column(UUID(as_uuid=False), ForeignKey("documents.id"), nullable=True)
    version = Column(Integer, default=1, nullable=False)

    category = relationship("Category", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    document_id = Column(UUID(as_uuid=False), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)  # id in the vector store

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        # Backs app.search.keyword.keyword_search()'s to_tsvector/plainto_tsquery
        # lookup. Without this, every keyword search does a sequential scan +
        # on-the-fly tsvector build over the whole table. create_all() creates
        # this automatically on Postgres; it's a no-op on other dialects.
        Index(
            "chunks_fts_idx",
            func.to_tsvector("english", text),
            postgresql_using="gin",
        ),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)
    role = Column(String, default="member")  # member, mentor, admin
    created_at = Column(DateTime, default=datetime.utcnow)


class ApprovalLog(Base):
    __tablename__ = "approval_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    document_id = Column(UUID(as_uuid=False), ForeignKey("documents.id"), nullable=False)
    reviewer = Column(String, nullable=True)
    decision = Column(String, nullable=False)  # approved, rejected
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    retrieval_score = Column(Float, nullable=True)
    was_helpful = Column(Boolean, nullable=True)
    # Snapshot of the [n]-indexed source chunks the answer was generated
    # from (same shape as generate_answer()'s "sources"), so the admin
    # answer-review UI can show exactly what the model saw without
    # re-running retrieval against a corpus that may have since changed.
    sources = Column(JSON, nullable=True)
    # Cached output of app.rag.citation_check.verify_citations() at
    # answer time, so the review queue doesn't have to recompute
    # embeddings for every historical answer just to render a list.
    citation_verified = Column(Boolean, nullable=True)
    citation_flags = Column(JSON, nullable=True)
    # True whenever citation verification failed at answer time, or an
    # admin has explicitly flagged the answer during review. Drives the
    # "needs attention" admin queue independently of `reviewed`.
    flagged_for_review = Column(Boolean, default=False, nullable=False)
    # Whether an admin has recorded a review decision for this answer
    # (see AnswerReviewLog). Distinct from `flagged_for_review`, which
    # tracks the outcome rather than whether review happened at all.
    reviewed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnswerReviewLog(Base):
    """Audit trail of admin review decisions on AI-generated answers.

    Mirrors ApprovalLog's role for documents: one append-only row per
    review action, while UsageLog.reviewed/flagged_for_review hold the
    current state for cheap filtering/listing.
    """
    __tablename__ = "answer_review_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    usage_log_id = Column(UUID(as_uuid=False), ForeignKey("usage_logs.id"), nullable=False)
    reviewer = Column(String, nullable=True)
    decision = Column(String, nullable=False)  # approved, flagged, dismissed
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)