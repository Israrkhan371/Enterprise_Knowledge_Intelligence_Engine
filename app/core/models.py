import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Float, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

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
    created_at = Column(DateTime, default=datetime.utcnow)
