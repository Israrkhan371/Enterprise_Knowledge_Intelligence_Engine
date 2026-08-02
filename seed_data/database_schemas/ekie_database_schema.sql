-- EKIE Database Schema
-- PostgreSQL schema for the Enterprise Knowledge Intelligence Engine.
-- Source of truth is app/core/models.py; this dump mirrors it exactly.

CREATE TABLE categories (
    id UUID PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    description TEXT
);
COMMENT ON TABLE categories IS 'Admin-managed document categories (e.g. sop, coding_standards).';

CREATE TABLE documents (
    id UUID PRIMARY KEY,
    title VARCHAR NOT NULL,
    source_type VARCHAR NOT NULL, -- pdf, docx, github, transcript, sop, etc.
    source_uri VARCHAR,
    category_id UUID REFERENCES categories(id),
    raw_text TEXT,
    status VARCHAR DEFAULT 'pending', -- pending, approved, rejected, stale
    quality_score FLOAT,
    uploaded_by VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
COMMENT ON TABLE documents IS 'One row per ingested source document, across every knowledge source type.';
COMMENT ON COLUMN documents.status IS 'Lifecycle: pending -> approved/rejected via admin review, or stale via outdated-knowledge detection.';

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_id VARCHAR -- id of the corresponding vector in ChromaDB
);
COMMENT ON TABLE document_chunks IS 'Chunked text used for embeddings/RAG; embedding_id links back to the vector store.';

-- Backs keyword_search()'s to_tsvector/plainto_tsquery lookup.
CREATE INDEX chunks_fts_idx ON document_chunks USING gin (to_tsvector('english', text));

CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR NOT NULL UNIQUE,
    name VARCHAR,
    role VARCHAR DEFAULT 'member', -- member, mentor, admin
    created_at TIMESTAMP
);

CREATE TABLE approval_logs (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id),
    reviewer VARCHAR,
    decision VARCHAR NOT NULL, -- approved, rejected
    comment TEXT,
    created_at TIMESTAMP
);
COMMENT ON TABLE approval_logs IS 'Audit trail of admin approve/reject decisions on uploaded documents.';

CREATE TABLE usage_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    query TEXT NOT NULL,
    answer TEXT,
    retrieval_score FLOAT,
    was_helpful BOOLEAN,
    created_at TIMESTAMP
);
COMMENT ON TABLE usage_logs IS 'Every /ask query and answer, used for usage analytics and knowledge-gap detection.';

-- Relationships:
--   documents.category_id       -> categories.id       (many documents per category)
--   document_chunks.document_id -> documents.id         (many chunks per document)
--   approval_logs.document_id   -> documents.id         (many approval decisions per document, one per review)
--   usage_logs.user_id          -> users.id              (many queries per user)
--
-- The Neo4j knowledge graph (Document/Entity nodes) is a separate store,
-- linked by document.id — see "Knowledge Graph Schema" for that schema.
