-- Phase 2: PostgreSQL semantic search support
-- Issue #6 follow-up: enable kb_embeddings storage with pgvector HNSW index.
--
-- Requires the vector extension. pgvector >= 0.7.0 introduced the
-- halfvec type (2-byte half-precision float, ~50% storage vs full vector
-- with negligible recall loss for 384-d sentence-transformer embeddings).
-- LocalPostgresClient._init_schema() detects the installed version at runtime
-- and falls back to vector(384) when halfvec is unavailable.
--
-- This file documents the canonical schema for code review / migration
-- runners. In practice the table is created automatically on first server
-- start (see LocalPostgresClient._init_schema in src/lore/db_client.py).

CREATE TABLE IF NOT EXISTS knowledge.kb_embeddings (
    kb_id        TEXT PRIMARY KEY
                 REFERENCES knowledge.kb_entries(kb_id) ON DELETE CASCADE,
    embedding    halfvec(384) NOT NULL,
    content_hash TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    model_dims   INTEGER NOT NULL DEFAULT 384,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Add created_at to existing deployments (idempotent).
ALTER TABLE knowledge.kb_embeddings
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

-- HNSW cosine index. Matches vectors.file_embeddings convention in the same DB.
CREATE INDEX IF NOT EXISTS idx_kb_embeddings_hnsw
    ON knowledge.kb_embeddings
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_kb_embeddings_model
    ON knowledge.kb_embeddings (model_name);

COMMENT ON TABLE knowledge.kb_embeddings IS
    'Sentence-transformer embeddings for knowledge.kb_entries. '
    'One row per KB entry; CASCADE-deleted when entry is removed.';

-- Fallback for pgvector < 0.7 (no halfvec type):
-- Replace 'halfvec(384)' with 'vector(384)' and
-- 'halfvec_cosine_ops' with 'vector_cosine_ops' above.
