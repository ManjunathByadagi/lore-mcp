-- Migration 005: Retrieval telemetry for hard negative mining (Issue #5, Phase 1)
--
-- Documentation only. This DDL is AUTO-APPLIED on startup by
-- LocalPostgresClient._init_schema() when LORE_HARD_NEGATIVE_MINING=true and a
-- PostgreSQL backend is configured. You do not need to run it by hand; it is
-- recorded here for reviewability and for environments that prefer to apply
-- migrations explicitly.
--
-- The statements below MUST stay byte-for-byte identical to
-- lore.telemetry.TELEMETRY_PG_SCHEMA (a unit test enforces this).
--
-- Idempotent: every statement uses CREATE ... IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS knowledge.retrieval_telemetry (
    query_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    topic TEXT,
    search_mode TEXT,
    retrieved_document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    result_count INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    parent_query_id TEXT REFERENCES knowledge.retrieval_telemetry(query_id) ON DELETE SET NULL,
    required_requery BOOLEAN NOT NULL DEFAULT FALSE,
    caller_agent TEXT,
    user_feedback_score INTEGER,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_session ON knowledge.retrieval_telemetry (session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_created ON knowledge.retrieval_telemetry (created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_parent ON knowledge.retrieval_telemetry (parent_query_id);
