-- Migration 007: Hard negative pairs table (Issue #5 Phase 3)
-- Idempotent. Applied at startup by ensure_hard_negative_schema() in telemetry.py.
-- ON DELETE RESTRICT: pairs are training signal; KB entry deletion is blocked
-- until pairs are explicitly cleared via refresh_hard_negatives or manual DELETE.

CREATE TABLE IF NOT EXISTS knowledge.hard_negative_pairs (
    pair_id          TEXT PRIMARY KEY,
    query_text       TEXT NOT NULL,
    doc_id           TEXT NOT NULL REFERENCES knowledge.kb_entries(kb_id) ON DELETE RESTRICT,
    signal_type      TEXT NOT NULL CHECK (signal_type IN ('explicit', 'behavioral')),
    source_query_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hn_pairs_query_doc
    ON knowledge.hard_negative_pairs (query_text, doc_id);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_doc_id
    ON knowledge.hard_negative_pairs (doc_id);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_signal
    ON knowledge.hard_negative_pairs (signal_type);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_last_seen
    ON knowledge.hard_negative_pairs (last_seen_at);
