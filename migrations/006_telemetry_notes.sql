-- Migration 006: Add notes column to retrieval_telemetry (Issue #5 Phase 2)
-- Idempotent: ADD COLUMN IF NOT EXISTS
ALTER TABLE knowledge.retrieval_telemetry
    ADD COLUMN IF NOT EXISTS notes TEXT;
