-- Migration 009: trust_score column on KB entries (Issue #14)
-- Idempotent. Applied at startup by LocalPostgresClient._init_schema() in
-- db_client.py (the KB_ENTRIES_TRUST_SCORE_DDL constant mirrors this file
-- byte-for-byte for the PostgreSQL backend; a unit test enforces parity).
--
-- trust_score is a confidence signal in [0.0, 1.0] (default 1.0) that lets
-- stored facts carry a weight that can degrade or improve over time based on
-- usage and feedback. Existing rows pick up the DEFAULT 1.0 automatically, so
-- no backfill is required.

ALTER TABLE knowledge.kb_entries
    ADD COLUMN IF NOT EXISTS trust_score REAL DEFAULT 1.0;

-- SQLite (knowledge_kb_entries) gets the equivalent column from _SQLITE_SCHEMA /
-- _CORE_SQLITE_STATEMENTS on fresh databases and from an idempotent
-- PRAGMA-guarded ALTER TABLE for pre-existing databases (SQLite lacks
-- ADD COLUMN IF NOT EXISTS). See SqliteClient._migrate_trust_score in
-- db_client.py.
