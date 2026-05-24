-- Migration 004: Dual-config FTS index for technical content (Issue #10)
--
-- The existing idx_kb_search index uses the 'english' text-search config, which
-- treats dotted identifiers like asyncio.gather as a single token.  When a user
-- queries asyncio gather (two separate words) the index cannot match.
--
-- This migration adds a second GIN index using the 'simple' config combined with
-- regexp_replace to split on common technical separators (. , / \ : _ -).
-- The updated fts_search_postgres query ORs both index conditions so that either
-- the English-stemmed path OR the simple-tokenized path can produce a match.
--
-- Idempotent: CREATE INDEX IF NOT EXISTS is safe to run multiple times.

CREATE INDEX IF NOT EXISTS idx_kb_search_simple
    ON knowledge.kb_entries
    USING gin(to_tsvector('simple', regexp_replace(
        coalesce(title,'') || ' ' || coalesce(content,''),
        '[.,/\:_-]', ' ', 'g')));
