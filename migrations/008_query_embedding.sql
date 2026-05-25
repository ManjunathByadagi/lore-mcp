-- Phase 4a: Store query embedding alongside retrieval telemetry for re-ranking
ALTER TABLE knowledge.retrieval_telemetry
    ADD COLUMN IF NOT EXISTS query_embedding halfvec(384);

-- NOTE: The HNSW index is NOT applied here.
-- Run via the backfill_query_embeddings tool (with --index flag)
-- to avoid blocking server startup on large tables.
-- Command: CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_retrieval_telemetry_qemb_hnsw
--          ON knowledge.retrieval_telemetry
--          USING hnsw (query_embedding halfvec_cosine_ops)
--          WITH (m = 16, ef_construction = 64);
