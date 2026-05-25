# Lore Semantic Search — Operations Guide

## Deployment Checklist

### Fresh Install (v0.6.0+)

```bash
# 1. Install base package
pip install lore-knowledge-mcp

# 2. Install semantic extras (optional but required for semantic/hybrid search)
pip install "lore-knowledge-mcp[semantic]"

# 3. Set environment (minimum for semantic search)
export DB_BACKEND=sqlite
export SQLITE_DB_PATH=/var/lib/lore/kb.db
export LORE_SEMANTIC_SEARCH=true

# 4. Start the server
lore-mcp
# or for HTTP/SSE mode:
lore-mcp --host 0.0.0.0 --port 8000

# 5. Verify startup: look for these log lines
#   "SqliteClient initialised at /var/lib/lore/kb.db (vec=True, fts5=True)"
#   "Loading embedding model sentence-transformers/all-MiniLM-L6-v2 (backend=onnx, ...)"
```

The model downloads on first `kb_search` or `kb_add` call (not at startup). First call takes ~2s for model load; subsequent calls are fast.

### Upgrade from v0.5.x

```bash
# 1. Bring the service down (via your process manager, e.g. systemctl or supervisorctl)

# 2. Upgrade the package
pip install --upgrade lore-knowledge-mcp
# If enabling semantic search:
pip install --upgrade "lore-knowledge-mcp[semantic]"

# 3. Add environment variables if enabling semantic search (see table below)
#    No DB migration needed: the schema is applied via idempotent CREATE IF NOT EXISTS
#    on first connection. FTS5 triggers are added automatically.

# 4. Bring the service back up

# 5. Backfill embeddings for existing entries (if enabling semantic search)
#    Call via MCP tool:
#    kb_backfill_embeddings(batch_size=32)
#    Or check coverage first:
#    kb_embedding_status()
```

**No data migration required.** The v0.6.0 schema changes are additive (`CREATE TABLE IF NOT EXISTS`, `CREATE VIRTUAL TABLE IF NOT EXISTS`, `CREATE TRIGGER IF NOT EXISTS`). Existing entries survive the upgrade. FTS5 index and vec0 table start empty; backfill populates them.

**Important:** After upgrade, existing entries are searchable via the legacy lexical path immediately. Hybrid/semantic search requires backfill to be run first.

---

## Environment Variables

| Variable | Default | Controls | When to change |
|---|---|---|---|
| `LORE_SEMANTIC_SEARCH` | `false` | Master switch for all semantic features | Set `true` to enable semantic/hybrid search |
| `LORE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Which HuggingFace model to use for embeddings | Switch to multilingual or higher-quality model |
| `LORE_EMBEDDING_BACKEND` | `onnx` | sentence-transformers inference backend (`onnx` or `torch`) | Set `torch` if ONNX export unavailable for chosen model |
| `LORE_EMBEDDING_CACHE_DIR` | `""` (HuggingFace default `~/.cache/huggingface`) | Local directory for model file cache | Set when running in air-gapped environment with pre-downloaded model |
| `LORE_EMBEDDING_BATCH_SIZE` | `32` | Number of texts encoded per `model.encode()` call | Reduce if OOM during backfill; increase for faster backfill on high-RAM hosts |
| `LORE_RRF_K` | `10` | Reciprocal Rank Fusion smoothing constant | Increase to 30-60 for corpora >10k entries |
| `LORE_SEARCH_MODE_DEFAULT` | `hybrid` | Default `search_mode` when caller omits it and semantic is on | Set `fts` for always-lexical, `semantic` for always-vector |
| `LORE_DEBUG_SEARCH` | `false` | Log per-request search stats (mode, query, FTS count, vec count, pool, corpus size) | Enable temporarily during troubleshooting |
| `DB_BACKEND` | `sqlite` | Database backend: `sqlite` or `postgres` | Set `postgres` for team deployments with PostgreSQL |
| `SQLITE_DB_PATH` | `$KNOWLEDGE_DATA_DIR/knowledge.db` | Path to SQLite database file | Change when running multiple Lore instances |
| `KNOWLEDGE_DATA_DIR` | OS-dependent | Base directory for SQLite file and other data | Override when data directory is non-default |

**PostgreSQL variables** (when `DB_BACKEND=postgres`):

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `knowledge` | Database name |
| `POSTGRES_USER` | - | Required |
| `POSTGRES_PASSWORD` | - | Required |

---

## Backfill Operations

Backfill is needed when:
- Upgrading from v0.5.x with existing entries and enabling semantic search
- Changing `LORE_EMBEDDING_MODEL` (all entries must be re-embedded with the new model)
- `LORE_SEMANTIC_SEARCH` was off when entries were added

### Running a Backfill

```python
# Dry run first - see how many entries need embedding
kb_backfill_embeddings(dry_run=True)
# Response: {"total_entries": 847, "needs_embedding": 412, "already_current": 435}

# Run the backfill
kb_backfill_embeddings(batch_size=32)
# Response: {"embedded": 412, "failed": 0, "already_current": 435}

# Verify coverage
kb_embedding_status()
# Response: {"total_entries": 847, "embedded": 847, "coverage_pct": 100.0}
```

### Estimating Time

Approximate throughput on CPU (single-core, `all-MiniLM-L6-v2`, ONNX backend):

| Batch size | Entries | Approximate time |
|---|---|---|
| 32 | 100 | ~10s |
| 32 | 1,000 | ~90s |
| 32 | 10,000 | ~15 min |

Reduce `batch_size` if you see OOM errors; increase it (up to 64-128) if RAM is available and you want faster throughput.

### Concurrent Backfill Protection

The backfill acquires a module-level `threading.Lock` (`_BACKFILL_LOCK`). If a second caller attempts backfill while one is running, it receives:

```json
{"ok": false, "message": "Another backfill is already running; try again shortly."}
```

### Partial Failure Recovery

If backfill fails mid-way (process killed, OOM, model error), it is safe to re-run. The content_hash guard skips already-embedded rows - only the rows that failed will be retried. Run `kb_embedding_status()` to confirm final coverage after recovery.

---

## Backend Selection

| Scenario | Use SQLite | Use PostgreSQL |
|---|---|---|
| Single-agent, local | Yes | - |
| Multi-agent, single host | Yes | Optional |
| Multi-agent, team (multiple hosts) | - | Yes |
| Air-gapped deployment | Yes | Yes |
| Requires HNSW ANN index | No (exact k-NN only) | Yes |
| KB > 50k entries | Marginal | Recommended |
| No database server available | Yes | - |

### SQLite to PostgreSQL Migration

No automated migration tool exists in v0.6.0. Manual process:
1. Export all entries: call `kb_search` repeatedly or use direct SQLite queries.
2. Set up PostgreSQL with the `knowledge` schema and pgvector extension.
3. Set `DB_BACKEND=postgres` and restart.
4. Re-add entries via `kb_add`.
5. Run `kb_backfill_embeddings` (not applicable in v0.6.0 - PostgreSQL semantic write path is Phase 2).

---

## PostgreSQL Prerequisites

> Note: The PostgreSQL semantic search path (Phase 2, v0.7.0) was in progress at the time of writing. The schema is auto-applied but the Python write/read integration is pending. This section documents what is required based on Issue #6 design.

### Requirements

1. **pgvector extension** installed in PostgreSQL:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **pgvector version >= 0.7** for `halfvec` support (50% storage savings):
   ```sql
   SELECT extversion FROM pg_extension WHERE extname = 'vector';
   ```
   If version < 0.7, Lore falls back to `vector(384)` automatically.

3. **`knowledge` schema** created:
   ```sql
   CREATE SCHEMA IF NOT EXISTS knowledge;
   ```

4. **User permissions**:
   ```sql
   GRANT USAGE ON SCHEMA knowledge TO lore_user;
   GRANT CREATE ON SCHEMA knowledge TO lore_user;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA knowledge TO lore_user;
   ```

5. **Install Python client** (`pgvector` adapter):
   ```bash
   pip install "lore-knowledge-mcp[semantic]"
   ```

### Auto-applied Schema (PostgreSQL)

When `LORE_SEMANTIC_SEARCH=true` and `DB_BACKEND=postgres`, `LocalPostgresClient._init_schema()` runs on first connection and creates:

```sql
CREATE TABLE IF NOT EXISTS knowledge.kb_embeddings (
    kb_id        TEXT PRIMARY KEY
                 REFERENCES knowledge.kb_entries(kb_id) ON DELETE CASCADE,
    embedding    halfvec(384) NOT NULL,  -- or vector(384) for pgvector < 0.7
    content_hash TEXT NOT NULL,
    model_name   TEXT NOT NULL,
    model_dims   INTEGER NOT NULL DEFAULT 384,
    embedded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_embeddings_hnsw
    ON knowledge.kb_embeddings
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

---

## Troubleshooting

### "model failed to load" / EmbeddingUnavailableError

**Symptoms:** `kb_search` with `semantic=True` returns `degraded=True`; log shows `EmbeddingUnavailableError` or `sentence-transformers is not installed`.

**Fix:**
```bash
# Verify the [semantic] extra is installed
pip show sentence-transformers onnxruntime
# If missing:
pip install "lore-knowledge-mcp[semantic]"

# Verify LORE_SEMANTIC_SEARCH is set
env | grep LORE_SEMANTIC_SEARCH

# If model download failed, try manually pre-downloading
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', backend='onnx')"
```

For air-gapped environments:
```bash
# On internet-connected machine, export the model to a local cache directory
# Then copy that directory to the air-gapped host and set:
export LORE_EMBEDDING_CACHE_DIR=/path/to/local-model-cache
```

### "sqlite-vec not loading"

**Symptoms:** Log shows `sqlite-vec not installed` or `Failed to load sqlite-vec`; `kb_embedding_status` shows `vec_extension_loaded: false`.

**Fix:**
```bash
# Verify package installed and version
python3 -c "import sqlite_vec; print(sqlite_vec.__version__)"
# Expected: 0.1.9

# If wrong version:
pip install "sqlite-vec==0.1.9"

# Verify Python SQLite supports load_extension
python3 -c "import sqlite3; c = sqlite3.connect(':memory:'); c.enable_load_extension(True); print('ok')"
# If this fails, your Python SQLite build has extensions disabled.
# Fix: use a different Python build (pyenv, conda, system package).
```

### "search returns nothing" (no results on hybrid/semantic queries)

**Symptoms:** `kb_search(query="...", search_mode="hybrid")` returns `count: 0` even though entries exist.

**Check in order:**

1. **Backfill not run:**
   ```python
   kb_embedding_status()  # Check coverage_pct
   kb_backfill_embeddings()  # Run if coverage < 100%
   ```

2. **Flag not set:**
   ```bash
   env | grep LORE_SEMANTIC_SEARCH
   # Must be "true"
   ```

3. **Model mismatch after model change:**
   ```python
   kb_embedding_status()
   # If model_name in output differs from current LORE_EMBEDDING_MODEL:
   kb_backfill_embeddings()  # Re-embeds with new model
   ```

4. **sqlite-vec not loaded:** See "sqlite-vec not loading" above.

### "search returns FTS only when I expect hybrid" (degraded mode)

**Symptoms:** Response includes `degraded: true` and `search_mode: "fts"` even though `search_mode="hybrid"` was requested.

**Background:** This was a Critic-identified bug in the original implementation (Issue #6, Fix 1). The symptom: `handle_kb_search` routed to the FTS-only fast path even when the caller specified hybrid mode, because the FTS fast path gate was evaluated before the semantic gate. The fix shipped in v0.6.0 checks `sqlite_vectors_ok` first and only falls to the FTS fast path if vectors are not available.

If you are still seeing this pattern:
- Confirm `vec_extension_loaded = True` (check startup logs: `vec=True`).
- Enable `LORE_DEBUG_SEARCH=true` and inspect the per-request log line:
  ```
  search[mode=hybrid] query="..." fts=12 vec=8 pool=60 corpus=120
  ```
  If `vec=0`, the vector pass is not running. Check vec extension state and whether backfill has been run.

### "performance degraded"

**Symptoms:** `kb_search` takes >500ms per query.

**Diagnosis:**
```bash
export LORE_DEBUG_SEARCH=true
# Observe log lines: fts=N, vec=N, pool=N, corpus=N
```

**Tuning options:**

| Problem | Fix |
|---|---|
| `pool` is large (>100) and corpus is large | `top_k` is large; reduce or accept the cost |
| `vec` pass is slow (corpus >50k) | Migrate to PostgreSQL + HNSW |
| Model load on every request | Warm the server with one query after startup |
| Large batch during backfill causes memory spikes | Reduce `LORE_EMBEDDING_BATCH_SIZE` |

### `semantic_enabled: false` after deployment

If `kb_embedding_status` reports `semantic_enabled: false` and `vec_extension_loaded: false`
immediately after a fresh deployment or migration, check PostgreSQL table ownership.
`_init_schema()` creates indexes on `knowledge.*` tables — PostgreSQL requires the executing
user to be the table owner for this, even when the index already exists.

**Diagnose:**
```sql
SELECT tablename, tableowner FROM pg_tables WHERE schemaname = 'knowledge';
```

All tables should be owned by the application user (e.g., `latvian_user`). If any are owned
by `postgres`, transfer ownership:

```sql
DO $$
DECLARE r RECORD;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'knowledge' LOOP
    EXECUTE 'ALTER TABLE knowledge.' || quote_ident(r.tablename) || ' OWNER TO latvian_user';
  END LOOP;
END $$;
```

Then restart the service: `systemctl restart lore.service`

> **Prevention**: Run this ownership transfer as part of every migration that creates new tables.

---

## Performance Tuning Reference

| Parameter | Default | Effect | Guidance |
|---|---|---|---|
| `LORE_RRF_K` | 10 | RRF smoothing constant | Increase to 30-60 for corpus >10k |
| `LORE_EMBEDDING_BATCH_SIZE` | 32 | Texts per `model.encode()` call | Increase for faster backfill on high-RAM hosts |
| Candidate pool | `min(corpus, 200, max(top_k*5, 50))` | How many candidates each pass fetches | Reduce `top_k` to shrink pool |
| HNSW `ef_construction` | 64 (PostgreSQL) | Build-time accuracy | Increase for large corpora; index-creation-time only |
| HNSW `m` | 16 (PostgreSQL) | Graph connectivity | Increase for higher recall; more memory |

---

## Rollback Procedure

If semantic search causes instability in production:

```
1. Set LORE_SEMANTIC_SEARCH=false in your .env or process manager config
2. Restart the service
3. Confirm startup log does NOT show vec=True / fts5=True
   kb_search falls back to legacy lexical path automatically.
```

The vec0 rows and embedding metadata in the database are preserved. Re-enabling `LORE_SEMANTIC_SEARCH=true` resumes from the last state - no backfill needed if content has not changed.
