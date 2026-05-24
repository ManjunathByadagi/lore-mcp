# Lore Semantic Search — Development Guide

## Local Development Setup

### Prerequisites

- Python 3.11 or 3.12 (the CI matrix covers both)
- Git
- A SQLite build with `enable_load_extension` support (standard CPython from python.org, pyenv, or conda)

### Install

```bash
git clone https://github.com/davidgut1982/lore-mcp.git
cd lore-mcp

# Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Install with all development and semantic extras
pip install -e ".[dev,semantic]"
```

The `dev` extra installs pytest, ruff, mypy, and pre-commit. The `semantic` extra installs sentence-transformers, sqlite-vec, onnxruntime, optimum, and pgvector.

### Running the Server Locally

```bash
export DB_BACKEND=sqlite
export SQLITE_DB_PATH=/tmp/lore-dev.db
export LORE_SEMANTIC_SEARCH=true

# stdio mode (for MCP client testing)
lore-mcp

# HTTP/SSE mode
lore-mcp --host 127.0.0.1 --port 8000
```

---

## Testing Strategy

### Test Layers

| Layer | Location | What it covers | When to run |
|---|---|---|---|
| Unit — RRF | `tests/test_rrf.py` | `reciprocal_rank_fusion`, `candidate_pool_size`, `rrf_k`, tie-breaking | Every commit |
| Unit — degradation | `tests/test_degradation.py` | Flag gates, fallback behavior, `compute_content_hash` | Every commit |
| Unit — basic | `tests/test_basic.py` | Core KB CRUD paths | Every commit |
| Integration | `tests/integration/test_sqlite_embeddings.py` | Full SQLite + FTS5 + sqlite-vec + model load path | Gated: `LORE_SEMANTIC_SEARCH=true` + deps installed |

### Running Tests

```bash
# Fast unit tests only (no model load)
pytest tests/ -v --tb=short -m "not slow and not integration"
# Or simply: pytest tests/ (integration tests skip automatically without the env var)

# Full test suite including integration (requires [semantic] deps and the env var)
export LORE_SEMANTIC_SEARCH=true
pytest tests/ -v --tb=short

# Single test file
pytest tests/test_rrf.py -v

# Integration tests only
pytest tests/integration/ -v --tb=short -m integration
```

Integration tests are decorated with `pytest.mark.slow`, `pytest.mark.integration`, and `pytest.mark.skipif(not _semantic_available(), ...)`. They skip cleanly when `LORE_SEMANTIC_SEARCH` is not `true` or when sentence-transformers / sqlite-vec are not installed.

### What the Tests Cover

**`test_rrf.py`** — pure logic, no I/O:
- Empty input, single list, multi-list overlap
- Tie-breaking determinism (`(-score, kb_id)` secondary sort key — the Critic-required fix)
- Duplicate IDs in a single ranking only count once
- Parametrized `candidate_pool_size` table for 5 corpus/top_k combinations
- `LORE_RRF_K` env var parsing (default, invalid value fallback, explicit `k=` override)

**`test_degradation.py`** — flag and gate behavior:
- `LORE_SEMANTIC_SEARCH=false` default
- Various true/false string formats (`TRUE`, `True`, `0`, `yes` — only `true` activates)
- `get_embedder()` raises `EmbeddingUnavailableError` when disabled
- `fts5_search_sqlite` and `vector_search_sqlite` short-circuit when extension flags are False
- `compute_content_hash` stability and None-tolerance

**`tests/integration/test_sqlite_embeddings.py`** — end-to-end against real SQLite:
- `vec_extension_loaded=True`, `fts5_available=True` after init
- `kb_add` produces `embedded=True` and populates meta row
- `kb_update` changes content_hash when content changes; skips re-embed when unchanged
- `kb_delete` removes vec0 + meta rows after KB row is gone (delete order: KB first, vec0 second)
- FTS5 search finds exact term matches
- Semantic search surfaces meaning-related entries over unrelated ones
- Hybrid search returns RRF scores on results
- Backfill is idempotent (second run embeds zero)
- Backfill lock rejects concurrent runs
- `kb_embedding_status` reports coverage accurately
- Legacy lexical path still works with `LORE_SEMANTIC_SEARCH=false`

---

## Adding a New Backend

To add a new vector store backend (e.g., MySQL, MariaDB, Cloudflare D1):

### 1. Create a new client class in `db_client.py`

The class must expose:
```python
self.vec_extension_loaded: bool  # True when vector search is available
self.fts5_available: bool         # True when FTS5/full-text search is available
def _get_connection(self): ...    # Return the underlying DB connection
def table(self, name: str): ...   # Return a query builder (TableQuery interface)
```

Pattern: follow `SqliteClient` (line ~1332) as the reference. The `vec_extension_loaded` and `fts5_available` flags are the only interface `server.py` and `search.py` need.

### 2. Add backend-specific search primitives to `search.py`

`search.py` currently has:
- `fts5_search_sqlite(db_client, query, topic, limit) -> list[dict]`
- `vector_search_sqlite(db_client, query_vector, topic, limit) -> list[dict]`
- `hybrid_search_sqlite(db_client, query, ...)` — orchestrates the two above

Add analogous functions for the new backend, e.g.:
- `fts_search_mysql(...)`
- `vector_search_mysql(...)`
- `hybrid_search_mysql(...)`

Keep `reciprocal_rank_fusion` and `candidate_pool_size` as-is — they are backend-agnostic.

### 3. Route in `server.py`

In `handle_kb_search` (line ~1009), extend the backend detection:
```python
backend = os.getenv("DB_BACKEND", "").strip().lower()
is_sqlite = backend == "sqlite"
is_mysql  = backend == "mysql"   # new
```

Add the write path in `_semantic_write_enabled` and `_embed_kb_entry` — these currently early-return for non-SQLite.

### 4. Add schema migration

Add the embedding table DDL to the new backend's schema initialization. The PostgreSQL schema in `LocalPostgresClient._init_schema()` (line ~118) is the reference for a relational backend.

### 5. Add integration tests

Create `tests/integration/test_<backend>_embeddings.py` following `test_sqlite_embeddings.py`. At minimum cover: extension loads, `kb_add` embeds, `kb_delete` removes vec rows, hybrid search fuses results.

---

## Adding a New Embedding Model

### Compatibility Requirements

Before changing the default or adding a new model:

1. **Dimension must be 384** — the SQLite `vec0` table and PostgreSQL `kb_embeddings` column are created with `FLOAT[384]` / `halfvec(384)`. A different dimension requires dropping and recreating these tables and running a full backfill. If you want to support a different dimension, make `EMBEDDING_DIM` dynamic and add a migration path.

2. **ONNX export available** — check the model's HuggingFace page for an ONNX export. If none exists, set `LORE_EMBEDDING_BACKEND=torch`.

3. **Max token length** — `all-MiniLM-L6-v2` truncates at 512 tokens (~350 words). For longer documents, a model with a larger context window is better.

4. **License** — model must be compatible with MIT-licensed software. Check the HuggingFace model card.

### Testing a New Model

```bash
# Load the candidate model manually
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('your/model', backend='onnx')
v = model.encode('test sentence', convert_to_numpy=True, normalize_embeddings=True)
print('dim:', len(v), 'type:', type(v[0]))
"

# Run integration tests with the new model
export LORE_EMBEDDING_MODEL=your/model
export LORE_SEMANTIC_SEARCH=true
pytest tests/integration/ -v
```

### Where to Change the Default

- `embeddings.py`, line `DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"` — change this constant.
- `pyproject.toml` — update the model name in any documentation strings.
- `README.md` — update the configuration table.
- This document and `docs/design-decisions.md`.

---

## Adding a New Search Mode

Current modes: `fts`, `semantic`, `hybrid`. To add a new mode (e.g., `rerank` for cross-encoder reranking):

1. **`search.py`** — add a new orchestration function (e.g., `rerank_search_sqlite(...)`) and handle it in `hybrid_search_sqlite` or a new top-level function.

2. **`server.py`** — add the new mode to the `search_mode` enum in the `kb_search` tool definition (line ~244) and add routing in `handle_kb_search` (line ~1032).

3. **`search.py: default_search_mode()`** — add the new mode to the validation set if it should be a valid default.

4. **Tests** — add unit tests for the new fusion/ranking logic and an integration test for the end-to-end path.

---

## CI Matrix

Every push and pull request to `main` runs `.github/workflows/ci.yml`:

| Step | Command | Notes |
|---|---|---|
| Lint | `ruff check src/` | Fails on any ruff warning |
| Format check | `ruff format src/ --check` | Fails on unformatted code |
| Tests (Python 3.11) | `pytest tests/ -v --tb=short` | env: `DB_BACKEND=sqlite` |
| Tests (Python 3.12) | `pytest tests/ -v --tb=short` | env: `DB_BACKEND=sqlite` |

Integration tests (`tests/integration/`) run in CI only when `LORE_SEMANTIC_SEARCH=true` is set. The current CI matrix does NOT set this variable, so integration tests skip automatically. They are run locally before releases.

**The full "gauntlet"** (run before tagging a release):
```bash
export LORE_SEMANTIC_SEARCH=true
export DB_BACKEND=sqlite

# Lint + format
ruff check src/
ruff format src/ --check

# Full test suite including integration
pytest tests/ -v --tb=short

# Type check
mypy src/lore/ --ignore-missing-imports
```

---

## Release Process

1. **Version bump** — update `version` in `pyproject.toml` and `__version__` in `src/lore/__init__.py`.

2. **CHANGELOG** — add an entry under `## [X.Y.Z] - YYYY-MM-DD` with Added/Changed/Fixed/Deferred sections. Follow the v0.6.0 entry as the template.

3. **Run the gauntlet** — see above. Integration tests must pass.

4. **Commit and tag:**
   ```bash
   git add pyproject.toml src/lore/__init__.py CHANGELOG.md
   git commit -m "chore: release vX.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```

5. **Create GitHub release** — `gh release create vX.Y.Z --title "vX.Y.Z" --notes-from-tag`. This triggers `.github/workflows/publish.yml`, which builds the package and publishes to PyPI via Trusted Publisher (OIDC — no API token required).

---

## Phase Boundaries

### v0.6.0 — MVP (shipped)

- Semantic search for SQLite backend only
- `all-MiniLM-L6-v2` ONNX default model
- FTS5 + sqlite-vec + RRF hybrid search
- `kb_backfill_embeddings` and `kb_embedding_status` tools
- Graceful degradation when any component unavailable
- Unit tests: RRF, degradation, flag gates
- Integration tests: full SQLite end-to-end

### v0.7.0 — Phase 2 (in progress at time of writing)

- **PostgreSQL semantic path** — write (`_embed_kb_entry` for Postgres), read (`vector_search_postgres`), and routing in `handle_kb_search` for non-SQLite backends
- The schema (`knowledge.kb_embeddings`, HNSW index) is already auto-applied by `LocalPostgresClient._init_schema()` — only the Python integration is pending
- PostgreSQL `fts5_available` equivalent (using `websearch_to_tsquery` or `plainto_tsquery`)

### Deferred (no milestone assigned)

| Feature | Tracking | Notes |
|---|---|---|
| `kb_reindex_embeddings` | Issue #6 | Full re-embedding on model change; backfill + cleanup in one operation |
| Cross-encoder reranker | Issue #6 | Second-pass reranking of RRF top-N using a cross-encoder model |
| `include_content` param on `kb_search` | Issue #6 | Return full entry content in search results (currently stripped for payload size) |
| Telemetry + hard-negative mining | Issue #5 | Collect query/click data to improve model fine-tuning |
| FastMCP migration | Issue #7 | Replace raw `mcp` library with FastMCP; enables clean lifespan hooks for model pre-loading |

---

## End-to-end testing

The `tests/e2e/` package provides a full HTTP-level test harness that runs
against a deployed Lore instance.  Tests are automatically **skipped** when no
live endpoint is configured, so they are safe to collect in CI at all times.

### Prerequisites

```
pip install -e ".[dev]"          # includes httpx, pyyaml, pytest
```

The harness does **not** require the `[semantic]` extra — it talks to the server
over HTTP and never loads the embedding model locally.

### Running tests

```bash
# Against the staging LXC
LORE_E2E_URL=http://lore-staging:5555 pytest tests/e2e/ -v

# Against a local dev server on port 5555
LORE_E2E_URL=http://localhost:5555 pytest tests/e2e/ -v

# Or use the Makefile shortcuts
make e2e-staging
make e2e-local
```

### Test structure

| File | Purpose |
|---|---|
| `client.py` | `LoreClient` — thin synchronous `httpx` wrapper around JSON-RPC 2.0 |
| `conftest.py` | Fixtures: `client`, `session_client`, `cleanup_topic`, `unique_id` |
| `test_smoke.py` | Service reachability, tools/list shape, embedding-status shape |
| `test_kb_lifecycle.py` | CRUD cycle: add → get → update → delete |
| `test_search_modes.py` | Response-shape assertions for `fts`, `semantic`, `hybrid` modes |
| `test_regression.py` | Corpus-driven rank/match regression suite |
| `test_backfill.py` | Backfill idempotency and coverage non-regression |
| `regression_corpus.yaml` | Seed articles + ranked queries for regression tests |
| `soak_runner.py` | Standalone 24–48 h continuous-load runner (not a pytest file) |

### Regression corpus

`regression_corpus.yaml` contains seed articles and per-article queries.  Each
query specifies:

- `mode` — `fts`, `semantic`, or `hybrid`
- `expect_match` — `true` if the seeded article should appear in results
- `max_rank` — maximum acceptable 1-based rank (when `expect_match: true`)

False-negative queries (`expect_match: false`) verify that unrelated content
does not contaminate results.

To add a new regression case, append an entry to the YAML and re-run:

```bash
LORE_E2E_URL=http://lore-staging:5555 pytest tests/e2e/test_regression.py -v
```

### Soak test

The soak runner exercises the service continuously for 24–48 hours.  It is
**not** a pytest file — run it directly:

```bash
# Quick smoke soak (10 minutes)
LORE_E2E_URL=http://lore-staging:5555 \
  python3 -m tests.e2e.soak_runner --duration 10m --log-file /tmp/soak.jsonl

# Full 24-hour staging run (use tmux or nohup)
make soak-staging
```

#### Failure thresholds

| Metric | Default | Flag |
|---|---|---|
| Error rate (5-min rolling window) | > 1 % | `--error-threshold 1.0` |
| P95 latency (any operation) | > 5 000 ms | `--latency-p95-ms 5000` |

The runner exits non-zero (code 1) as soon as either threshold is breached.  Use
`--window` to change the rolling window from the default 5 minutes.

#### Structured logging

Every event emitted by the soak runner is a single-line JSON object:

```json
{"ts": "2026-05-24T12:00:00+00:00", "level": "INFO", "msg": "event:op",
 "event": {"type": "op", "op": "kb_search_hybrid", "ok": true, "latency_ms": 42.3}}
```

Parse with `jq` for ad-hoc analysis:

```bash
jq 'select(.event.type == "metrics_summary")' /tmp/soak.jsonl
```

### CI integration

The e2e suite is designed to be gated behind a `[e2e]` CI label or a separate
pipeline stage.  The `LORE_E2E_URL` variable must be set for tests to actually
execute; without it all tests are collected but skipped with a clear message:

```
SKIPPED [reason] LORE_E2E_URL not set; e2e tests require a live Lore endpoint
```

Add a dedicated job in your CI configuration:

```yaml
e2e:
  needs: [deploy-staging]
  env:
    LORE_E2E_URL: http://lore-staging:5555
  run: |
    pip install -e ".[dev]"
    pytest tests/e2e/ -v --tb=short
```
