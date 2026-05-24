# Lore Semantic Search — Design Decisions

This document is a decision log for the semantic search subsystem introduced in v0.6.0 (Issue #6). Each entry explains what was chosen, what was considered, and why — so future contributors can change decisions with full context, not just the outcome.

Source material: [Issue #6](https://github.com/davidgut1982/lore-mcp/issues/6) (design spec + Critic findings), the research report linked therein, and inline comments in `src/lore/embeddings.py`, `src/lore/search.py`, and `src/lore/db_client.py`.

---

## Decision: Default embedding model

**Choice:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ONNX backend)

**Alternatives considered:**
- `BAAI/bge-small-en-v1.5` (384d) — good MTEB scores, similar size
- `nomic-ai/nomic-embed-text-v1` (768d) — stronger on long documents
- `mixedbread-ai/mxbai-embed-large-v1` (1024d) — very high quality, larger
- `paraphrase-multilingual-MiniLM-L12-v2` (384d) — same family, multilingual

**Rationale:**
`all-MiniLM-L6-v2` is the model used by mcp-memory-service (the most widely deployed semantic MCP server), which provides cross-project validation. It is 22.7M parameters, approximately 90 MB on disk, and loads in under 2 seconds. It requires no license acknowledgement and is MIT-licensed. MTEB scores are solid for the use case (short-to-medium operational notes, runbooks, architecture decisions). Critically, it produces 384-d vectors — the same as the planned multilingual override — so the database schema and HNSW index do not need to change when switching.

**Trade-offs accepted:**
- Not optimal for documents >512 tokens (long runbooks will be truncated at the tokenizer level).
- English-centric; multilingual content requires the override model.
- Slightly lower retrieval quality than bge-small-v1.5 on some benchmarks, but the gap is negligible for typical KB entry lengths.

**Revisit if:** A user's KB is primarily non-English, or entries regularly exceed 512 tokens. Switch to `paraphrase-multilingual-MiniLM-L12-v2` (same schema) or a 768d+ model (requires vec0 table recreation and full backfill).

---

## Decision: ONNX inference backend

**Choice:** `sentence-transformers` with `backend="onnx"` (via `optimum[onnxruntime]`)

**Alternatives considered:**
- Full PyTorch (`torch`) backend — the default for sentence-transformers
- ONNX via direct `onnxruntime` calls without sentence-transformers

**Rationale:**
`sentence-transformers>=2.7` exposes `SentenceTransformer(model, backend="onnx")` which uses `onnxruntime` for inference without pulling PyTorch as a runtime dependency. PyTorch is 2–4 GB installed; `onnxruntime` is ~50 MB. For a server that runs as an MCP daemon, minimizing install footprint matters. The `[semantic]` extra therefore pins `optimum[onnxruntime]>=1.17.0` instead of `torch`. The code falls back to the default backend if `backend="onnx"` raises a `TypeError` or `ValueError` (older sentence-transformers versions), ensuring compatibility across versions.

**Trade-offs accepted:**
- ONNX model files are fetched from HuggingFace on first load if not cached; requires internet access on first run.
- Some custom models may not have ONNX exports available; users must set `LORE_EMBEDDING_BACKEND=torch` in those cases.
- ONNX inference is single-threaded by default in onnxruntime unless configured via `OMP_NUM_THREADS`.

**Revisit if:** A model with no ONNX export is needed, or if PyTorch becomes the lighter option (unlikely in the near term).

---

## Decision: sqlite-vec for vector storage

**Choice:** `sqlite-vec==0.1.9` (exact pin), `vec0` virtual table

**Alternatives considered:**
- `sqlite-vss` — the predecessor to sqlite-vec (older, less maintained)
- DuckDB embedded — full analytical engine with vector support
- Custom numpy-in-blob approach — serialize vectors as BLOBs, compute cosine in Python

**Rationale:**
sqlite-vec is purpose-built for SQLite vector search. It integrates as a loadable extension (no separate process), supports exact k-NN via `MATCH ? AND k = ?` syntax, and is maintained by the same team behind sqlite-vss. For Lore's corpus sizes (thousands to tens of thousands of entries), exact k-NN is fast enough and eliminates ANN approximation errors. The `vec0` virtual table stores 384-d float32 vectors as 1536-byte BLOBs.

The version is pinned exactly because sqlite-vec is pre-1.0 and has made breaking API changes between minor versions. An unpinned dependency would silently break `sqlite_vec.load(conn)` or `sqlite_vec.serialize_float32()` calls on upgrade. The pin is intentional and documented in `pyproject.toml`.

**Trade-offs accepted:**
- sqlite-vec is pre-v1.0: the API may change; pin updates require testing.
- No HNSW approximation in SQLite — exact k-NN is O(n) at query time. At 100k entries this becomes measurable (~100ms); at 1M entries it is unacceptable.
- `conn.enable_load_extension(True)` must be called before loading; some Python SQLite builds have this disabled. CI uses the standard CPython build where it is available.

**Revisit if:** KB corpus grows beyond ~50k entries (consider PostgreSQL + pgvector HNSW), or sqlite-vec reaches v1.0 with a stable API.

---

## Decision: halfvec(384) on PostgreSQL

**Choice:** `halfvec(384)` when pgvector ≥ 0.7, fallback to `vector(384)`

**Alternatives considered:**
- `vector(384)` always — simpler, widely supported
- `bit(384)` binary quantization — smallest, lowest quality
- `vector(768)` or larger — if a higher-dimensional model were chosen

**Rationale:**
`halfvec` (16-bit float) uses exactly half the storage of `vector` (32-bit float): 384 × 2 = 768 bytes per row vs 384 × 4 = 1536 bytes. At 100k entries, that is 73 MB vs 146 MB just for embeddings. For `all-MiniLM-L6-v2`, the retrieval quality loss from float32 → float16 quantization is negligible (measured at <0.1% NDCG drop on MTEB). `halfvec` was introduced in pgvector 0.7; `_init_schema()` detects the installed version and falls back to `vector` automatically.

**Trade-offs accepted:**
- `halfvec` is only available in pgvector ≥ 0.7; older installs use `vector`.
- If someone creates the table with `vector` and later upgrades pgvector, the column type does not auto-migrate. A manual migration would be needed to reclaim storage.

**Revisit if:** A model with more than 384 dimensions is adopted (storage savings from halfvec become even more significant at 768d+).

---

## Decision: HNSW index (not IVFFlat)

**Choice:** HNSW with `m=16, ef_construction=64`

**Alternatives considered:**
- IVFFlat — traditional inverted-file with flat quantization
- No index (exact scan) — trivial, only suitable for very small corpora

**Rationale:**
HNSW (Hierarchical Navigable Small World) is better suited to Lore's usage pattern: a single process performing many individual k-NN queries rather than bulk-loading and rebuilding. IVFFlat requires a training step (`CREATE INDEX` needs `>= 3 * lists` rows to be accurate), which means a fresh database has poor recall until enough entries exist. HNSW builds incrementally — every insert immediately improves the graph. `m=16` (max connections per layer) and `ef_construction=64` are pgvector defaults and provide a good accuracy/build-speed tradeoff for corpora up to ~1M rows.

**Trade-offs accepted:**
- HNSW build time is slower than IVFFlat for very large datasets.
- Higher memory use during index build (for ef_construction).
- No HNSW available in SQLite's sqlite-vec; the SQLite path uses exact k-NN.

**Revisit if:** Corpus reaches millions of rows and build time becomes a bottleneck.

---

## Decision: Reciprocal Rank Fusion (not weighted linear combination)

**Choice:** RRF (`score = Σ 1/(k + rank)`) to fuse FTS5 and vector rankings

**Alternatives considered:**
- Weighted linear combination: `α × bm25_score + (1-α) × cosine_score`
- Score normalization + sum: normalize both to [0,1] then add
- Cascade: use vector search only when FTS returns fewer than N results

**Rationale:**
BM25 scores and cosine similarity scores are not comparable. BM25 is unbounded and corpus-dependent; cosine similarity is in [-1, 1]. Any linear combination requires empirically tuning `α` per corpus, which is impractical for a general-purpose server. RRF operates on rank positions, not raw scores — it is invariant to score scale and distribution. The formula `1/(k + rank)` is well-studied (Cormack et al. 2009) and has been shown to match or outperform tuned linear combinations on heterogeneous corpora without any hyperparameter fitting.

**Trade-offs accepted:**
- RRF discards score magnitude; a BM25 score of 0.001 and 100.0 at the same rank are treated identically.
- Requires both lists to be present; if only one signal is available, pure-mode is more efficient.
- No learning; cannot be trained on user feedback (see Issue #5 for telemetry/hard-negative mining, a deferred feature).

**Revisit if:** User feedback telemetry is collected and a learned ranker becomes feasible (Issue #5). RRF can serve as the training signal generator.

---

## Decision: RRF k=10 default

**Choice:** `LORE_RRF_K=10` default, configurable

**Alternatives considered:**
- k=60 — the value used in the original Cormack et al. paper
- k=1 — maximum position-sensitivity
- Dynamic k based on corpus size

**Rationale:**
The smoothing constant `k` controls how steeply rank position is discounted. At k=10, the score difference between rank 1 and rank 2 is `1/11 - 1/12 ≈ 0.0076` — meaningful. At k=60, that gap shrinks to `0.00021`. For small KBs (dozens to hundreds of entries), k=10 provides appropriate sensitivity; a document at rank 3 still outscores one at rank 8. The original paper used k=60 for large IR benchmarks; for Lore's typical corpora (hundreds to low thousands), k=10 is more discriminating. The README documents the guidance: increase to 30–60 for corpora >10k entries.

**Trade-offs accepted:**
- No single k is optimal for all corpus sizes; users with large KBs should tune.
- The default of 10 was not empirically validated on Lore's specific data; it follows the mcp-memory-service convention.

**Revisit if:** Usage data shows consistently poor result ordering for large corpora.

---

## Decision: Separate `knowledge_kb_embeddings` table (not a column in `knowledge_kb_entries`)

**Choice:** Two auxiliary tables: `knowledge_kb_vec_embeddings` (vec0 virtual table) and `knowledge_kb_embedding_meta` (plain table for staleness tracking)

**Alternatives considered:**
- Add a `BLOB` column to `knowledge_kb_entries` — store the serialized vector inline
- Single `kb_embeddings` table with both the vector and metadata

**Rationale:**
vec0 virtual tables (sqlite-vec) and pgvector columns are separate storage mechanisms that cannot be combined with standard table columns in SQLite. The vec0 virtual table must be its own `CREATE VIRTUAL TABLE` statement. Metadata (model name, content hash, timestamps) cannot be stored in vec0 — hence the separate `knowledge_kb_embedding_meta` table. This separation also means the schema degrades cleanly: if vec0 is unavailable, `knowledge_kb_entries` is unaffected and the server boots normally.

On PostgreSQL, the design mirrors this: `knowledge.kb_embeddings` is a separate table with a FK to `knowledge.kb_entries(kb_id) ON DELETE CASCADE`, keeping the core entries table schema-stable across semantic feature rollout.

**Trade-offs accepted:**
- Two extra tables to manage; deletes require explicit cleanup of both (FKs + explicit delete in `_delete_kb_embedding`).
- vec0 does not support FK constraints, so the application must enforce referential integrity.

**Revisit if:** A future SQLite embedding library supports combined table/vector storage.

---

## Decision: content_hash for staleness detection

**Choice:** SHA-256 of `(title + \x1f + content)` stored in `knowledge_kb_embedding_meta`

**Alternatives considered:**
- `updated_at` timestamp comparison — simpler but unreliable (timestamps can be equal for rapid updates)
- Store embedded text directly — accurate but doubles storage
- Model version only — catches model changes but not content changes

**Rationale:**
The hash detects two independent causes of staleness: (1) the content changed (title or body edited), (2) the model changed. Both are necessary for correct cache invalidation. SHA-256 is deterministic, fast, and 64 hex characters is negligible storage. The unit separator `\x1f` between title and content prevents hash collisions where concatenation alone would be ambiguous (e.g., `title="ab" + content="c"` vs `title="a" + content="bc"`).

`kb_backfill_embeddings` recomputes the hash for every row and skips rows where `meta_hash == compute_content_hash(title, content) AND meta_model == current_model`.

**Trade-offs accepted:**
- Hash comparison on every backfill run is O(n) — cheap for typical KB sizes.
- Tags, author, and other metadata are excluded from the hash. Changing tags alone does not trigger re-embedding (correct behavior; those fields are not part of the embedded text).

---

## Decision: Feature flag default-off (`LORE_SEMANTIC_SEARCH=false`)

**Choice:** Semantic search is disabled by default; existing deployments are unaffected

**Alternatives considered:**
- Default-on — opt out to disable
- Progressive rollout — enable for new installs only

**Rationale:**
The `[semantic]` extra adds ~500 MB of dependencies (sentence-transformers, onnxruntime, sqlite-vec). Making semantic search opt-in ensures that users upgrading from v0.5.x do not suddenly download 500 MB on restart, do not face a model-load delay on first query, and are not broken by a missing sqlite-vec extension build. Zero change to default behavior is the primary design constraint for the v0.6.0 release.

**Trade-offs accepted:**
- Users must explicitly enable the feature; it will not be discovered automatically.
- The feature flag check is scattered across multiple modules (`embeddings.py`, `search.py`, `db_client.py`, `server.py`) — each module is responsible for checking the flag for its own path.

**Revisit if:** The `[semantic]` extra is bundled into the base install (if dependencies shrink significantly), or if usage data shows most users want it on by default.

---

## Decision: Local embeddings (no cloud API)

**Choice:** `sentence-transformers` running locally via ONNX

**Alternatives considered:**
- OpenAI `text-embedding-3-small` (1536d) — high quality, paid, external
- Cohere Embed v3 — multilingual, paid
- Jina AI Embeddings — free tier available
- Self-hosted `text-embeddings-inference` sidecar (HuggingFace TEI)

**Rationale:**
Lore's core value proposition is self-hosted, air-gapped operation. Users run it for operational knowledge that may include credentials, infrastructure topology, or sensitive design decisions — none of which should leave the host. A cloud embedding API would add an API key requirement, a network dependency, a cost, and a privacy concern. Local ONNX inference has no latency beyond the first model load (cold start ~2s), and typical `all-MiniLM-L6-v2` inference is 5–20ms per batch on CPU.

**Trade-offs accepted:**
- Cold start on first request after server start (model load ~2s).
- CPU-only inference; no GPU acceleration in the default path.
- Model quality ceiling is lower than large cloud models.

**Revisit if:** A user use case requires embedding quality far above what local models provide, and they are willing to accept the privacy and cost trade-offs.

---

## Decision: FTS5 upgrade from SQLite LIKE

**Choice:** FTS5 with Porter stemmer and `bm25()` ranking, replacing `LIKE`-based lexical search

**Alternatives considered:**
- Keep SQLite `LIKE` — the existing behavior before v0.6.0
- FTS3/FTS4 — older SQLite FTS modules
- Full-text search via external engine (MeiliSearch, Typesense)

**Rationale:**
SQLite `LIKE '%query%'` scans every row, does not stem, cannot rank by relevance, and breaks on multi-word queries. FTS5 with `tokenize='porter unicode61'` handles stemming (search → searching → searched), ranks by BM25, and uses an inverted index for O(log n) lookup. The upgrade is transparent to existing users: if FTS5 is unavailable (old SQLite build), the code falls back to `_CORE_SQLITE_STATEMENTS` (core tables only, no FTS) and the legacy LIKE path continues to work.

**Trade-offs accepted:**
- FTS5 triggers add latency to every INSERT/UPDATE/DELETE (~1ms per row at typical KB sizes — negligible).
- FTS5 requires a separate `CREATE VIRTUAL TABLE` and three triggers; schema is more complex.
- FTS5 is not available in all SQLite builds (WebAssembly, some embedded builds). The degradation path handles this.

**Revisit if:** Someone needs a richer query language (phrase proximity, field boosts). FTS5 supports these natively.

---

## Decision: Fire-and-forget embedding at write time, advisory lock at backfill

**Choice:** `kb_add` embeds best-effort (no lock, failure does not block); `kb_backfill_embeddings` acquires a module-level `threading.Lock` before starting

**Alternatives considered:**
- Lock on every write — prevents double-embedding but adds latency to every `kb_add`
- Queue + background worker — decouples embedding from write path entirely
- No lock on backfill — allow concurrent backfill runs

**Rationale:**
Individual writes are rare in practice (single agent adding an entry); locking would add latency to the most common write operation without benefit. The content_hash guard inside `_embed_kb_entry` makes repeated embeds of the same content idempotent — so even if two agents call `kb_add` with the same entry simultaneously, the second embed is a no-op (hash match). Backfill is the dangerous operation: it iterates every row. Two concurrent backfills would race, double-encode, and waste time. The `_BACKFILL_LOCK` prevents this; a concurrent call returns an error immediately (`"Another backfill is already running"`).

**Trade-offs accepted:**
- The lock is in-process only (`threading.Lock`). Multi-process deployments (e.g., multiple `lore-mcp` workers behind a load balancer) are not protected. The hash guard still makes concurrent backfills correct — just wasteful.
- Fire-and-forget means a fresh install with `kb_add` during model download will record `embedded=false` for early entries. `kb_backfill_embeddings` is the recovery path.

**Revisit if:** Lore is deployed multi-process and backfill performance becomes a concern. A database-level advisory lock (PostgreSQL `pg_try_advisory_lock`) is the correct solution for that case.

---

## Decision: Singleton + threading.Lock (no FastMCP lifespan)

**Choice:** Module-level `_embedder` singleton protected by `threading.Lock` in `embeddings.py`

**Alternatives considered:**
- FastMCP `@server.on_startup` lifespan hook — load model at server start
- Per-request model load — correct but 2s latency on every call
- `functools.lru_cache` — simpler but not thread-safe on first load

**Rationale:**
FastMCP's lifespan API (Issue #7 — deferred) would be the clean solution. At the time of v0.6.0, Lore uses raw `mcp` library without FastMCP; there is no `on_startup` hook. The singleton + double-checked locking pattern is the standard thread-safe lazy-initialization idiom in Python. The `_embed_lock` ensures only one thread loads the model while other threads wait; once loaded, the lock is not needed for reads (`_embedder` is never mutated after load). `reset_for_tests()` clears the singleton under the lock for test isolation.

**Trade-offs accepted:**
- The first request after server start incurs a ~2s model load delay.
- The singleton is process-global; model name changes (via `LORE_EMBEDDING_MODEL`) are only picked up on first load within a process. Restart required to change model.
- Not compatible with fork-based multiprocessing without reinitialization.

**Revisit if:** FastMCP migration (Issue #7) proceeds and a clean `on_startup` hook becomes available.
