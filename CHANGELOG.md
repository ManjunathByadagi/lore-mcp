# Changelog

All notable changes to this project are documented here.

## [0.6.0] - 2026-05-24

> **Note:** This release was yanked from PyPI on 2026-05-24. Existing installs continue to work; new installs will skip this version. A successor release will follow once the staging and end-to-end testing pipeline is established. The feature set remains intact and accurate.

### Added
- **Semantic & Hybrid Search** (Issue #6) — Lore now finds entries by meaning, not just keywords
  - Local sentence-transformers embeddings via ONNX (no API key, no external calls)
  - FTS5 (BM25) lexical search replaces SQLite LIKE fallback
  - sqlite-vec cosine similarity for vector search
  - Reciprocal Rank Fusion (RRF) hybrid mode combining lexical + semantic
  - New optional `[semantic]` extra: `pip install lore-knowledge-mcp[semantic]`
  - Opt-in via `LORE_SEMANTIC_SEARCH=true` (default: off, zero impact on existing users)
  - Configurable via `LORE_EMBEDDING_MODEL`, `LORE_RRF_K`, `LORE_DEBUG_SEARCH`
  - Multilingual support via `paraphrase-multilingual-MiniLM-L12-v2` (same 384d)
- New MCP tool: `kb_backfill_embeddings` — generate embeddings for existing KB entries (idempotent)
- New MCP tool: `kb_embedding_status` — report embedding coverage and model info
- New module `src/lore/embeddings.py` — singleton model loader with content_hash for stale detection
- New module `src/lore/search.py` — RRF, candidate pool sizing, hybrid orchestration
- FTS5 virtual table + triggers for SQLite (replaces unranked LIKE search)

### Changed
- SQLite schema now applied via `executescript()` instead of split-on-semicolon (required for trigger blocks)
- `handle_kb_search` accepts new optional params: `semantic`, `hybrid`, `search_mode`, `top_k`
- Search response includes new fields: `search_mode`, `model`, `rrf_k`, `score` (when applicable)
- `kb_add` embeds at write time (best-effort — KB entry still succeeds if embed fails)
- `kb_update` re-embeds only when content changes (content_hash comparison)
- `kb_delete` correctly orders deletes: `knowledge_kb_entries` first, then vec0 row

### Fixed
- RRF tie-breaking is now deterministic (secondary sort by `kb_id`)
- FTS5 fast path no longer requires semantic flag to activate
- `[semantic]` extra now includes `optimum[onnxruntime]` for clean fresh installs

### Deferred to Phase 2 (Issue #6 follow-up)
- PostgreSQL semantic path (pgvector HNSW with `halfvec(384)`) — schema migration file present, Python integration pending
- `kb_reindex_embeddings` for full re-embedding on model change
- Cross-encoder reranker
- `include_content` param on `kb_search`

## [0.5.0] — 2026-05-23

### Changed
- Renamed Python package from `knowledge_mcp` to `lore`
- Renamed systemd service from `knowledge-mcp` to `lore`
- Renamed CLI entry point from `knowledge-mcp` to `lore-mcp`
- Updated project metadata URLs to `lore-mcp` identity
- `lore-mcp` now accepts `--host`/`--port` to run as HTTP/SSE server (stdio remains default)
- Default backend changed to `sqlite` so a clean install boots without any external services

### Fixed
- DB password removed from `main()` (was hardcoded for dev convenience)
- DB password removed from systemd unit (was duplicated from `.env`)
- Old `knowledge-mcp.service` masked to prevent accidental double-start
- `__version__` bumped from 0.1.0 to 0.5.0 to match `pyproject.toml`

## [0.4.0] — 2026-05-23

### Added
- Attribution model: `author`, `source_type`, `verified` fields on all KB entries
- `kb_ingest_doc`/`kb_ingest_dir` now accept `author` and `source_type` parameters
- Query sanitization for `kb_search` to prevent PostgREST filter injection
- Ruff linter + formatter (`make lint`, `make format`, `make fix`)

### Changed
- Rebranded from "Advanced Knowledge MCP" to **Lore**
- Renamed `research_*` tools to `investigation_*` (better reflects ops use case)
- Tool surface: 38 → 29 tools (streamlined)

### Removed
- Knowledge Graph tools (`kg_*`) — unused in practice
- Source tracking tools (`research_add_source`, etc.) — unused in practice
- `kb_link_to_source` tool

## [0.3.0] — 2026-04-07

### Added
- SQLite backend support (no database server required)
- MCP Index: scan and search across all configured MCP servers
- Multi-search: query KB, investigations, journal, and transcripts simultaneously

## [0.2.0] — 2025-12-08

### Added
- Research workflows (notes, experiments, source linking)
- Document ingestion with change detection (SHA-256 hashing)
- Supabase backend support

## [0.1.0] — 2025-12-01

### Added
- Initial release: Knowledge Base with semantic search
- Journal system for decision logging
- PostgreSQL backend
