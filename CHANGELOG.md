# Changelog

All notable changes to this project are documented here.

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
