#!/usr/bin/env python3
"""
Lore MCP Server - the operational knowledge layer for engineers and AI agents.

Unified knowledge management system for:
- Knowledge Base (structured operational knowledge with attribution)
- Investigations (ops debugging notes and structured experiments)
- Journal (decision log and config snapshots)
- MCP Index (tool discovery and search)
- Search (local files, corpora, transcripts, multi-source)

Spec: Lore v0.4.0 — KG and source-tracking tool surfaces removed; research surface
renamed to investigations; attribution model added (author, source_type, verified).
"""

import glob as glob_module
import hashlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import jsonschema
import sentry_sdk
import yaml
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .db_client import DatabaseBackend, get_db_client

# Import document processor and MCP scanner
from .doc_processor import DocumentProcessor
from .env_config import get_env, require_env
from .mcp_index_scanner import MCPIndexScanner
from .response import ErrorCodes, ResponseEnvelope

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Sentry
SENTRY_DSN = get_env("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        environment=get_env("SENTRY_ENVIRONMENT", "development"),
        release=get_env("SENTRY_RELEASE", "latvian-lab@1.0.0"),
    )
    logger.info("Sentry monitoring enabled")

# Initialize MCP server
app = Server("knowledge-mcp")

# Database Configuration (will be initialized in main())
# Database Configuration
try:
    db = get_db_client()
    backend = os.getenv("DB_BACKEND", "local")
    logger.info(f"Connected to database backend: {backend}")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    db = None


# Search Configuration (consolidated from search-mcp)
LATVIAN_LEARNING_ROOT = Path(get_env("LATVIAN_LEARNING_ROOT", "/srv/latvian_learning"))
LATVIAN_XTTS_ROOT = Path(get_env("LATVIAN_XTTS_ROOT", "/srv/latvian_xtts"))
INGEST_ROOT = Path(get_env("INGEST_ROOT", "/srv/ingest"))
KNOWLEDGE_DATA_DIR = Path(get_env("KNOWLEDGE_DATA_DIR", "/srv/latvian_mcp/data/knowledge"))


def json_serializer(obj):
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def format_response(response: dict) -> list[types.TextContent]:
    """Format response as MCP TextContent."""
    return [types.TextContent(type="text", text=json.dumps(response, default=json_serializer))]


def _sanitize_search_query(query: str) -> str:
    """Strip PostgREST filter metacharacters to prevent filter injection."""
    # Remove commas, dots, parentheses, and PostgREST operator tokens
    sanitized = re.sub(r"[,.()\[\]]", " ", query)
    # Strip known PostgREST operator patterns
    sanitized = re.sub(
        r"\b(wfts|plfts|fts|phfts|ilike|like|eq|neq|gt|gte|lt|lte|in|is)\b",
        "",
        sanitized,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace
    return " ".join(sanitized.split()).strip()


def _coerce_arguments(arguments: dict, schema: dict) -> dict:
    """Coerce JSON-stringified array/object values into proper Python types.

    Claude Code sometimes serializes array or object parameters as JSON strings
    instead of proper arrays/objects. Three encoding formats are handled:

    1. JSON-encoded:  tags='["a","b","c"]'  → ["a","b","c"]
    2. Comma-separated: tags='a,b,c'        → ["a","b","c"]
    3. Space-separated: tags='a b c'        → ["a","b","c"]

    Formats 2 and 3 only apply to fields whose schema items type is "string".
    JSON encoding is always tried first.

    Args:
        arguments: Raw arguments dict from the MCP call.
        schema: The tool's inputSchema dict.

    Returns:
        A new dict with coerced values; original is not mutated.
    """
    if not arguments or not schema:
        return arguments

    properties = schema.get("properties", {})
    if not properties:
        return arguments

    coerced = dict(arguments)
    for field, field_schema in properties.items():
        if field not in coerced:
            continue
        value = coerced[field]
        expected_type = field_schema.get("type")
        if expected_type == "array" and isinstance(value, str):
            # Try JSON parse first (handles '["a","b"]' format)
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    coerced[field] = parsed
                    continue
            except (json.JSONDecodeError, ValueError):
                pass
            # Fall back to splitting plain-text strings (space or comma separated).
            # Only safe for string-item arrays; skip for numeric/object item arrays.
            items_schema = field_schema.get("items", {})
            if items_schema.get("type", "string") == "string":
                stripped = value.strip()
                if stripped:
                    # Prefer comma-split if commas present, else space-split
                    if "," in stripped:
                        parts = [p.strip() for p in stripped.split(",") if p.strip()]
                    else:
                        parts = stripped.split()
                    if parts:
                        coerced[field] = parts
        elif expected_type == "object" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    coerced[field] = parsed
            except (json.JSONDecodeError, ValueError):
                pass  # Leave the value as-is; jsonschema will report the error
        elif expected_type == "boolean" and isinstance(value, str):
            lower = value.strip().lower()
            if lower in ("true", "1", "yes"):
                coerced[field] = True
            elif lower in ("false", "0", "no"):
                coerced[field] = False
            # else: leave as-is; jsonschema validation will catch invalid values
    return coerced


# =============================================================================
# Tool Registration
# =============================================================================

# Module-level tool definitions used by both list_tools() and _get_tool_schema().
# Keeping them here avoids duplicating schemas and enables synchronous schema
# lookups inside call_tool() for the array-coercion fix.
_TOOL_DEFINITIONS = [
    # Knowledge Base Tools (6)
    types.Tool(
        name="kb_add",
        description="Add a knowledge base entry",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic"},
                "title": {"type": "string", "description": "Entry title"},
                "content": {"type": "string", "description": "Entry content"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
                "author": {
                    "type": "string",
                    "description": "Who is creating this entry (your name, agent name, or system). Optional.",
                },
                "source_type": {
                    "type": "string",
                    "description": "Origin: 'human', 'agent', or 'system'. Optional, defaults to null.",
                },
            },
            "required": ["topic", "title", "content"],
        },
    ),
    types.Tool(
        name="kb_search",
        description=(
            "Search knowledge base. Lexical FTS5 (or LIKE fallback) by default; "
            "set semantic=true / hybrid=true / search_mode=hybrid to use vector "
            "embeddings + RRF fusion (requires LORE_SEMANTIC_SEARCH=true)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "topic": {"type": "string", "description": "Filter by topic"},
                "top_k": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Number of results to return (default 20).",
                },
                "semantic": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force semantic (vector) search only. Shortcut for search_mode='semantic'.",
                },
                "hybrid": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force hybrid (FTS5 + vector + RRF). Shortcut for search_mode='hybrid'.",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["fts", "semantic", "hybrid"],
                    "description": (
                        "Explicit search mode. Overrides semantic/hybrid flags. "
                        "Falls back to FTS when semantic is unavailable."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="kb_get",
        description="Get full KB entry by ID",
        inputSchema={
            "type": "object",
            "properties": {"kb_id": {"type": "string", "description": "KB entry ID"}},
            "required": ["kb_id"],
        },
    ),
    types.Tool(
        name="kb_list",
        description="List KB entries",
        inputSchema={
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Filter by topic"}},
        },
    ),
    types.Tool(
        name="kb_update",
        description="Update existing KB entry content and metadata",
        inputSchema={
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "UUID of entry to update"},
                "content": {"type": "string", "description": "New content text"},
                "metadata": {"type": "object", "description": "Updated metadata object"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Updated tags array",
                },
                "topic": {"type": "string", "description": "Updated topic/category for the entry"},
                "verified": {
                    "type": ["boolean", "null"],
                    "description": "Mark entry as human-verified (true), disputed (false), or reset to unreviewed (null).",
                },
            },
            "required": ["entry_id"],
        },
    ),
    types.Tool(
        name="kb_delete",
        description="Delete existing KB entry from database",
        inputSchema={
            "type": "object",
            "properties": {
                "entry_id": {"type": "string", "description": "UUID of entry to delete"},
                "confirm": {
                    "type": "boolean",
                    "description": "Confirmation flag for safety",
                    "default": False,
                },
            },
            "required": ["entry_id"],
        },
    ),
    # Investigations Tools (5)
    types.Tool(
        name="investigation_add",
        description="Add an investigation entry (open or append to an ops investigation)",
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["topic", "title", "content"],
        },
    ),
    types.Tool(
        name="investigation_list",
        description="List investigations",
        inputSchema={
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "Filter by topic"}},
        },
    ),
    types.Tool(
        name="investigation_get",
        description="Get a single investigation entry by ID",
        inputSchema={
            "type": "object",
            "properties": {"note_id": {"type": "string"}},
            "required": ["note_id"],
        },
    ),
    types.Tool(
        name="investigation_log_experiment",
        description="Log a structured experiment within an investigation (hypothesis, methodology, results, conclusion)",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "hypothesis": {"type": "string"},
                "methodology": {"type": "string"},
                "results": {"type": "object"},
                "conclusion": {"type": "string"},
            },
            "required": ["title"],
        },
    ),
    types.Tool(
        name="investigation_list_experiments",
        description="List logged investigation experiments",
        inputSchema={"type": "object", "properties": {}},
    ),
    # Journal Tools (4)
    types.Tool(
        name="journal_append",
        description="Append journal entry",
        inputSchema={
            "type": "object",
            "properties": {
                "entry_type": {
                    "type": "string",
                    "enum": ["daily", "milestone", "reflection", "idea"],
                },
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["entry_type", "content"],
        },
    ),
    types.Tool(
        name="journal_list",
        description="List journal entries",
        inputSchema={"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}},
    ),
    types.Tool(
        name="journal_get",
        description="Get journal entry",
        inputSchema={
            "type": "object",
            "properties": {"entry_id": {"type": "string"}},
            "required": ["entry_id"],
        },
    ),
    types.Tool(
        name="snapshot_config",
        description="Snapshot current config",
        inputSchema={
            "type": "object",
            "properties": {"config_name": {"type": "string"}, "config_data": {"type": "object"}},
            "required": ["config_name", "config_data"],
        },
    ),
    # Document Ingestion Tools (4) - v1.3
    types.Tool(
        name="kb_ingest_doc",
        description="Ingest single markdown file into KB with change detection",
        inputSchema={
            "type": "object",
            "properties": {
                "doc_path": {"type": "string", "description": "Absolute path to markdown file"},
                "strategy": {
                    "type": "string",
                    "enum": ["full", "chunked", "summary"],
                    "default": "chunked",
                    "description": "Ingestion strategy: full (one entry), chunked (by sections), summary (GPT summary)",
                },
                "chunk_size": {
                    "type": "integer",
                    "default": 2000,
                    "description": "Max tokens per chunk (chunked strategy only)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional tags",
                },
                "overwrite": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace existing KB entries from this doc",
                },
                "author": {
                    "type": "string",
                    "description": "Who is ingesting (optional, defaults to None)",
                },
                "source_type": {
                    "type": "string",
                    "default": "system",
                    "description": "Source type for attribution (defaults to 'system' since ingestion is automated)",
                },
            },
            "required": ["doc_path"],
        },
    ),
    types.Tool(
        name="kb_ingest_dir",
        description="Batch ingest directory of markdown files",
        inputSchema={
            "type": "object",
            "properties": {
                "dir_path": {"type": "string", "description": "Directory to scan"},
                "pattern": {
                    "type": "string",
                    "default": "*.md",
                    "description": "File pattern (e.g., *.md)",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["full", "chunked", "summary"],
                    "default": "chunked",
                },
                "recursive": {
                    "type": "boolean",
                    "default": True,
                    "description": "Scan subdirectories",
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Patterns to exclude",
                },
                "author": {
                    "type": "string",
                    "description": "Who is ingesting (optional, defaults to None)",
                },
                "source_type": {
                    "type": "string",
                    "default": "system",
                    "description": "Source type for attribution (defaults to 'system' since ingestion is automated)",
                },
            },
            "required": ["dir_path"],
        },
    ),
    types.Tool(
        name="kb_sync_status",
        description="Check sync state between source docs and KB",
        inputSchema={
            "type": "object",
            "properties": {"dir_path": {"type": "string", "description": "Directory to check"}},
            "required": ["dir_path"],
        },
    ),
    # Semantic Search Tools (2) - v0.6
    types.Tool(
        name="kb_backfill_embeddings",
        description=(
            "Embed any KB entries that are missing or stale (model/content "
            "changed). Idempotent: skips entries whose stored content_hash "
            "still matches. Requires LORE_SEMANTIC_SEARCH=true."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "batch_size": {
                    "type": "integer",
                    "default": 32,
                    "minimum": 1,
                    "maximum": 512,
                    "description": "How many entries to encode per batch (default 32).",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional cap on entries to process this run.",
                },
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, report what would be embedded without writing.",
                },
            },
        },
    ),
    types.Tool(
        name="kb_embedding_status",
        description=(
            "Report embedding coverage: total entries, embedded count, missing "
            "count, current model, and per-model breakdown."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    # MCP Index Tools (5)
    types.Tool(
        name="mcp_index_scan",
        description="Scan all MCP servers and index their tools. By default, scans only configured servers (66% token savings).",
        inputSchema={
            "type": "object",
            "properties": {
                "triggered_by": {
                    "type": "string",
                    "default": "manual",
                    "description": "Source of scan (manual, cron, deployment)",
                },
                "config_filter": {
                    "type": "boolean",
                    "default": True,
                    "description": "If true (default), scan only servers in ~/.claude.json. Set false to scan all servers.",
                },
            },
        },
    ),
    types.Tool(
        name="mcp_index_search",
        description="Search for MCP tools by description/capability",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "category": {
                    "type": "string",
                    "description": "Optional category filter (search, storage, processing, etc.)",
                },
                "limit": {"type": "integer", "default": 20, "description": "Maximum results"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="mcp_index_get_server",
        description="Get all tools for a specific MCP server",
        inputSchema={
            "type": "object",
            "properties": {
                "server_id": {"type": "string", "description": "Server ID (e.g., 'knowledge-mcp')"}
            },
            "required": ["server_id"],
        },
    ),
    types.Tool(
        name="mcp_index_get_tool",
        description="Get detailed information about a specific tool",
        inputSchema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Tool name (e.g., 'kb_search')"}
            },
            "required": ["tool_name"],
        },
    ),
    types.Tool(
        name="mcp_index_rebuild",
        description="Force rebuild of entire MCP index (same as mcp_index_scan)",
        inputSchema={"type": "object", "properties": {}},
    ),
    # ================================================================
    # SEARCH TOOLS (consolidated from search-mcp)
    # ================================================================
    types.Tool(
        name="search_local",
        description="Search local files by content (lexical mode)",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Paths to search (defaults: learning, xtts, knowledge)",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File extensions to search (default: txt, json, md, py, yaml)",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="search_corpora",
        description="Search across corpus manifests (JSONL files)",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "corpus_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific corpus IDs to search (optional)",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="search_transcripts",
        description="Search transcript segments from Whisper outputs",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "speaker": {"type": "string", "description": "Filter by speaker (optional)"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="multi_search",
        description="Combined search across all sources (local, knowledge, corpora, transcripts)",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    types.Tool(
        name="deduplicate_results",
        description="Remove duplicate search results based on text similarity",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Array of search result objects",
                },
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (0-1, default: 0.9)",
                },
            },
            "required": ["results"],
        },
    ),
    types.Tool(
        name="cluster_results",
        description="Cluster search results by topic/source type",
        inputSchema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Array of search result objects",
                },
                "num_clusters": {
                    "type": "integer",
                    "description": "Number of clusters (default: 5)",
                },
            },
            "required": ["results"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all available tools."""
    return _TOOL_DEFINITIONS


# Pre-built schema lookup table for O(1) access in call_tool().
_TOOL_SCHEMA_MAP: dict[str, dict] = {t.name: t.inputSchema for t in _TOOL_DEFINITIONS}


def _get_tool_schema(name: str) -> dict:
    """Return the inputSchema for a named tool, or empty dict if not found."""
    return _TOOL_SCHEMA_MAP.get(name, {})


@app.call_tool(validate_input=False)
async def call_tool(name: str, arguments: Any) -> list[types.TextContent]:
    """Route tool calls to handlers.

    validate_input is disabled at the framework level so we can coerce
    JSON-stringified array/object values (a Claude Code serialization quirk)
    before running jsonschema validation ourselves.
    """
    try:
        # Coerce JSON-stringified arrays/objects, then validate
        arguments = _coerce_arguments(arguments or {}, _get_tool_schema(name))
        schema = _get_tool_schema(name)
        if schema:
            try:
                jsonschema.validate(instance=arguments, schema=schema)
            except jsonschema.ValidationError as exc:
                return format_response(
                    ResponseEnvelope.error(
                        ErrorCodes.INVALID_INPUT, f"Input validation error: {exc.message}"
                    )
                )

        # KB Tools
        if name == "kb_add":
            return format_response(handle_kb_add(**arguments))
        elif name == "kb_search":
            return format_response(handle_kb_search(**arguments))
        elif name == "kb_get":
            return format_response(handle_kb_get(**arguments))
        elif name == "kb_list":
            return format_response(handle_kb_list(**arguments))
        elif name == "kb_update":
            return format_response(handle_kb_update(**arguments))
        elif name == "kb_delete":
            return format_response(handle_kb_delete(**arguments))

        # Investigations Tools
        elif name == "investigation_add":
            return format_response(handle_investigation_add(**arguments))
        elif name == "investigation_list":
            return format_response(handle_investigation_list(**arguments))
        elif name == "investigation_get":
            return format_response(handle_investigation_get(**arguments))
        elif name == "investigation_log_experiment":
            return format_response(handle_investigation_log_experiment(**arguments))
        elif name == "investigation_list_experiments":
            return format_response(handle_investigation_list_experiments(**arguments))

        # Journal Tools
        elif name == "journal_append":
            return format_response(handle_journal_append(**arguments))
        elif name == "journal_list":
            return format_response(handle_journal_list(**arguments))
        elif name == "journal_get":
            return format_response(handle_journal_get(**arguments))
        elif name == "snapshot_config":
            return format_response(handle_snapshot_config(**arguments))

        # Document Ingestion Tools (v1.3)
        elif name == "kb_ingest_doc":
            return format_response(handle_kb_ingest_doc(**arguments))
        elif name == "kb_ingest_dir":
            return format_response(handle_kb_ingest_dir(**arguments))
        elif name == "kb_sync_status":
            return format_response(handle_kb_sync_status(**arguments))

        # Semantic Search Tools (v0.6)
        elif name == "kb_backfill_embeddings":
            return format_response(handle_kb_backfill_embeddings(**arguments))
        elif name == "kb_embedding_status":
            return format_response(handle_kb_embedding_status(**arguments))

        # MCP Index Tools
        elif name == "mcp_index_scan":
            return format_response(handle_mcp_index_scan(**arguments))
        elif name == "mcp_index_search":
            return format_response(handle_mcp_index_search(**arguments))
        elif name == "mcp_index_get_server":
            return format_response(handle_mcp_index_get_server(**arguments))
        elif name == "mcp_index_get_tool":
            return format_response(handle_mcp_index_get_tool(**arguments))
        elif name == "mcp_index_rebuild":
            return format_response(handle_mcp_index_rebuild(**arguments))

        # Search Tools (consolidated from search-mcp)
        elif name == "search_local":
            return format_response(handle_search_local(**arguments))
        elif name == "search_corpora":
            return format_response(handle_search_corpora(**arguments))
        elif name == "search_transcripts":
            return format_response(handle_search_transcripts(**arguments))
        elif name == "multi_search":
            return format_response(handle_multi_search(**arguments))
        elif name == "deduplicate_results":
            return format_response(handle_deduplicate_results(**arguments))
        elif name == "cluster_results":
            return format_response(handle_cluster_results(**arguments))

        else:
            return format_response(
                ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Unknown tool: {name}")
            )
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return format_response(ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e)))


# =============================================================================
# Knowledge Base Handlers
# =============================================================================


def _semantic_write_enabled() -> bool:
    """Whether to embed on the KB write path. SQLite + vec0 only for MVP."""
    backend = os.getenv("DB_BACKEND", "").strip().lower()
    if backend != "sqlite":
        # PostgreSQL semantic write path: Phase 2.
        return False
    if not getattr(db, "vec_extension_loaded", False):
        return False
    flag = os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower()
    return flag == "true"


def _embed_kb_entry(
    kb_id: str,
    title: str,
    content: str,
    *,
    expected_hash: str | None = None,
) -> tuple[bool, str | None]:
    """Embed a KB entry and upsert into vec0 + meta tables.

    Best-effort: failures are logged but do not raise. Returns ``(ok, content_hash)``.
    When ``expected_hash`` matches the meta row, the embed is skipped and the
    function returns ``(True, expected_hash)``.
    """
    try:
        from lore.embeddings import (
            EMBEDDING_DIM,
            EmbeddingUnavailableError,
            compute_content_hash,
            encode_text,
        )
        from lore.embeddings import _model_name as _embedder_model_name
    except ImportError as exc:
        logger.warning("Embeddings module not importable: %s", exc)
        return False, None

    content_hash = compute_content_hash(title, content)
    if expected_hash is not None and expected_hash == content_hash:
        return True, content_hash

    try:
        vector = encode_text(f"{title}\n\n{content}")
    except EmbeddingUnavailableError as exc:
        logger.warning("Skipping embed for %s: %s", kb_id, exc)
        return False, content_hash
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to encode kb entry %s: %s", kb_id, exc)
        return False, content_hash

    if len(vector) != EMBEDDING_DIM:
        logger.error(
            "Embedding dim mismatch for %s: got %d, expected %d",
            kb_id,
            len(vector),
            EMBEDDING_DIM,
        )
        return False, content_hash

    try:
        import sqlite_vec
    except ImportError as exc:
        logger.warning("sqlite-vec not importable at write time: %s", exc)
        return False, content_hash

    try:
        conn = db._get_connection()
        blob = sqlite_vec.serialize_float32(vector)
        # Upsert into vec0: delete-then-insert; vec0 doesn't support INSERT OR REPLACE.
        conn.execute("DELETE FROM knowledge_kb_vec_embeddings WHERE kb_id = ?", (kb_id,))
        conn.execute(
            "INSERT INTO knowledge_kb_vec_embeddings(kb_id, embedding) VALUES(?, ?)",
            (kb_id, blob),
        )
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_kb_embedding_meta "
            "(kb_id, model_name, embedding_dim, content_hash, updated_at) "
            "VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
            (kb_id, _embedder_model_name(), EMBEDDING_DIM, content_hash),
        )
        conn.commit()
        return True, content_hash
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist embedding for %s: %s", kb_id, exc)
        return False, content_hash


def _delete_kb_embedding(kb_id: str) -> None:
    """Remove vec0 + meta rows for ``kb_id``. Idempotent / best-effort."""
    if os.getenv("DB_BACKEND", "").strip().lower() != "sqlite":
        return
    if not getattr(db, "vec_extension_loaded", False):
        return
    try:
        conn = db._get_connection()
        conn.execute("DELETE FROM knowledge_kb_vec_embeddings WHERE kb_id = ?", (kb_id,))
        # FK cascade should clear knowledge_kb_embedding_meta on its own when the
        # KB row is gone — but the row is gone before we get here, so be explicit.
        conn.execute("DELETE FROM knowledge_kb_embedding_meta WHERE kb_id = ?", (kb_id,))
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete embedding for %s: %s", kb_id, exc)


def _get_embedding_meta(kb_id: str) -> dict | None:
    """Return embedding_meta row for ``kb_id`` or None."""
    if os.getenv("DB_BACKEND", "").strip().lower() != "sqlite":
        return None
    if not getattr(db, "vec_extension_loaded", False):
        return None
    try:
        conn = db._get_connection()
        cur = conn.execute(
            "SELECT kb_id, model_name, embedding_dim, content_hash, created_at, updated_at "
            "FROM knowledge_kb_embedding_meta WHERE kb_id = ?",
            (kb_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "kb_id": row[0],
            "model_name": row[1],
            "embedding_dim": row[2],
            "content_hash": row[3],
            "created_at": row[4],
            "updated_at": row[5],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read embedding meta for %s: %s", kb_id, exc)
        return None


def handle_kb_add(
    topic: str,
    title: str,
    content: str,
    tags: list = None,
    author: str = None,
    source_type: str = None,
) -> dict:
    """Add KB entry.

    Optional attribution fields (author, source_type) support multi-agent
    provenance tracking. See `verified` flag (set via kb_update) for human review state.
    """
    try:
        kb_id = f"kb_{uuid.uuid4().hex[:12]}"

        entry = {
            "kb_id": kb_id,
            "topic": topic,
            "title": title,
            "content": content,
            "tags": tags or [],
            "author": author,
            "source_type": source_type,
        }

        db.table("knowledge.kb_entries").insert(entry).execute()

        # Best-effort embed at write time. The KB entry succeeds even if the
        # embedder fails (e.g. model download in flight). Backfill recovers.
        embedded = False
        if _semantic_write_enabled():
            embedded, _ = _embed_kb_entry(kb_id, title, content)

        return ResponseEnvelope.success(
            f"Added KB entry: {title}",
            {
                "kb_id": kb_id,
                "topic": topic,
                "author": author,
                "source_type": source_type,
                "embedded": embedded,
            },
        )
    except Exception as e:
        logger.error(f"Error adding KB entry: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_kb_search(
    query: str,
    topic: str = None,
    top_k: int = 20,
    semantic: bool = False,
    hybrid: bool = False,
    search_mode: str = None,
) -> dict:
    """Search KB entries.

    Routing (in order):
      1. If ``search_mode`` is given, use it verbatim ("fts" | "semantic" | "hybrid").
      2. Else if ``semantic=True``, use "semantic".
      3. Else if ``hybrid=True``, use "hybrid".
      4. Else fall back to legacy lexical search (existing FTS / LIKE path).

    Semantic and hybrid modes require ``LORE_SEMANTIC_SEARCH=true`` and the
    [semantic] extras. When unavailable, the call degrades to lexical search.
    """
    try:
        from lore import search as _search  # local import: tolerant of degraded envs

        # Determine the requested mode.
        if search_mode in {"fts", "semantic", "hybrid"}:
            requested_mode: str = search_mode
        elif semantic:
            requested_mode = "semantic"
        elif hybrid:
            requested_mode = "hybrid"
        else:
            requested_mode = "fts"

        # Bound top_k defensively (schema already constrains 1..200, but the
        # handler is also invoked from internal callers).
        try:
            top_k_int = int(top_k)
        except (TypeError, ValueError):
            top_k_int = 20
        top_k_int = max(1, min(200, top_k_int))

        # Decide whether we can actually use the semantic/hybrid path.
        backend = os.getenv("DB_BACKEND", "").strip().lower()
        is_sqlite = backend == "sqlite"
        wants_vectors = requested_mode in {"semantic", "hybrid"}

        # The SQLite hybrid path needs:
        #   - semantic enabled
        #   - vec0 extension loaded on the connection
        #   - fts5 available for hybrid mode
        #   - an embedder we can call
        sqlite_vectors_ok = (
            is_sqlite and _search.semantic_enabled() and getattr(db, "vec_extension_loaded", False)
        )

        if wants_vectors and sqlite_vectors_ok:
            from lore.embeddings import EmbeddingUnavailableError, encode_text, get_model_name

            def _encode_query(q: str) -> list[float] | None:
                try:
                    return encode_text(q)
                except EmbeddingUnavailableError as exc:
                    logger.warning("Embedding unavailable, falling back: %s", exc)
                    return None

            effective_mode = requested_mode
            if effective_mode == "hybrid" and not getattr(db, "fts5_available", False):
                effective_mode = "semantic"

            results = _search.hybrid_search_sqlite(
                db,
                query,
                topic=topic,
                top_k=top_k_int,
                search_mode=effective_mode,
                encode_query=_encode_query,
            )
            resp_data: dict = {
                "results": results,
                "count": len(results),
                "search_mode": effective_mode,
                "requested_mode": requested_mode,
            }
            if effective_mode in {"semantic", "hybrid"}:
                resp_data["model"] = get_model_name()
            if effective_mode == "hybrid":
                resp_data["rrf_k"] = _search.rrf_k()
            return ResponseEnvelope.success(
                f"Found {len(results)} KB entries (mode={effective_mode})",
                resp_data,
            )

        # SQLite FTS5-only fast path (no embeddings required).
        if is_sqlite and requested_mode == "fts" and getattr(db, "fts5_available", False):
            results = _search.fts5_search_sqlite(db, query, topic, top_k_int)
            # Strip content from response (consistent with hybrid path).
            results = [{k: v for k, v in r.items() if k != "content"} for r in results]
            return ResponseEnvelope.success(
                f"Found {len(results)} KB entries (mode=fts)",
                {
                    "results": results,
                    "count": len(results),
                    "search_mode": "fts",
                    "requested_mode": requested_mode,
                },
            )

        # Legacy lexical search path. Preserves today's behavior on SQLite (LIKE
        # via SqliteTableQuery.or_) and PostgreSQL (websearch_to_tsquery).
        safe_query = _sanitize_search_query(query)
        tsquery_safe = _sanitize_search_query(query).replace(" ", " & ")

        query_builder = db.table("knowledge.kb_entries").select(
            "kb_id, topic, title, tags, author, source_type, verified"
        )

        if topic:
            query_builder = query_builder.eq("topic", topic)

        try:
            query_builder = query_builder.or_(f"title.wfts.{safe_query},content.wfts.{safe_query}")
        except Exception as fts_error:
            logger.warning(f"Websearch FTS failed, using plain FTS: {fts_error}")
            query_builder = query_builder.or_(
                f"title.plfts.{tsquery_safe},content.plfts.{tsquery_safe}"
            )

        result = query_builder.limit(max(top_k_int, 50)).execute()

        envelope_data = {
            "results": result.data,
            "count": len(result.data),
            "search_mode": "lexical",
            "requested_mode": requested_mode,
        }
        if wants_vectors and not sqlite_vectors_ok:
            envelope_data["degraded"] = True
            envelope_data["degraded_reason"] = (
                "semantic search unavailable (LORE_SEMANTIC_SEARCH=false, "
                "sqlite-vec missing, or non-SQLite backend); served lexical results"
            )

        return ResponseEnvelope.success(
            f"Found {len(result.data)} KB entries",
            envelope_data,
        )
    except Exception as e:
        logger.error(f"Error searching KB: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_kb_get(kb_id: str) -> dict:
    """Get KB entry details."""
    try:
        result = (
            db.table("knowledge.kb_entries").select("*").eq("kb_id", kb_id).maybe_single().execute()
        )

        if not result or not result.data:
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"KB entry not found: {kb_id}")

        return ResponseEnvelope.success(f"KB entry: {result.data['title']}", result.data)
    except Exception as e:
        logger.error(f"Error getting KB entry: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_kb_list(topic: str = None) -> dict:
    """List KB entries."""
    try:
        query = (
            db.table("knowledge.kb_entries")
            .select("kb_id, topic, title, tags, author, source_type, verified, created_at")
            .order("created_at", desc=True)
        )

        if topic:
            query = query.eq("topic", topic)

        result = query.limit(100).execute()

        return ResponseEnvelope.success(
            f"Found {len(result.data)} KB entries",
            {"entries": result.data, "count": len(result.data)},
        )
    except Exception as e:
        logger.error(f"Error listing KB entries: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


_VERIFIED_SENTINEL = object()


def handle_kb_update(
    entry_id: str,
    content: str = None,
    metadata: dict = None,
    tags: list = None,
    topic: str = None,
    verified: Any = _VERIFIED_SENTINEL,
) -> dict:
    """Update existing KB entry with partial updates support.

    Updates only the provided fields, preserving existing fields not specified.
    Re-embeds content if content changes. Updates updated_at timestamp.

    `verified` accepts True (human-verified), False (disputed), or None (reset to
    unreviewed). Omit the argument entirely to leave the verified state unchanged.
    """
    try:
        # First, verify the entry exists
        existing_result = (
            db.table("knowledge.kb_entries")
            .select("*")
            .eq("kb_id", entry_id)
            .maybe_single()
            .execute()
        )

        if not existing_result or not existing_result.data:
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"KB entry not found: {entry_id}")

        existing_entry = existing_result.data

        # Build update object with only provided fields
        update_data = {}

        if content is not None:
            update_data["content"] = content
            # Note: In a full implementation, you'd regenerate embeddings here
            # when content changes. This is simplified for the basic CRUD operation.

        if metadata is not None:
            # Merge with existing metadata if it exists
            current_metadata = existing_entry.get("metadata", {})
            if isinstance(current_metadata, dict):
                current_metadata.update(metadata)
                update_data["metadata"] = current_metadata
            else:
                update_data["metadata"] = metadata

        if tags is not None:
            update_data["tags"] = tags

        if topic is not None:
            update_data["topic"] = topic

        if verified is not _VERIFIED_SENTINEL:
            # Allow True, False, or explicit None (reset to unreviewed)
            update_data["verified"] = verified

        # Always update the updated_at timestamp
        update_data["updated_at"] = datetime.utcnow().isoformat()

        # Only proceed with update if there are fields to update beyond timestamp
        if set(update_data.keys()) == {"updated_at"}:
            return ResponseEnvelope.error(
                ErrorCodes.INVALID_INPUT,
                "No fields provided for update. Specify content, metadata, tags, topic, and/or verified.",
            )

        # Perform the update
        db.table("knowledge.kb_entries").update(update_data).eq("kb_id", entry_id).execute()

        # Fetch and return the updated entry
        updated_result = (
            db.table("knowledge.kb_entries")
            .select("*")
            .eq("kb_id", entry_id)
            .maybe_single()
            .execute()
        )

        # Re-embed only if title or content changed. We compare against the
        # stored content_hash to avoid wasted encode calls.
        re_embedded = False
        if _semantic_write_enabled() and updated_result and updated_result.data:
            updated_entry = updated_result.data
            new_title = updated_entry.get("title") or existing_entry.get("title") or ""
            new_content = updated_entry.get("content") or existing_entry.get("content") or ""
            meta = _get_embedding_meta(entry_id)
            existing_hash = meta["content_hash"] if meta else None
            re_embedded, _ = _embed_kb_entry(
                entry_id,
                new_title,
                new_content,
                expected_hash=existing_hash,
            )

        updated_fields = list(update_data.keys())
        return ResponseEnvelope.success(
            f"Updated KB entry '{existing_entry['title']}' (fields: {', '.join(updated_fields)})",
            {
                "kb_id": entry_id,
                "updated_fields": updated_fields,
                "entry": updated_result.data,
                "re_embedded": re_embedded,
            },
        )

    except Exception as e:
        logger.error(f"Error updating KB entry {entry_id}: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_kb_delete(entry_id: str, confirm: bool = False) -> dict:
    """Delete existing KB entry from database with safety confirmation.

    Requires explicit confirmation for safety. Deletes entry and associated embeddings.
    Returns deleted entry details for audit trail.
    """
    try:
        # Safety check: require explicit confirmation
        if not confirm:
            return ResponseEnvelope.error(
                ErrorCodes.INVALID_INPUT,
                "Deletion requires explicit confirmation. Set confirm=True to proceed.",
            )

        # First, verify the entry exists and get its details for audit trail
        existing_result = (
            db.table("knowledge.kb_entries")
            .select("*")
            .eq("kb_id", entry_id)
            .maybe_single()
            .execute()
        )

        if not existing_result or not existing_result.data:
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"KB entry not found: {entry_id}")

        deleted_entry = existing_result.data
        entry_title = deleted_entry.get("title", "Untitled")

        # Delete the entry FIRST. SQLite vec0 tables don't support FK cascade,
        # but the embedding_meta row will cascade via the regular ON DELETE CASCADE
        # since it references knowledge_kb_entries.
        db.table("knowledge.kb_entries").delete().eq("kb_id", entry_id).execute()

        # Then clean up the vec0 row (and meta row defensively, in case PRAGMA
        # foreign_keys is off on this connection).
        _delete_kb_embedding(entry_id)

        return ResponseEnvelope.success(
            f"Deleted KB entry '{entry_title}' ({entry_id})",
            {"kb_id": entry_id, "deleted_entry": deleted_entry},
        )

    except Exception as e:
        logger.error(f"Error deleting KB entry {entry_id}: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


# =============================================================================
# Investigation Handlers
# (DB tables `research_notes` and `research_experiments` are unchanged; only the
# MCP tool/handler surface has been renamed to "investigation".)
# =============================================================================


def handle_investigation_add(topic: str, title: str, content: str, tags: list = None) -> dict:
    """Add an investigation entry."""
    try:
        note_id = f"note_{uuid.uuid4().hex[:12]}"

        note = {
            "note_id": note_id,
            "topic": topic,
            "title": title,
            "content": content,
            "tags": tags or [],
        }

        db.table("knowledge.research_notes").insert(note).execute()

        return ResponseEnvelope.success(f"Added investigation entry: {title}", {"note_id": note_id})
    except Exception as e:
        logger.error(f"Error adding investigation entry: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_investigation_list(topic: str = None) -> dict:
    """List investigations."""
    try:
        query = (
            db.table("knowledge.research_notes")
            .select("note_id, topic, title, tags, created_at")
            .order("created_at", desc=True)
        )

        if topic:
            query = query.eq("topic", topic)

        result = query.limit(100).execute()

        return ResponseEnvelope.success(
            f"Found {len(result.data)} investigations",
            {"investigations": result.data, "count": len(result.data)},
        )
    except Exception as e:
        logger.error(f"Error listing investigations: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_investigation_get(note_id: str) -> dict:
    """Get a single investigation entry."""
    try:
        result = (
            db.table("knowledge.research_notes")
            .select("*")
            .eq("note_id", note_id)
            .maybe_single()
            .execute()
        )

        if not result or not result.data:
            return ResponseEnvelope.error(
                ErrorCodes.NOT_FOUND, f"Investigation not found: {note_id}"
            )

        return ResponseEnvelope.success(f"Investigation: {result.data['title']}", result.data)
    except Exception as e:
        logger.error(f"Error getting investigation: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_investigation_log_experiment(
    title: str,
    hypothesis: str = None,
    methodology: str = None,
    results: dict = None,
    conclusion: str = None,
) -> dict:
    """Log an experiment within an investigation."""
    try:
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"

        experiment = {
            "experiment_id": experiment_id,
            "title": title,
            "hypothesis": hypothesis,
            "methodology": methodology,
            "results": results or {},
            "conclusion": conclusion,
        }

        db.table("knowledge.research_experiments").insert(experiment).execute()

        return ResponseEnvelope.success(
            f"Logged investigation experiment: {title}", {"experiment_id": experiment_id}
        )
    except Exception as e:
        logger.error(f"Error logging investigation experiment: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_investigation_list_experiments() -> dict:
    """List investigation experiments."""
    try:
        result = (
            db.table("knowledge.research_experiments")
            .select("experiment_id, title, created_at")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )

        return ResponseEnvelope.success(
            f"Found {len(result.data)} investigation experiments",
            {"experiments": result.data, "count": len(result.data)},
        )
    except Exception as e:
        logger.error(f"Error listing investigation experiments: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


# =============================================================================
# Journal Handlers
# =============================================================================


def handle_journal_append(entry_type: str, content: str, tags: list = None) -> dict:
    """Append journal entry."""
    try:
        entry_id = f"jrnl_{uuid.uuid4().hex[:12]}"

        entry = {
            "entry_id": entry_id,
            "date": date.today().isoformat(),
            "entry_type": entry_type,
            "content": content,
            "tags": tags or [],
        }

        db.table("knowledge.journal_entries").insert(entry).execute()

        return ResponseEnvelope.success(
            f"Added journal entry ({entry_type})", {"entry_id": entry_id, "date": entry["date"]}
        )
    except Exception as e:
        logger.error(f"Error appending journal: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_journal_list(limit: int = 20) -> dict:
    """List journal entries."""
    try:
        result = (
            db.table("knowledge.journal_entries")
            .select("entry_id, date, entry_type, tags")
            .order("date", desc=True)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        return ResponseEnvelope.success(
            f"Found {len(result.data)} journal entries",
            {"entries": result.data, "count": len(result.data)},
        )
    except Exception as e:
        logger.error(f"Error listing journal: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_journal_get(entry_id: str) -> dict:
    """Get journal entry."""
    try:
        result = (
            db.table("knowledge.journal_entries")
            .select("*")
            .eq("entry_id", entry_id)
            .maybe_single()
            .execute()
        )

        if not result or not result.data:
            return ResponseEnvelope.error(
                ErrorCodes.NOT_FOUND, f"Journal entry not found: {entry_id}"
            )

        return ResponseEnvelope.success(f"Journal entry from {result.data['date']}", result.data)
    except Exception as e:
        logger.error(f"Error getting journal entry: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_snapshot_config(config_name: str, config_data: dict) -> dict:
    """Snapshot config as journal entry."""
    try:
        entry_id = f"jrnl_{uuid.uuid4().hex[:12]}"

        content = f"Config snapshot: {config_name}\n\n```yaml\n{yaml.dump(config_data)}\n```"

        entry = {
            "entry_id": entry_id,
            "date": date.today().isoformat(),
            "entry_type": "milestone",
            "content": content,
            "tags": ["config-snapshot", config_name],
        }

        db.table("knowledge.journal_entries").insert(entry).execute()

        return ResponseEnvelope.success(
            f"Snapshotted config: {config_name}", {"entry_id": entry_id}
        )
    except Exception as e:
        logger.error(f"Error snapshotting config: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


# =============================================================================
# Document Ingestion Handlers (v1.3)
# =============================================================================


def handle_kb_ingest_doc(
    doc_path: str,
    strategy: str = "chunked",
    chunk_size: int = 2000,
    tags: list[str] = None,
    overwrite: bool = False,
    author: str = None,
    source_type: str = "system",
) -> dict:
    """Ingest single markdown document into KB."""
    try:
        doc_path = Path(doc_path).resolve()
        if not doc_path.exists():
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Document not found: {doc_path}")

        # Read document and extract frontmatter
        processor = DocumentProcessor(chunk_size=chunk_size)
        content, metadata = processor.read_document(str(doc_path))

        # Compute hash for change detection
        doc_hash = processor.compute_hash(content)

        # Check if document already synced
        existing_sync = (
            db.table("knowledge.kb_doc_sync")
            .select("*")
            .eq("doc_path", str(doc_path))
            .maybe_single()
            .execute()
        )

        if (
            existing_sync
            and existing_sync.data
            and existing_sync.data.get("doc_hash") == doc_hash
            and not overwrite
        ):
            return ResponseEnvelope.success(
                f"Document unchanged: {doc_path.name}",
                {
                    "doc_path": str(doc_path),
                    "status": "unchanged",
                    "doc_hash": doc_hash,
                    "kb_ids": existing_sync.data.get("kb_ids", []),
                },
            )

        # Delete old KB entries if overwriting
        if overwrite and existing_sync and existing_sync.data:
            old_kb_ids = existing_sync.data.get("kb_ids", [])
            if old_kb_ids:
                db.table("knowledge.kb_entries").delete().in_("kb_id", old_kb_ids).execute()
                logger.info(f"Deleted {len(old_kb_ids)} old KB entries for {doc_path.name}")

        # Extract topic and title
        topic = metadata.get("topic") or processor.extract_topic_from_path(str(doc_path))
        base_title = processor.generate_title(content, str(doc_path))
        doc_tags = tags or []
        if "tags" in metadata:
            doc_tags.extend(metadata["tags"])

        # Ingest based on strategy
        kb_ids = []

        if strategy == "full":
            # Single KB entry for entire document
            kb_id = f"kb_{uuid.uuid4().hex[:12]}"
            entry = {
                "kb_id": kb_id,
                "topic": topic,
                "title": base_title,
                "content": content,
                "tags": doc_tags,
                "source_doc": str(doc_path),
                "source_section": None,
                "line_range": [1, len(content.split("\n"))],
                "author": author,
                "source_type": source_type,
            }
            db.table("knowledge.kb_entries").insert(entry).execute()
            kb_ids.append(kb_id)

        elif strategy == "chunked":
            # Split by sections
            chunks = processor.chunk_by_sections(content, chunk_size)
            for i, chunk in enumerate(chunks):
                kb_id = f"kb_{uuid.uuid4().hex[:12]}"
                title = (
                    f"{base_title} - {chunk.section}"
                    if chunk.section
                    else f"{base_title} (part {i + 1})"
                )
                entry = {
                    "kb_id": kb_id,
                    "topic": topic,
                    "title": title,
                    "content": chunk.content,
                    "tags": doc_tags,
                    "source_doc": str(doc_path),
                    "source_section": chunk.section,
                    "line_range": [chunk.line_start, chunk.line_end],
                    "author": author,
                    "source_type": source_type,
                }
                db.table("knowledge.kb_entries").insert(entry).execute()
                kb_ids.append(kb_id)

        elif strategy == "summary":
            # TODO: Implement GPT summary strategy
            return ResponseEnvelope.error(
                ErrorCodes.INVALID_ARGUMENT,
                "Summary strategy not yet implemented. Use 'full' or 'chunked'.",
            )

        # Update sync tracking
        sync_data = {
            "doc_path": str(doc_path),
            "doc_hash": doc_hash,
            "kb_ids": kb_ids,
            "last_synced_at": datetime.utcnow().isoformat(),
            "last_modified_at": datetime.fromtimestamp(doc_path.stat().st_mtime).isoformat(),
            "strategy": strategy,
            "metadata": metadata,
        }

        if existing_sync and existing_sync.data:
            db.table("knowledge.kb_doc_sync").update(sync_data).eq(
                "doc_path", str(doc_path)
            ).execute()
            status = "updated"
        else:
            db.table("knowledge.kb_doc_sync").insert(sync_data).execute()
            status = "created"

        return ResponseEnvelope.success(
            f"Ingested {doc_path.name}: {len(kb_ids)} KB entries {status}",
            {
                "doc_path": str(doc_path),
                "kb_entries_created": len(kb_ids),
                "kb_ids": kb_ids,
                "doc_hash": doc_hash,
                "status": status,
            },
        )

    except Exception as e:
        logger.error(f"Error ingesting document {doc_path}: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


async def handle_kb_ingest_dir(
    dir_path: str,
    pattern: str = "*.md",
    strategy: str = "chunked",
    recursive: bool = True,
    exclude_patterns: list[str] = None,
    author: str = None,
    source_type: str = "system",
) -> dict:
    """Batch ingest directory (5x faster with async/await)."""
    import asyncio

    USE_ASYNC_INGESTION = os.getenv("ENABLE_ASYNC_INGESTION", "true").lower() == "true"

    try:
        dir_path = Path(dir_path).resolve()
        if not dir_path.exists():
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Directory not found: {dir_path}")

        # Find all matching files
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # Apply exclude patterns
        if exclude_patterns:
            import fnmatch

            files = [
                f
                for f in files
                if not any(fnmatch.fnmatch(str(f), pat) for pat in exclude_patterns)
            ]

        if not files:
            return ResponseEnvelope.success(
                f"No files found matching pattern: {pattern}",
                {"processed": 0, "created": 0, "updated": 0, "unchanged": 0, "errors": []},
            )

        created = 0
        updated = 0
        unchanged = 0
        errors = []

        if USE_ASYNC_INGESTION:
            # NEW: True async with asyncio.gather (5x faster, non-blocking)
            import aiofiles

            async def ingest_file_async(filepath: Path) -> dict:
                """Async file ingestion with aiofiles."""
                try:
                    # Async file I/O
                    async with aiofiles.open(filepath, encoding="utf-8") as f:
                        content = await f.read()

                    # Extract frontmatter and hash
                    processor = DocumentProcessor(chunk_size=2000)
                    _, metadata = processor.read_document(str(filepath))
                    doc_hash = processor.compute_hash(content)

                    # Check if document already synced
                    existing_sync = (
                        db.table("knowledge.kb_doc_sync")
                        .select("*")
                        .eq("doc_path", str(filepath))
                        .maybe_single()
                        .execute()
                    )

                    if (
                        existing_sync
                        and existing_sync.data
                        and existing_sync.data.get("doc_hash") == doc_hash
                    ):
                        return {"status": "unchanged", "doc_path": str(filepath)}

                    # Ingest document (synchronous DB calls - Supabase client isn't async)
                    result = handle_kb_ingest_doc(
                        doc_path=str(filepath),
                        strategy=strategy,
                        chunk_size=2000,
                        tags=None,
                        overwrite=False,
                        author=author,
                        source_type=source_type,
                    )

                    return {
                        "status": result.get("data", {}).get("status", "unknown"),
                        "doc_path": str(filepath),
                        "result": result,
                    }

                except Exception as e:
                    return {"status": "error", "doc_path": str(filepath), "error": str(e)}

            # Process files concurrently with asyncio.gather
            import time

            start = time.perf_counter()

            results = await asyncio.gather(*[ingest_file_async(f) for f in files])

            duration_s = time.perf_counter() - start
            logger.info(f"Async ingestion completed in {duration_s:.2f}s")

            # Aggregate results
            for res in results:
                status = res.get("status")
                if status == "created":
                    created += 1
                elif status == "updated":
                    updated += 1
                elif status == "unchanged":
                    unchanged += 1
                elif status == "error":
                    errors.append(
                        {
                            "doc_path": res.get("doc_path"),
                            "error": "ingestion_error",
                            "message": res.get("error"),
                        }
                    )
        else:
            # OLD: ThreadPoolExecutor (fallback for testing)
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(
                        handle_kb_ingest_doc,
                        str(f),
                        strategy,
                        2000,
                        None,
                        False,
                        author,
                        source_type,
                    ): f
                    for f in files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result.get("ok"):
                            status = result.get("data", {}).get("status")
                            if status == "created":
                                created += 1
                            elif status == "updated":
                                updated += 1
                            elif status == "unchanged":
                                unchanged += 1
                        else:
                            errors.append(
                                {
                                    "doc_path": str(file_path),
                                    "error": result.get("error"),
                                    "message": result.get("message"),
                                }
                            )
                    except Exception as e:
                        errors.append(
                            {
                                "doc_path": str(file_path),
                                "error": "unexpected_exception",
                                "message": str(e),
                            }
                        )

        return ResponseEnvelope.success(
            f"Processed {len(files)} files: {created} created, {updated} updated, {unchanged} unchanged",
            {
                "processed": len(files),
                "created": created,
                "updated": updated,
                "unchanged": unchanged,
                "errors": errors,
            },
        )

    except Exception as e:
        logger.error(f"Error ingesting directory {dir_path}: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_kb_sync_status(dir_path: str) -> dict:
    """Check sync state between source docs and KB."""
    try:
        dir_path = Path(dir_path).resolve()
        if not dir_path.exists():
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Directory not found: {dir_path}")

        # Get all markdown files in directory
        md_files = list(dir_path.rglob("*.md"))
        md_paths = {str(f.resolve()): f for f in md_files}

        # Get all sync records
        sync_records = db.table("knowledge.kb_doc_sync").select("*").execute()

        synced_paths = {r["doc_path"]: r for r in sync_records.data}

        # Classify files
        synced = 0
        modified = 0
        new = 0
        details = []

        for path_str, path_obj in md_paths.items():
            if path_str in synced_paths:
                sync_rec = synced_paths[path_str]
                file_mtime = datetime.fromtimestamp(path_obj.stat().st_mtime)
                last_synced = datetime.fromisoformat(
                    sync_rec["last_modified_at"].replace("Z", "+00:00")
                )

                if file_mtime > last_synced:
                    modified += 1
                    status = "modified"
                else:
                    synced += 1
                    status = "synced"

                details.append(
                    {
                        "doc_path": path_str,
                        "status": status,
                        "last_synced": sync_rec["last_synced_at"],
                        "doc_modified": file_mtime.isoformat(),
                        "kb_ids": sync_rec["kb_ids"],
                    }
                )
            else:
                new += 1
                details.append(
                    {
                        "doc_path": path_str,
                        "status": "new",
                        "last_synced": None,
                        "doc_modified": datetime.fromtimestamp(
                            path_obj.stat().st_mtime
                        ).isoformat(),
                        "kb_ids": [],
                    }
                )

        # Find orphaned KB entries (source doc deleted)
        orphaned_kb_ids = []
        for sync_path, sync_rec in synced_paths.items():
            if sync_path not in md_paths:
                orphaned_kb_ids.extend(sync_rec["kb_ids"])

        return ResponseEnvelope.success(
            f"Sync status: {synced} synced, {modified} modified, {new} new, {len(orphaned_kb_ids)} orphaned",
            {
                "total_docs": len(md_files),
                "synced": synced,
                "modified": modified,
                "new": new,
                "orphaned_kb_entries": len(orphaned_kb_ids),
                "details": details,
            },
        )

    except Exception as e:
        logger.error(f"Error checking sync status: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


# =============================================================================
# Semantic Search Maintenance Handlers (v0.6)
# =============================================================================


# Module-level lock that prevents concurrent backfill runs from competing for
# the embedder (and from double-embedding the same rows). Threading.Lock is
# sufficient here: handlers are sync and the process is single-tenant.
_BACKFILL_LOCK = threading.Lock()


def handle_kb_backfill_embeddings(
    batch_size: int = 32,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Embed any KB entries missing or stale embeddings.

    A row is considered stale when:
      - It has no row in ``knowledge_kb_embedding_meta``, OR
      - Its computed content_hash differs from the stored hash, OR
      - The stored model_name differs from the current LORE_EMBEDDING_MODEL.

    The backfill is wrapped in an advisory lock so two callers don't race
    each other to embed the same row. Per-row hash guards inside
    ``_embed_kb_entry`` make the loop safe even if the lock isn't honored
    (e.g. multi-process deployments).
    """
    if os.getenv("DB_BACKEND", "").strip().lower() != "sqlite":
        return ResponseEnvelope.error(
            ErrorCodes.INVALID_INPUT,
            "kb_backfill_embeddings currently supports the sqlite backend only.",
        )
    if not _semantic_write_enabled():
        return ResponseEnvelope.error(
            ErrorCodes.INVALID_INPUT,
            "Semantic search not enabled. Set LORE_SEMANTIC_SEARCH=true and install "
            "the [semantic] extra.",
        )

    try:
        from lore.embeddings import _model_name as _embedder_model_name
        from lore.embeddings import compute_content_hash
    except ImportError as exc:
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(exc))

    current_model = _embedder_model_name()

    if not _BACKFILL_LOCK.acquire(blocking=False):
        return ResponseEnvelope.error(
            ErrorCodes.INVALID_INPUT,
            "Another backfill is already running; try again shortly.",
        )

    try:
        conn = db._get_connection()
        # Pull (kb_id, title, content) joined with current meta hash + model.
        rows = conn.execute(
            """
            SELECT e.kb_id, e.title, e.content,
                   m.content_hash AS meta_hash,
                   m.model_name   AS meta_model
            FROM knowledge_kb_entries e
            LEFT JOIN knowledge_kb_embedding_meta m ON m.kb_id = e.kb_id
            """
        ).fetchall()

        to_embed: list[tuple[str, str, str]] = []
        skipped_current = 0
        for r in rows:
            kb_id, title, content, meta_hash, meta_model = (
                r[0],
                r[1] or "",
                r[2] or "",
                r[3],
                r[4],
            )
            expected_hash = compute_content_hash(title, content)
            if meta_hash == expected_hash and meta_model == current_model:
                skipped_current += 1
                continue
            to_embed.append((kb_id, title, content))

        if limit is not None:
            to_embed = to_embed[: int(limit)]

        if dry_run:
            return ResponseEnvelope.success(
                f"Backfill dry run: {len(to_embed)} entries would be embedded",
                {
                    "total_entries": len(rows),
                    "needs_embedding": len(to_embed),
                    "already_current": skipped_current,
                    "model": current_model,
                    "dry_run": True,
                },
            )

        # Encode in batches for throughput; persist one row at a time so a
        # mid-batch failure still produces partial progress.
        embedded = 0
        failed = 0
        bs = max(1, int(batch_size))
        for start in range(0, len(to_embed), bs):
            batch = to_embed[start : start + bs]
            for kb_id, title, content in batch:
                ok, _ = _embed_kb_entry(kb_id, title, content)
                if ok:
                    embedded += 1
                else:
                    failed += 1

        return ResponseEnvelope.success(
            f"Backfill complete: embedded={embedded}, failed={failed}",
            {
                "total_entries": len(rows),
                "embedded": embedded,
                "failed": failed,
                "already_current": skipped_current,
                "model": current_model,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("kb_backfill_embeddings failed: %s", exc, exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(exc))
    finally:
        _BACKFILL_LOCK.release()


def handle_kb_embedding_status() -> dict:
    """Report embedding coverage and configuration."""
    if os.getenv("DB_BACKEND", "").strip().lower() != "sqlite":
        return ResponseEnvelope.success(
            "kb_embedding_status: non-sqlite backend (PostgreSQL semantic is Phase 2)",
            {
                "backend": os.getenv("DB_BACKEND", "unknown"),
                "semantic_enabled": False,
                "phase": 2,
            },
        )

    try:
        from lore.embeddings import EMBEDDING_DIM
        from lore.embeddings import _model_name as _embedder_model_name
    except ImportError:
        return ResponseEnvelope.success(
            "Embeddings module not installed",
            {
                "backend": "sqlite",
                "semantic_enabled": False,
                "embeddings_module": False,
            },
        )

    try:
        conn = db._get_connection()
        total = conn.execute("SELECT COUNT(*) FROM knowledge_kb_entries").fetchone()[0]
        if getattr(db, "vec_extension_loaded", False):
            embedded = conn.execute("SELECT COUNT(*) FROM knowledge_kb_embedding_meta").fetchone()[
                0
            ]
            per_model_rows = conn.execute(
                "SELECT model_name, COUNT(*) FROM knowledge_kb_embedding_meta "
                "GROUP BY model_name ORDER BY COUNT(*) DESC"
            ).fetchall()
            per_model = {row[0]: row[1] for row in per_model_rows}
        else:
            embedded = 0
            per_model = {}
        missing = max(0, total - embedded)

        return ResponseEnvelope.success(
            f"Embedding coverage: {embedded}/{total} entries",
            {
                "backend": "sqlite",
                "semantic_enabled": _semantic_write_enabled(),
                "vec_extension_loaded": bool(getattr(db, "vec_extension_loaded", False)),
                "fts5_available": bool(getattr(db, "fts5_available", False)),
                "current_model": _embedder_model_name(),
                "embedding_dim": EMBEDDING_DIM,
                "total_entries": total,
                "embedded": embedded,
                "missing": missing,
                "coverage_pct": round(100.0 * embedded / total, 2) if total else 0.0,
                "per_model": per_model,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("kb_embedding_status failed: %s", exc, exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(exc))


# =============================================================================
# MCP Index Handlers
# =============================================================================


def handle_mcp_index_scan(triggered_by: str = "manual", config_filter: bool = True) -> dict:
    """Scan all MCP servers and index their tools."""
    try:
        scanner = MCPIndexScanner(db)
        result = scanner.scan_all_servers(triggered_by=triggered_by, config_filter=config_filter)

        return ResponseEnvelope.success(
            f"Scanned {result['servers_scanned']} servers, indexed {result['tools_indexed']} tools",
            result,
        )

    except Exception as e:
        logger.error(f"Error scanning MCP index: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_mcp_index_search(query: str, category: str = None, limit: int = 20) -> dict:
    """Search for MCP tools by description/capability."""
    try:
        scanner = MCPIndexScanner(db)
        results = scanner.search_tools(query, category, limit)

        # Auto re-index watchdog: if results are empty, check if index is stale
        re_index_triggered = False
        if len(results) == 0:
            # Query last scan time from mcp_index_versions
            try:
                last_scan_result = (
                    db.table("mcp_index_versions")
                    .select("scan_time")
                    .order("scan_time", desc=True)
                    .limit(1)
                    .execute()
                )

                if last_scan_result.data and len(last_scan_result.data) > 0:
                    last_scan_time = datetime.fromisoformat(
                        last_scan_result.data[0]["scan_time"].replace("Z", "+00:00")
                    )
                    time_since_scan = (
                        datetime.now(last_scan_time.tzinfo) - last_scan_time
                    ).total_seconds()

                    # If stale (>1 hour = 3600 seconds), trigger background re-index
                    if time_since_scan > 3600:
                        logger.info(
                            f"MCP Index stale ({time_since_scan / 3600:.1f}h old), triggering background re-index"
                        )

                        # Launch background re-index using threading
                        def background_reindex():
                            try:
                                scanner_bg = MCPIndexScanner(db)
                                result = scanner_bg.scan_all_servers(triggered_by="auto_watchdog")
                                logger.info(
                                    f"Auto re-index complete: {result['servers_scanned']} servers, {result['tools_indexed']} tools"
                                )
                            except Exception as e:
                                logger.error(f"Background re-index failed: {e}", exc_info=True)

                        thread = threading.Thread(target=background_reindex, daemon=True)
                        thread.start()
                        re_index_triggered = True
                else:
                    # No scan history found, trigger initial scan
                    logger.info(
                        "No MCP Index scan history found, triggering initial background scan"
                    )

                    def background_reindex():
                        try:
                            scanner_bg = MCPIndexScanner(db)
                            result = scanner_bg.scan_all_servers(
                                triggered_by="auto_watchdog_initial"
                            )
                            logger.info(
                                f"Initial auto scan complete: {result['servers_scanned']} servers, {result['tools_indexed']} tools"
                            )
                        except Exception as e:
                            logger.error(f"Background initial scan failed: {e}", exc_info=True)

                    thread = threading.Thread(target=background_reindex, daemon=True)
                    thread.start()
                    re_index_triggered = True

            except Exception as e:
                logger.warning(f"Failed to check MCP Index staleness: {e}")

        # Build response with re-index metadata
        response_data = {
            "results": results,
            "query": query,
            "category": category,
            "re_index_triggered": re_index_triggered,
        }

        message = f"Found {len(results)} tools matching '{query}'"
        if re_index_triggered:
            message += " (re-index triggered in background, retry in 30 seconds)"

        return ResponseEnvelope.success(message, response_data)

    except Exception as e:
        logger.error(f"Error searching MCP index: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_mcp_index_get_server(server_id: str) -> dict:
    """Get all tools for a specific MCP server."""
    try:
        scanner = MCPIndexScanner(db)
        result = scanner.get_server_tools(server_id)

        if not result:
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Server not found: {server_id}")

        server = result["server"]
        tools = result["tools"]

        return ResponseEnvelope.success(
            f"Server {server_id} has {len(tools)} tools", {"server": server, "tools": tools}
        )

    except Exception as e:
        logger.error(f"Error getting server tools: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_mcp_index_get_tool(tool_name: str) -> dict:
    """Get detailed information about a specific tool."""
    try:
        scanner = MCPIndexScanner(db)
        tool = scanner.get_tool_details(tool_name)

        if not tool:
            return ResponseEnvelope.error(ErrorCodes.NOT_FOUND, f"Tool not found: {tool_name}")

        return ResponseEnvelope.success(f"Found tool: {tool['full_name']}", {"tool": tool})

    except Exception as e:
        logger.error(f"Error getting tool details: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_mcp_index_rebuild() -> dict:
    """Force rebuild of entire MCP index."""
    try:
        scanner = MCPIndexScanner(db)
        result = scanner.scan_all_servers(triggered_by="rebuild")

        return ResponseEnvelope.success(
            f"Rebuilt index: {result['servers_scanned']} servers, {result['tools_indexed']} tools",
            result,
        )

    except Exception as e:
        logger.error(f"Error rebuilding MCP index: {e}", exc_info=True)
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


# =============================================================================
# Search Handlers (consolidated from search-mcp)
# =============================================================================


def search_file_content(file_path: Path, query: str) -> dict | None:
    """Search a single file for query string."""
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
            if query.lower() in content.lower():
                # Find context around match
                pos = content.lower().find(query.lower())
                start = max(0, pos - 100)
                end = min(len(content), pos + len(query) + 100)
                snippet = content[start:end]

                return {
                    "file": str(file_path),
                    "match_count": content.lower().count(query.lower()),
                    "snippet": snippet,
                    "file_size": file_path.stat().st_size,
                }
    except Exception as e:
        logger.warning(f"Error searching {file_path}: {e}")
    return None


def handle_search_local(query: str, paths: list[str] = None, file_types: list[str] = None) -> dict:
    """Search local files by content."""
    try:
        # Default paths
        if not paths:
            paths = [str(LATVIAN_LEARNING_ROOT), str(LATVIAN_XTTS_ROOT), str(KNOWLEDGE_DATA_DIR)]

        # Default file types
        if not file_types:
            file_types = ["txt", "json", "md", "py", "yaml", "yml"]

        results = []
        file_count = 0

        for search_path in paths:
            path_obj = Path(search_path)
            if not path_obj.exists():
                continue

            for file_type in file_types:
                for file_path in path_obj.rglob(f"*.{file_type}"):
                    file_count += 1
                    result = search_file_content(file_path, query)
                    if result:
                        results.append(result)

                    # Limit results
                    if len(results) >= 100:
                        break

        results.sort(key=lambda x: x["match_count"], reverse=True)

        return ResponseEnvelope.success(
            f"Found {len(results)} matches in {file_count} files",
            {
                "results": results[:50],  # Return top 50
                "total_matches": len(results),
                "files_searched": file_count,
            },
        )
    except Exception as e:
        logger.error(f"Error in handle_search_local: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_search_corpora(query: str, corpus_ids: list[str] = None) -> dict:
    """Search across corpus manifests."""
    try:
        results = []
        corpora_dir = INGEST_ROOT / "corpora"

        if not corpora_dir.exists():
            return ResponseEnvelope.success(
                "Corpora directory not found (expected until data ingested)",
                {"results": [], "count": 0},
            )

        # Search corpus manifest files
        for manifest_file in corpora_dir.glob("*.jsonl"):
            # Filter by corpus_ids if specified
            if corpus_ids and manifest_file.stem not in corpus_ids:
                continue

            with open(manifest_file) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = json.loads(line)
                        # Search in transcript and metadata
                        if (
                            query.lower() in entry.get("text", "").lower()
                            or query.lower() in json.dumps(entry.get("metadata", {})).lower()
                        ):
                            results.append(
                                {
                                    "corpus": manifest_file.stem,
                                    "line": line_num,
                                    "segment_id": entry.get("segment_id", "unknown"),
                                    "text": entry.get("text", "")[:200],
                                    "metadata": entry.get("metadata", {}),
                                }
                            )

                            if len(results) >= 100:
                                break
                    except json.JSONDecodeError:
                        continue

            if len(results) >= 100:
                break

        return ResponseEnvelope.success(
            f"Found {len(results)} matches in corpora",
            {"results": results[:50], "total_matches": len(results)},
        )
    except Exception as e:
        logger.error(f"Error in handle_search_corpora: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_search_transcripts(query: str, speaker: str = None) -> dict:
    """Search transcript segments."""
    try:
        results = []

        # Search in whisper extracted directories
        search_dirs = [
            LATVIAN_XTTS_ROOT / "whisper_extracted",
            LATVIAN_XTTS_ROOT / "whisper_extracted_enhanced",
            LATVIAN_XTTS_ROOT / "whisper_extracted_normalized",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for json_file in search_dir.rglob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                        # Filter by speaker if specified
                        if speaker and data.get("speaker") != speaker:
                            continue

                        # Search in text
                        text = data.get("text", "")
                        if query.lower() in text.lower():
                            results.append(
                                {
                                    "file": str(json_file.relative_to(LATVIAN_XTTS_ROOT)),
                                    "speaker": data.get("speaker", "unknown"),
                                    "text": text[:200],
                                    "duration": data.get("duration_seconds"),
                                    "timestamp": data.get("start_time"),
                                }
                            )

                            if len(results) >= 100:
                                break
                except (OSError, json.JSONDecodeError):
                    continue

            if len(results) >= 100:
                break

        return ResponseEnvelope.success(
            f"Found {len(results)} transcript matches",
            {"results": results[:50], "total_matches": len(results)},
        )
    except Exception as e:
        logger.error(f"Error in handle_search_transcripts: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_multi_search(query: str) -> dict:
    """Combined search across all sources."""
    try:
        results = {"local": [], "corpora": [], "transcripts": [], "knowledge": {}}

        # Local search (limited)
        local_result = handle_search_local(
            query, paths=[str(KNOWLEDGE_DATA_DIR)], file_types=["json", "md"]
        )
        if local_result.get("ok"):
            results["local"] = local_result["data"]["results"][:10]

        # Knowledge search using kb_search
        knowledge_result = handle_kb_search(query)
        if knowledge_result.get("ok"):
            results["knowledge"] = {"kb_entries": knowledge_result["data"]["results"]}

        # Corpora search
        corpora_result = handle_search_corpora(query)
        if corpora_result.get("ok"):
            results["corpora"] = corpora_result["data"]["results"][:10]

        # Transcript search
        transcript_result = handle_search_transcripts(query)
        if transcript_result.get("ok"):
            results["transcripts"] = transcript_result["data"]["results"][:10]

        total_matches = (
            len(results["local"])
            + len(results["knowledge"].get("kb_entries", []))
            + len(results["corpora"])
            + len(results["transcripts"])
        )

        return ResponseEnvelope.success(
            f"Multi-search found {total_matches} matches across all sources",
            {"results": results, "total_matches": total_matches},
        )
    except Exception as e:
        logger.error(f"Error in handle_multi_search: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_deduplicate_results(results: list[dict], threshold: float = 0.9) -> dict:
    """Remove duplicate search results."""
    try:
        # Simple deduplication based on exact text matches
        seen = set()
        deduped = []

        for result in results:
            # Create a key from result text/content
            key = result.get("text", "") or result.get("content", "") or result.get("snippet", "")
            key_normalized = key.lower().strip()

            if key_normalized and key_normalized not in seen:
                seen.add(key_normalized)
                deduped.append(result)

        removed = len(results) - len(deduped)

        return ResponseEnvelope.success(
            f"Removed {removed} duplicates, {len(deduped)} unique results remaining",
            {"results": deduped, "removed_count": removed, "unique_count": len(deduped)},
        )
    except Exception as e:
        logger.error(f"Error in handle_deduplicate_results: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def handle_cluster_results(results: list[dict], num_clusters: int = 5) -> dict:
    """Cluster search results by topic."""
    try:
        clusters = {}

        for result in results:
            # Determine cluster key
            if "file" in result:
                file_path = Path(result["file"])
                cluster_key = file_path.suffix or "other"
            elif "corpus" in result:
                cluster_key = "corpus"
            elif "speaker" in result:
                cluster_key = "transcript"
            else:
                cluster_key = "other"

            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(result)

        cluster_summary = {cluster: len(items) for cluster, items in clusters.items()}

        return ResponseEnvelope.success(
            f"Clustered {len(results)} results into {len(clusters)} groups",
            {
                "clusters": clusters,
                "cluster_summary": cluster_summary,
                "total_results": len(results),
            },
        )
    except Exception as e:
        logger.error(f"Error in handle_cluster_results: {e}")
        return ResponseEnvelope.error(ErrorCodes.UNEXPECTED_EXCEPTION, str(e))


def main() -> None:
    """Entry point for the 'lore-mcp' console script.

    By default runs as an MCP stdio server. If --host/--port are passed,
    starts the HTTP/SSE wrapper instead (useful for systemd or Docker
    deployments where the MCP client speaks HTTP).

    Backend configuration is read entirely from the environment
    (DB_BACKEND, KNOWLEDGE_DATA_DIR, DB_HOST, DB_PORT, DB_NAME, DB_USER,
    DB_PASSWORD, SUPABASE_URL, SUPABASE_KEY). The default backend is
    'sqlite' so a clean checkout boots without external services.
    """
    import argparse
    import asyncio

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="lore-mcp",
        description="Lore MCP server (stdio by default, HTTP/SSE with --host/--port).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address for HTTP/SSE mode (e.g. 0.0.0.0). Omit for stdio mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port for HTTP/SSE mode (e.g. 5555). Omit for stdio mode.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lore-mcp {__version__}",
    )
    args = parser.parse_args()

    # Default to the zero-friction SQLite backend if none is configured.
    # All other backends require their own env vars and will fail loudly
    # in db_client if misconfigured — we never inject credentials here.
    os.environ.setdefault("DB_BACKEND", "sqlite")

    global db
    db = get_db_client()
    backend = os.getenv("DB_BACKEND", "sqlite")
    logger.info(f"Connected to database backend: {backend}")

    if args.host is not None or args.port is not None:
        # HTTP/SSE mode — delegate to the wrapper, which mounts our 'app'.
        import uvicorn

        from .mcp_http_wrapper_sse import create_app

        host = args.host or "127.0.0.1"
        port = args.port or 5555
        starlette_app = create_app(app, "lore.server")
        logger.info(f"Starting Lore MCP HTTP server on {host}:{port}")
        uvicorn.run(starlette_app, host=host, port=port, log_level="info")
        return

    logger.info("Starting Lore MCP server (stdio)")

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
