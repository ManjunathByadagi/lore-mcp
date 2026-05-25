#!/usr/bin/env python3
"""
Lore MCP Server (FastMCP 3.3.1 edition).

This is a drop-in alternative front-end for the Lore knowledge layer built on
PrefectHQ/FastMCP. It reuses the *verbatim* handler logic from ``lore.server``
(``handle_*`` functions, ``_coerce_arguments``, ``json_serializer``,
``ResponseEnvelope``) so the database layer, business rules, and
input/output schemas are 100% backward compatible.

Migration notes (Issue #7) and the plain-JSON endpoint (Issue #9):

* Tools are registered as thin ``@mcp.tool()`` wrappers that coerce the
  Claude Code serialization quirks (comma/space-separated lists) and forward
  to the original handler. FastMCP runs sync tools in a threadpool, so the
  sync handlers are kept sync; ``kb_ingest_dir`` is async and is awaited.
* The module-global ``db`` lives in ``lore.server``. The FastMCP lifespan
  initialises it once at startup and the handlers keep referencing it — no
  handler signature changes.
* HTTP surface:
    - ``/mcp``      plain JSON-RPC (backward compatible with the prior custom
                    SSE wrapper's ``/mcp`` contract; now bare JSON, no SSE).
    - ``/jsonrpc``  plain JSON-RPC (Issue #9 — explicit bare-JSON endpoint).
    - ``/health``   liveness probe.
    - ``/stream``   FastMCP-native Streamable HTTP transport (json_response,
                    stateless) for clients that speak the real MCP protocol.
* CORS is enabled with ``allow_origins=["*"]`` to match the prior wrapper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastmcp import FastMCP
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Reuse the verbatim handler logic and helpers from the existing server.
# Importing lore.server is intentional: it owns the module-global ``db`` and
# the handler functions. We rebind ``lore.server.db`` in the lifespan.
# ---------------------------------------------------------------------------
from lore import server as _srv
from lore.db_client import get_db_client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic emits a one-time warning when a sentinel object() is used as a
# parameter default (kb_update.verified). The default is correctly excluded
# from the JSON schema; silence the noise so logs stay clean.
warnings.filterwarnings(
    "ignore",
    message="Default value .* is not JSON serializable; excluding default from JSON schema",
)

# Sentinel reused verbatim from lore.server so kb_update can distinguish
# "argument omitted" (leave unchanged) from "verified=null" (reset).
_VERIFIED_SENTINEL = _srv._VERIFIED_SENTINEL


def _json(result: dict) -> str:
    """Serialize a handler's business-envelope dict to a JSON string.

    Uses the same serializer as the stdio server so datetime objects render
    identically across transports.
    """
    return json.dumps(result, default=_srv.json_serializer)


def _coerce_tags(value: Any) -> Any:
    """Coerce a tags-like value into a list using the legacy rules.

    Mirrors ``lore.server._coerce_arguments`` for a single string-array field:
      1. JSON-encoded:    '["a","b"]' -> ["a","b"]
      2. Comma-separated: 'a,b,c'      -> ["a","b","c"]
      3. Space-separated: 'a b c'      -> ["a","b","c"]

    ``None`` and existing lists pass through unchanged.
    """
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        if "," in stripped:
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
        else:
            parts = stripped.split()
        return parts or None
    return value


def _semantic_enabled() -> bool:
    """Whether semantic search is enabled via env var (matches embeddings module)."""
    return os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Lifespan: initialise the module-global db once and preload the embedder.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lore_lifespan(server: FastMCP):  # noqa: ANN201 - FastMCP lifespan signature
    """FastMCP lifespan: initialise DB and (optionally) preload the embedder.

    Keeps ``lore.server.db`` as the single source of truth so the verbatim
    handlers keep working unchanged. Default backend is SQLite for a
    zero-friction boot.
    """
    os.environ.setdefault("DB_BACKEND", "sqlite")
    try:
        _srv.db = get_db_client()
        logger.info("DB backend: %s", os.getenv("DB_BACKEND", "sqlite"))
    except Exception as exc:
        logger.error("FATAL: database initialization failed: %s", exc)
        raise  # abort startup — do not run with db=None

    if _semantic_enabled():
        try:
            from lore.embeddings import get_embedder

            get_embedder()  # preload model at startup
            logger.info("Embedding model preloaded at startup")
        except Exception as exc:  # noqa: BLE001 - best-effort preload
            logger.warning("Embedder preload skipped: %s", exc)

    yield {}


mcp: FastMCP = FastMCP("knowledge-mcp", version="0.6.0", lifespan=lore_lifespan)


# ===========================================================================
# Tool wrappers (34). Each accepts typed params, coerces Claude Code quirks,
# calls the verbatim handler, and returns a JSON string. Descriptions are
# copied from lore.server._TOOL_DEFINITIONS to keep parity.
# ===========================================================================

# --- Knowledge Base (6) ----------------------------------------------------


@mcp.tool(description="Add a knowledge base entry")
def kb_add(
    topic: str,
    title: str,
    content: str,
    tags: str | list[str] | None = None,
    author: str | None = None,
    source_type: str | None = None,
) -> str:
    return _json(
        _srv.handle_kb_add(
            topic=topic,
            title=title,
            content=content,
            tags=_coerce_tags(tags),
            author=author,
            source_type=source_type,
        )
    )


@mcp.tool(
    description=(
        "Search knowledge base. Lexical FTS5 (or LIKE fallback) by default; "
        "set semantic=true / hybrid=true / search_mode=hybrid to use vector "
        "embeddings + RRF fusion (requires LORE_SEMANTIC_SEARCH=true)."
    )
)
def kb_search(
    query: str,
    topic: str | None = None,
    top_k: int = 20,
    semantic: bool = False,
    hybrid: bool = False,
    search_mode: Literal["fts", "semantic", "hybrid"] | None = None,
    session_id: str | None = None,
    parent_query_id: str | None = None,
    required_requery: bool = False,
    caller_agent: str | None = None,
) -> str:
    return _json(
        _srv.handle_kb_search(
            query=query,
            topic=topic,
            top_k=top_k,
            semantic=semantic,
            hybrid=hybrid,
            search_mode=search_mode,
            session_id=session_id,
            parent_query_id=parent_query_id,
            required_requery=required_requery,
            caller_agent=caller_agent,
        )
    )


@mcp.tool(description="Get full KB entry by ID")
def kb_get(kb_id: str) -> str:
    return _json(_srv.handle_kb_get(kb_id=kb_id))


@mcp.tool(description="List KB entries")
def kb_list(topic: str | None = None) -> str:
    return _json(_srv.handle_kb_list(topic=topic))


@mcp.tool(description="Update existing KB entry content and metadata")
def kb_update(
    entry_id: str,
    content: str | None = None,
    metadata: dict | None = None,
    tags: str | list[str] | None = None,
    topic: str | None = None,
    verified: bool | None = _VERIFIED_SENTINEL,
) -> str:
    # Preserve the sentinel semantics: only forward ``verified`` when the
    # client actually supplied it (so omission != reset-to-null).
    return _json(
        _srv.handle_kb_update(
            entry_id=entry_id,
            content=content,
            metadata=metadata,
            tags=_coerce_tags(tags),
            topic=topic,
            verified=verified,
        )
    )


@mcp.tool(description="Delete existing KB entry from database")
def kb_delete(entry_id: str, confirm: bool = False) -> str:
    return _json(_srv.handle_kb_delete(entry_id=entry_id, confirm=confirm))


# --- Investigations (5) ----------------------------------------------------


@mcp.tool(description="Add an investigation entry (open or append to an ops investigation)")
def investigation_add(
    topic: str,
    title: str,
    content: str,
    tags: str | list[str] | None = None,
) -> str:
    return _json(
        _srv.handle_investigation_add(
            topic=topic, title=title, content=content, tags=_coerce_tags(tags)
        )
    )


@mcp.tool(description="List investigations")
def investigation_list(topic: str | None = None) -> str:
    return _json(_srv.handle_investigation_list(topic=topic))


@mcp.tool(description="Get a single investigation entry by ID")
def investigation_get(note_id: str) -> str:
    return _json(_srv.handle_investigation_get(note_id=note_id))


@mcp.tool(
    description=(
        "Log a structured experiment within an investigation "
        "(hypothesis, methodology, results, conclusion)"
    )
)
def investigation_log_experiment(
    title: str,
    hypothesis: str | None = None,
    methodology: str | None = None,
    results: dict | None = None,
    conclusion: str | None = None,
) -> str:
    return _json(
        _srv.handle_investigation_log_experiment(
            title=title,
            hypothesis=hypothesis,
            methodology=methodology,
            results=results,
            conclusion=conclusion,
        )
    )


@mcp.tool(description="List logged investigation experiments")
def investigation_list_experiments() -> str:
    return _json(_srv.handle_investigation_list_experiments())


# --- Journal (4) -----------------------------------------------------------


@mcp.tool(description="Append journal entry")
def journal_append(
    entry_type: Literal["daily", "milestone", "reflection", "idea"],
    content: str,
    tags: str | list[str] | None = None,
) -> str:
    return _json(
        _srv.handle_journal_append(entry_type=entry_type, content=content, tags=_coerce_tags(tags))
    )


@mcp.tool(description="List journal entries")
def journal_list(limit: int = 20) -> str:
    return _json(_srv.handle_journal_list(limit=limit))


@mcp.tool(description="Get journal entry")
def journal_get(entry_id: str) -> str:
    return _json(_srv.handle_journal_get(entry_id=entry_id))


@mcp.tool(description="Snapshot current config")
def snapshot_config(config_name: str, config_data: dict) -> str:
    return _json(_srv.handle_snapshot_config(config_name=config_name, config_data=config_data))


# --- Document Ingestion (3) ------------------------------------------------


@mcp.tool(description="Ingest single markdown file into KB with change detection")
def kb_ingest_doc(
    doc_path: str,
    strategy: Literal["full", "chunked", "summary"] = "chunked",
    chunk_size: int = 2000,
    tags: str | list[str] | None = None,
    overwrite: bool = False,
    author: str | None = None,
    source_type: str = "system",
) -> str:
    return _json(
        _srv.handle_kb_ingest_doc(
            doc_path=doc_path,
            strategy=strategy,
            chunk_size=chunk_size,
            tags=_coerce_tags(tags),
            overwrite=overwrite,
            author=author,
            source_type=source_type,
        )
    )


@mcp.tool(description="Batch ingest directory of markdown files")
async def kb_ingest_dir(
    dir_path: str,
    pattern: str = "*.md",
    strategy: Literal["full", "chunked", "summary"] = "chunked",
    recursive: bool = True,
    exclude_patterns: str | list[str] | None = None,
    author: str | None = None,
    source_type: str = "system",
    confirm_production: bool = False,
) -> str:
    # handle_kb_ingest_dir is async — await it.
    result = await _srv.handle_kb_ingest_dir(
        dir_path=dir_path,
        pattern=pattern,
        strategy=strategy,
        recursive=recursive,
        exclude_patterns=_coerce_tags(exclude_patterns),
        author=author,
        source_type=source_type,
        confirm_production=confirm_production,
    )
    return _json(result)


@mcp.tool(description="Check sync state between source docs and KB")
def kb_sync_status(dir_path: str) -> str:
    return _json(_srv.handle_kb_sync_status(dir_path=dir_path))


# --- Semantic Search (2) ---------------------------------------------------


@mcp.tool(
    description=(
        "Embed any KB entries that are missing or stale (model/content changed). "
        "Idempotent: skips entries whose stored content_hash still matches. "
        "Requires LORE_SEMANTIC_SEARCH=true."
    )
)
def kb_backfill_embeddings(
    batch_size: int = 32,
    limit: int | None = None,
    dry_run: bool = False,
    confirm_production: bool = False,
) -> str:
    return _json(
        _srv.handle_kb_backfill_embeddings(
            batch_size=batch_size,
            limit=limit,
            dry_run=dry_run,
            confirm_production=confirm_production,
        )
    )


@mcp.tool(
    description=(
        "Report embedding coverage: total entries, embedded count, missing count, "
        "current model, and per-model breakdown."
    )
)
def kb_embedding_status() -> str:
    return _json(_srv.handle_kb_embedding_status())


# --- MCP Index (5) ---------------------------------------------------------


@mcp.tool(
    description=(
        "Scan all MCP servers and index their tools. By default, scans only "
        "configured servers (66% token savings)."
    )
)
def mcp_index_scan(triggered_by: str = "manual", config_filter: bool = True) -> str:
    return _json(_srv.handle_mcp_index_scan(triggered_by=triggered_by, config_filter=config_filter))


@mcp.tool(description="Search for MCP tools by description/capability")
def mcp_index_search(query: str, category: str | None = None, limit: int = 20) -> str:
    return _json(_srv.handle_mcp_index_search(query=query, category=category, limit=limit))


@mcp.tool(description="Get all tools for a specific MCP server")
def mcp_index_get_server(server_id: str) -> str:
    return _json(_srv.handle_mcp_index_get_server(server_id=server_id))


@mcp.tool(description="Get detailed information about a specific tool")
def mcp_index_get_tool(tool_name: str) -> str:
    return _json(_srv.handle_mcp_index_get_tool(tool_name=tool_name))


@mcp.tool(description="Force rebuild of entire MCP index (same as mcp_index_scan)")
def mcp_index_rebuild() -> str:
    return _json(_srv.handle_mcp_index_rebuild())


# --- Search (6) ------------------------------------------------------------


@mcp.tool(description="Search local files by content (lexical mode)")
def search_local(
    query: str,
    paths: str | list[str] | None = None,
    file_types: str | list[str] | None = None,
) -> str:
    return _json(
        _srv.handle_search_local(
            query=query, paths=_coerce_tags(paths), file_types=_coerce_tags(file_types)
        )
    )


@mcp.tool(description="Search across corpus manifests (JSONL files)")
def search_corpora(query: str, corpus_ids: str | list[str] | None = None) -> str:
    return _json(_srv.handle_search_corpora(query=query, corpus_ids=_coerce_tags(corpus_ids)))


@mcp.tool(description="Search transcript segments from Whisper outputs")
def search_transcripts(query: str, speaker: str | None = None) -> str:
    return _json(_srv.handle_search_transcripts(query=query, speaker=speaker))


@mcp.tool(description="Combined search across all sources (local, knowledge, corpora, transcripts)")
def multi_search(query: str) -> str:
    return _json(_srv.handle_multi_search(query=query))


@mcp.tool(description="Remove duplicate search results based on text similarity")
def deduplicate_results(results: list[dict], threshold: float = 0.9) -> str:
    return _json(_srv.handle_deduplicate_results(results=results, threshold=threshold))


@mcp.tool(description="Cluster search results by topic/source type")
def cluster_results(results: list[dict], num_clusters: int = 5) -> str:
    return _json(_srv.handle_cluster_results(results=results, num_clusters=num_clusters))


# --- Retrieval Telemetry (3) — Issue #5 Phase 2 ----------------------------


@mcp.tool(
    description=(
        "Score or annotate a prior kb_search result by its query_id "
        "(retrieval telemetry, issue #5). Supply user_feedback_score, notes, or "
        "both; an omitted field is left unchanged (cannot be reset to null). No "
        "effect unless LORE_HARD_NEGATIVE_MINING=true on a PostgreSQL backend."
    )
)
def log_retrieval_feedback(
    query_id: str,
    user_feedback_score: int | None = None,
    notes: str | None = None,
) -> str:
    return _json(
        _srv.handle_log_retrieval_feedback(
            query_id=query_id,
            user_feedback_score=user_feedback_score,
            notes=notes,
        )
    )


@mcp.tool(
    description=(
        "Read retrieval telemetry rows (issue #5). Selector precedence: "
        "query_id > session_id > topic > recent. Returns newest-first."
    )
)
def get_retrieval_telemetry(
    query_id: str | None = None,
    session_id: str | None = None,
    topic: str | None = None,
    limit: int = 50,
) -> str:
    return _json(
        _srv.handle_get_retrieval_telemetry(
            query_id=query_id,
            session_id=session_id,
            topic=topic,
            limit=limit,
        )
    )


@mcp.tool(
    description=(
        "Aggregate retrieval telemetry stats (issue #5): totals, feedback "
        "coverage, requery count, average score, oldest/newest timestamps. "
        "Optionally scoped by session_id and/or topic."
    )
)
def get_telemetry_stats(session_id: str | None = None, topic: str | None = None) -> str:
    return _json(_srv.handle_get_telemetry_stats(session_id=session_id, topic=topic))


# ===========================================================================
# HTTP routes: plain JSON-RPC (/mcp and /jsonrpc), health, native /stream.
# ===========================================================================

PROTOCOL_VERSION = "2025-11-25"


async def _tools_list_payload() -> list[dict]:
    """Return the tools array as plain dicts (name, description, inputSchema)."""
    tools = await mcp._list_tools()
    return [t.to_mcp_tool().model_dump(exclude_none=True) for t in tools]


async def _jsonrpc_dispatch(request: Request) -> JSONResponse:
    """Dispatch a JSON-RPC 2.0 request to the FastMCP tool engine, plain JSON.

    Handles ``initialize``, ``initialized``/``notifications/initialized``,
    ``tools/list`` and ``tools/call``. Returns bare ``application/json`` with
    NO SSE framing (Issue #9). Tool results are wrapped in the MCP
    TextContent envelope (``result.content[0].text``) for backward
    compatibility with the existing e2e client.
    """
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        )

    method = body.get("method")
    request_id = body.get("id")
    params = body.get("params", {}) or {}

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "knowledge-mcp", "version": "0.6.0"},
        }
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})

    if method in ("notifications/initialized", "initialized"):
        return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {}})

    if method == "tools/list":
        try:
            tools = await _tools_list_payload()
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"tools": tools}})
        except Exception as exc:  # noqa: BLE001
            logger.error("tools/list failed: %s", exc, exc_info=True)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": "Internal error", "data": str(exc)},
                }
            )

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if not name:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params",
                        "data": "Missing required parameter 'name'",
                    },
                }
            )
        try:
            tool_result = await mcp.call_tool(name, arguments)
            content = getattr(tool_result, "content", tool_result)
            items = []
            for item in content:
                text = getattr(item, "text", None)
                if text is not None:
                    items.append({"type": "text", "text": text})
                elif isinstance(item, dict):
                    items.append(item)
                else:
                    items.append({"type": "text", "text": str(item)})
            return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": {"content": items}})
        except Exception as exc:  # noqa: BLE001
            logger.error("tools/call %s failed: %s", name, exc, exc_info=True)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Tool execution error",
                        "data": str(exc),
                    },
                }
            )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "Method not found", "data": f"Unknown: {method}"},
        }
    )


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:  # noqa: ARG001
    """Liveness probe."""
    return JSONResponse({"status": "healthy"})


@mcp.custom_route("/jsonrpc", methods=["POST"])
async def jsonrpc_plain(request: Request) -> JSONResponse:
    """Plain JSON-RPC endpoint (Issue #9): bare application/json, no SSE."""
    return await _jsonrpc_dispatch(request)


@mcp.custom_route("/mcp", methods=["POST"])
async def mcp_plain(request: Request) -> JSONResponse:
    """Backward-compatible /mcp endpoint: plain JSON-RPC (no SSE framing).

    Preserves the prior custom wrapper's /mcp contract so existing
    ``~/.mcp.json`` clients and the e2e harness keep working without changes.
    Clients that want the native MCP Streamable HTTP transport can use
    ``/stream``.
    """
    return await _jsonrpc_dispatch(request)


def _build_cors_middleware() -> list[Middleware]:
    return [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]


def main() -> None:
    """Entry point.

    Default: stdio MCP server. With --host/--port: HTTP server exposing the
    plain-JSON /mcp and /jsonrpc routes plus the FastMCP-native Streamable
    HTTP transport at /stream.
    """
    parser = argparse.ArgumentParser(
        prog="lore-mcp",
        description="Lore MCP server on FastMCP (stdio by default, HTTP with --host/--port).",
    )
    parser.add_argument("--host", default=None, help="Bind address for HTTP mode (e.g. 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=None, help="TCP port for HTTP mode (e.g. 5556)."
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    args = parser.parse_args()

    if args.version:
        from lore import __version__

        print(f"lore-mcp (fastmcp) {__version__}")
        return

    if args.host is not None or args.port is not None:
        host = args.host or "127.0.0.1"
        port = args.port or 5556
        logger.info("Starting Lore FastMCP HTTP server on %s:%s", host, port)
        # /stream — FastMCP native Streamable-HTTP transport (MCP spec 2025-03-26).
        #
        # This endpoint is CURRENTLY UNUSED by production clients: the existing
        # ~/.mcp.json configurations point to /mcp (plain JSON-RPC, no SSE framing)
        # which is preserved for backward compatibility via the custom_route handlers
        # defined above. /jsonrpc is an alias for the same dispatcher.
        #
        # /stream is retained as the forward-compatible upgrade path. When clients
        # migrate to the native MCP Streamable-HTTP transport they can switch their
        # endpoint URL from /mcp to /stream with no other changes required.
        #
        # stateless_http=True: no session state between requests (safe for
        #   load-balanced / single-process deployments).
        # json_response=True: responses are JSON rather than SSE event-stream.
        mcp.run(
            transport="http",
            host=host,
            port=port,
            path="/stream",
            stateless_http=True,
            json_response=True,
            middleware=_build_cors_middleware(),
            show_banner=False,
        )
        return

    logger.info("Starting Lore FastMCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
