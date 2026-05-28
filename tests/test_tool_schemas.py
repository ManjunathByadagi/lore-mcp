"""Syrupy snapshot tests for all FastMCP tool schemas.

Each registered tool gets its own parametrized snapshot so a single schema
change (name, description, inputSchema, outputSchema) produces a focused diff
pointing at exactly that tool rather than a bulk diff across all 38.

The schema captured here is the public JSON Schema every MCP client sees via
``tools/list`` — i.e., the output of ``tool.to_mcp_tool().model_dump()``.
Internal FastMCP metadata (the ``meta`` key) is stripped before snapshotting.

Usage:
    # Generate/update baseline snapshots:
    pytest tests/test_tool_schemas.py --snapshot-update

    # Verify no drift (CI):
    pytest tests/test_tool_schemas.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest
from syrupy.assertion import SnapshotAssertion

# ---------------------------------------------------------------------------
# Ensure the server module is importable without a real DB.  We only need the
# FastMCP registry; the lifespan (DB init) is never triggered during schema
# inspection.
# ---------------------------------------------------------------------------
os.environ.setdefault("DB_BACKEND", "sqlite")


def _get_all_tools() -> list[dict[str, Any]]:
    """Return all registered tool schemas, sorted by name, meta stripped.

    Runs the async ``mcp._list_tools()`` in a new event loop so this helper
    can be called from a synchronous context (pytest fixtures / parametrize).
    The result is deterministic: tools are sorted by name and the schema dict
    is JSON-round-tripped with ``sort_keys=True`` to guarantee stable ordering
    regardless of Python dict insertion order.
    """
    # Import after env var is set so the module-level sentinels bind correctly.
    import lore.server_fastmcp as fm

    tools = asyncio.run(fm.mcp._list_tools())

    result = []
    for tool in sorted(tools, key=lambda t: t.name):
        mcp_tool = tool.to_mcp_tool()
        raw: dict[str, Any] = mcp_tool.model_dump(exclude_none=True)
        # Strip internal FastMCP metadata — clients never see this.
        raw.pop("meta", None)
        # Normalize to a stable JSON string, then parse back to a plain dict.
        # This removes any Python-specific ordering from nested dicts.
        normalized: dict[str, Any] = json.loads(json.dumps(raw, sort_keys=True))
        result.append({"name": tool.name, "schema": normalized})

    return result


# ---------------------------------------------------------------------------
# Build the parametrize list once at collection time.
# ---------------------------------------------------------------------------
_ALL_TOOLS = _get_all_tools()
_TOOL_IDS = [entry["name"] for entry in _ALL_TOOLS]


@pytest.mark.parametrize("tool_entry", _ALL_TOOLS, ids=_TOOL_IDS)
def test_tool_schema_snapshot(tool_entry: dict[str, Any], snapshot: SnapshotAssertion) -> None:
    """Each tool's public schema must match its stored snapshot.

    If a tool's name, description, inputSchema, or outputSchema changes, this
    test will fail with a diff showing exactly what changed.  Run with
    ``--snapshot-update`` to accept intentional changes.
    """
    assert tool_entry["schema"] == snapshot
