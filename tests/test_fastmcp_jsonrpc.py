"""Tests for server_fastmcp._jsonrpc_dispatch and HTTP routes.

Exercises:
- /health: healthy (db set) and unhealthy (db=None)
- /jsonrpc and /mcp: initialize, initialized, notifications/initialized,
  tools/list (success + exception), tools/call (success + missing name +
  call exception + dict item + str item), unknown method, parse error
- _build_http_middleware: returns a non-empty list

We use build_http_app() to get a real Starlette app (the same stack prod uses)
and starlette.testclient.TestClient for in-process HTTP.

The FastMCP `mcp` object is module-level; we monkeypatch its `call_tool` and
`_list_tools` to avoid touching a real DB.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import lore.server as _srv
import lore.server_fastmcp as fm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> TestClient:
    """Build a TestClient from build_http_app(). Same middleware stack as prod."""
    app = fm.build_http_app()
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_healthy_when_db_is_set(monkeypatch, tmp_path):
    """health returns 200/healthy when db is initialised."""
    from lore.db_client import SqliteClient

    db = SqliteClient(db_path=str(tmp_path / "test.db"))
    monkeypatch.setattr(_srv, "db", db)

    client = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    db.close()


def test_health_unhealthy_when_db_is_none(monkeypatch):
    """health returns 503 when db is None (not yet initialised)."""
    monkeypatch.setattr(_srv, "db", None)

    client = _make_client()
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# /jsonrpc — initialize
# ---------------------------------------------------------------------------


def test_jsonrpc_initialize_returns_protocol_version():
    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 1
    assert data["result"]["protocolVersion"] == fm.PROTOCOL_VERSION
    assert "serverInfo" in data["result"]


# ---------------------------------------------------------------------------
# /jsonrpc — initialized / notifications/initialized
# ---------------------------------------------------------------------------


def test_jsonrpc_initialized_returns_empty_result():
    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 2, "method": "initialized", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == {}


def test_jsonrpc_notifications_initialized_returns_empty_result():
    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "notifications/initialized",
        "params": {},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"] == {}


# ---------------------------------------------------------------------------
# /jsonrpc — tools/list
# ---------------------------------------------------------------------------


def test_jsonrpc_tools_list_returns_tool_array(monkeypatch):
    """tools/list should return the list from _tools_list_payload."""
    fake_tool = {"name": "kb_search", "description": "Search KB", "inputSchema": {}}

    async def _fake_tools_list():
        return [fake_tool]

    monkeypatch.setattr(fm, "_tools_list_payload", _fake_tools_list)

    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data["result"]
    assert any(t["name"] == "kb_search" for t in data["result"]["tools"])


def test_jsonrpc_tools_list_error_returns_32603(monkeypatch):
    """If _tools_list_payload raises, return -32603 internal error."""

    async def _boom():
        raise RuntimeError("list blew up")

    monkeypatch.setattr(fm, "_tools_list_payload", _boom)

    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32603


# ---------------------------------------------------------------------------
# /jsonrpc — tools/call
# ---------------------------------------------------------------------------


def test_jsonrpc_tools_call_missing_name_returns_32602():
    """tools/call without name returns -32602 Invalid params."""
    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"arguments": {}},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32602


def test_jsonrpc_tools_call_success_text_content(monkeypatch):
    """tools/call with TextContent items serialises each to {type, text}."""
    from mcp.types import TextContent

    class _FakeResult:
        content = [TextContent(type="text", text='{"ok":true}')]

    async def _fake_call_tool(name, args):
        return _FakeResult()

    monkeypatch.setattr(fm.mcp, "call_tool", _fake_call_tool)

    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "kb_get", "arguments": {"kb_id": "kb_1"}},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data["result"]
    assert data["result"]["content"][0]["type"] == "text"


def test_jsonrpc_tools_call_success_dict_item(monkeypatch):
    """tools/call with dict items in content passes them through unchanged."""

    class _FakeResult:
        content = [{"type": "text", "text": "hello from dict"}]

    async def _fake_call_tool(name, args):
        return _FakeResult()

    monkeypatch.setattr(fm.mcp, "call_tool", _fake_call_tool)

    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "some_tool", "arguments": {}},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["content"][0]["text"] == "hello from dict"


def test_jsonrpc_tools_call_success_other_item(monkeypatch):
    """Items that are neither TextContent nor dict are str()-ed."""

    class _FakeResult:
        content = [42]  # raw int — should be str()-ed

    async def _fake_call_tool(name, args):
        return _FakeResult()

    monkeypatch.setattr(fm.mcp, "call_tool", _fake_call_tool)

    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "some_tool", "arguments": {}},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["content"][0]["text"] == "42"


def test_jsonrpc_tools_call_exception_returns_32603(monkeypatch):
    """When mcp.call_tool raises, the response carries -32603."""

    async def _fail(name, args):
        raise ValueError("no such tool")

    monkeypatch.setattr(fm.mcp, "call_tool", _fail)

    client = _make_client()
    payload = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "bad_tool", "arguments": {}},
    }
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32603
    assert "no such tool" in data["error"]["data"]


# ---------------------------------------------------------------------------
# /jsonrpc — unknown method
# ---------------------------------------------------------------------------


def test_jsonrpc_unknown_method_returns_32601():
    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 11, "method": "ping", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# /mcp mirrors /jsonrpc (backward compat)
# ---------------------------------------------------------------------------


def test_mcp_endpoint_initialize_works():
    """The /mcp endpoint delegates to the same _jsonrpc_dispatch."""
    client = _make_client()
    payload = {"jsonrpc": "2.0", "id": 20, "method": "initialize", "params": {}}
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["protocolVersion"] == fm.PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# Parse error (malformed JSON body)
# ---------------------------------------------------------------------------


def test_jsonrpc_parse_error_on_invalid_json():
    """Sending malformed JSON returns -32700 Parse error."""
    client = _make_client()
    resp = client.post(
        "/jsonrpc",
        content=b"not json at all }{",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["error"]["code"] == -32700


# ---------------------------------------------------------------------------
# _build_http_middleware
# ---------------------------------------------------------------------------


def test_build_http_middleware_returns_non_empty_list():
    """_build_http_middleware must return at least the CORS + auth middlewares."""
    middleware = fm._build_http_middleware()
    assert isinstance(middleware, list)
    assert len(middleware) >= 2


# ---------------------------------------------------------------------------
# _semantic_enabled helper
# ---------------------------------------------------------------------------


def test_semantic_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    assert fm._semantic_enabled() is False


def test_semantic_enabled_true_when_env_set(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    assert fm._semantic_enabled() is True
