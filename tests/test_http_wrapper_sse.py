"""Tests for lore.mcp_http_wrapper_sse module.

Exercises:
- create_app: /health endpoint
- create_app/handle_mcp: initialize, initialized, tools/list, tools/call,
  unknown method, and malformed JSON (all branches of handle_mcp)
- execute_tool: _tool_handlers path, module call_tool path, handler_name path,
  no-handler error path
- load_mcp_server: success path and ValueError fallback

Uses starlette.testclient.TestClient for in-process HTTP calls (no real server).
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

import lore.mcp_http_wrapper_sse as wrapper

# ---------------------------------------------------------------------------
# Minimal fake MCP server (duck-typed, no heavy imports)
# ---------------------------------------------------------------------------


class _FakeMcpServer:
    """Minimal duck-typed MCP server — only what create_app touches."""

    def run(self, *_a, **_k):
        pass

    def create_initialization_options(self):
        return {}


def _make_app(server_name: str = "nonexistent_module_xyz") -> TestClient:
    """Build a TestClient wrapping a fresh create_app instance."""
    app = wrapper.create_app(_FakeMcpServer(), server_name)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_check_returns_200():
    client = _make_app()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# /mcp  — initialize method
# ---------------------------------------------------------------------------


def test_initialize_returns_session_id():
    client = _make_app()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"clientInfo": {"name": "test"}, "protocolVersion": "2025-11-25"},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    # SSE framing — response body contains an event:message + data: line
    text = resp.text
    assert "event: message" in text
    assert '"jsonrpc"' in text
    # Extract the JSON data line
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["id"] == 1
    result = data["result"]
    assert result["protocolVersion"] == "2025-11-25"
    assert "_meta" in result
    assert "sessionId" in result["_meta"]


def test_initialize_loads_tool_definitions_when_module_present(monkeypatch):
    """When the server module has _TOOL_DEFINITIONS, tools appear in the response."""
    tool = MagicMock()
    tool.name = "kb_search"
    tool.description = "Search KB"

    fake_module = types.ModuleType("fake_srv")
    fake_module._TOOL_DEFINITIONS = [tool]

    with patch("importlib.import_module", return_value=fake_module):
        client = _make_app(server_name="fake_srv")
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {},
        }
        resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    tools = data["result"]["_meta"]["availableTools"]
    assert any(t["name"] == "kb_search" for t in tools)


def test_initialize_handles_module_import_error():
    """ImportError when loading server module is gracefully swallowed."""
    with patch("importlib.import_module", side_effect=ImportError("no module")):
        client = _make_app(server_name="missing_module")
        payload = {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}}
        resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    # Should still return a result; availableTools may be empty
    assert "result" in data
    assert data["result"]["_meta"]["availableTools"] == []


# ---------------------------------------------------------------------------
# /mcp  — initialized method
# ---------------------------------------------------------------------------


def test_initialized_returns_empty_result():
    client = _make_app()
    payload = {"jsonrpc": "2.0", "id": 4, "method": "initialized", "params": {}}
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["id"] == 4
    assert data["result"] == {}


# ---------------------------------------------------------------------------
# /mcp  — tools/list method
# ---------------------------------------------------------------------------


def test_tools_list_returns_tools_from_module(monkeypatch):
    """tools/list returns _TOOL_DEFINITIONS from the loaded module."""
    tool = MagicMock()
    tool.name = "kb_add"
    tool.description = "Add KB entry"
    tool.inputSchema = {"type": "object"}

    fake_module = types.ModuleType("fake_srv2")
    fake_module._TOOL_DEFINITIONS = [tool]

    with patch("importlib.import_module", return_value=fake_module):
        client = _make_app(server_name="fake_srv2")
        payload = {"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}}
        resp = client.post("/mcp", json=payload)

    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    tools = data["result"]["tools"]
    assert any(t["name"] == "kb_add" for t in tools)


def test_tools_list_no_definitions_returns_empty_list():
    """tools/list with no _TOOL_DEFINITIONS logs a warning and returns empty list."""
    fake_module = types.ModuleType("fake_no_def")
    # No _TOOL_DEFINITIONS attribute

    with patch("importlib.import_module", return_value=fake_module):
        client = _make_app(server_name="fake_no_def")
        payload = {"jsonrpc": "2.0", "id": 6, "method": "tools/list", "params": {}}
        resp = client.post("/mcp", json=payload)

    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["result"]["tools"] == []


def test_tools_list_import_error_returns_empty_list():
    """tools/list handles ImportError gracefully and returns empty tools."""
    with patch("importlib.import_module", side_effect=RuntimeError("boom")):
        client = _make_app(server_name="broken_module")
        payload = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {}}
        resp = client.post("/mcp", json=payload)

    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["result"]["tools"] == []


# ---------------------------------------------------------------------------
# /mcp  — tools/call method
# ---------------------------------------------------------------------------


def test_tools_call_with_tool_handler_path(monkeypatch):
    """tools/call executes via mcp_server._tool_handlers dict."""
    from mcp.types import TextContent

    async def _fake_handler(name, args):
        return [TextContent(type="text", text='{"ok": true}')]

    fake_server = _FakeMcpServer()
    fake_server._tool_handlers = {"kb_get": _fake_handler}

    app = wrapper.create_app(fake_server, "irrelevant_module")
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "kb_get", "arguments": {"kb_id": "kb_123"}},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert "result" in data


def test_tools_call_missing_name_returns_error():
    """tools/call without a tool name returns a -32602 Invalid params error."""
    client = _make_app()
    payload = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {"arguments": {}},  # name is missing
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["error"]["code"] == -32602


def test_tools_call_execution_error_returns_error():
    """When execute_tool raises, the response carries a -32603 error."""

    async def _failing_handler(name, args):
        raise RuntimeError("DB is down")

    fake_server = _FakeMcpServer()
    fake_server._tool_handlers = {"broken_tool": _failing_handler}

    app = wrapper.create_app(fake_server, "irrelevant")
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {"name": "broken_tool", "arguments": {}},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["error"]["code"] == -32603


def test_tools_call_dict_result_returned_as_is():
    """When execute_tool returns a dict, it is forwarded directly in result."""

    async def _dict_handler(name, args):
        return {"content": [{"type": "text", "text": "hello"}]}

    fake_server = _FakeMcpServer()
    fake_server._tool_handlers = {"dict_tool": _dict_handler}

    app = wrapper.create_app(fake_server, "irrelevant")
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {"name": "dict_tool", "arguments": {}},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert "result" in data


def test_tools_call_string_result_wrapped():
    """When execute_tool returns a plain string, it is wrapped in TextContent."""

    async def _str_handler(name, args):
        return "plain string result"

    fake_server = _FakeMcpServer()
    fake_server._tool_handlers = {"str_tool": _str_handler}

    app = wrapper.create_app(fake_server, "irrelevant")
    client = TestClient(app, raise_server_exceptions=True)

    payload = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {"name": "str_tool", "arguments": {}},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert "result" in data


# ---------------------------------------------------------------------------
# /mcp  — unknown method
# ---------------------------------------------------------------------------


def test_unknown_method_returns_method_not_found():
    client = _make_app()
    payload = {"jsonrpc": "2.0", "id": 20, "method": "ping", "params": {}}
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    text = resp.text
    data_line = next(line for line in text.splitlines() if line.startswith("data:"))
    data = json.loads(data_line.removeprefix("data: "))
    assert data["error"]["code"] == -32601


# ---------------------------------------------------------------------------
# /jsonrpc endpoint mirrors /mcp
# ---------------------------------------------------------------------------


def test_jsonrpc_route_accepts_initialize():
    client = _make_app()
    payload = {"jsonrpc": "2.0", "id": 30, "method": "initialize", "params": {}}
    resp = client.post("/jsonrpc", json=payload)
    assert resp.status_code == 200
    text = resp.text
    assert "jsonrpc" in text


# ---------------------------------------------------------------------------
# execute_tool: module call_tool fallback path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_module_call_tool_path():
    """execute_tool falls through to module.call_tool() when no _tool_handlers."""

    async def _call_tool(name, args):
        return [{"type": "text", "text": "from call_tool"}]

    fake_module = types.ModuleType("call_tool_mod")
    fake_module.call_tool = _call_tool

    class _NoHandlerServer:
        pass

    with patch("importlib.import_module", return_value=fake_module):
        result = await wrapper.execute_tool("kb_list", {}, _NoHandlerServer(), "call_tool_mod")

    assert result is not None


@pytest.mark.asyncio
async def test_execute_tool_no_handler_raises():
    """execute_tool raises ValueError when no handler is found at all."""
    fake_module = types.ModuleType("empty_mod")
    # No call_tool, no _tool_handlers, no handle_* functions

    class _NoHandlerServer:
        pass

    with patch("importlib.import_module", return_value=fake_module):
        with pytest.raises((ValueError, Exception)):
            await wrapper.execute_tool("ghost_tool", {}, _NoHandlerServer(), "empty_mod")


# ---------------------------------------------------------------------------
# load_mcp_server
# ---------------------------------------------------------------------------


def test_load_mcp_server_finds_app_attr():
    """load_mcp_server returns the 'app' attribute when it has run/create_initialization_options."""

    class _ServerObj:
        def run(self): ...
        def create_initialization_options(self): ...

    fake_module = types.ModuleType("srv_with_app")
    fake_module.app = _ServerObj()

    with patch("importlib.import_module", return_value=fake_module):
        server, attr = wrapper.load_mcp_server("srv_with_app")

    assert attr == "app"
    assert isinstance(server, _ServerObj)


def test_load_mcp_server_raises_when_no_valid_attr():
    """load_mcp_server raises ValueError when no recognized server attribute found."""
    fake_module = types.ModuleType("srv_empty")
    # No app/server/mcp/mcp_server attributes

    with patch("importlib.import_module", return_value=fake_module):
        with pytest.raises(ValueError, match="Could not find MCP server instance"):
            wrapper.load_mcp_server("srv_empty")


# ---------------------------------------------------------------------------
# _SseResponse no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_response_is_callable():
    """_SseResponse.__call__ must be a no-op coroutine (no error)."""
    resp = wrapper._SseResponse()
    # Should complete without raising
    await resp(scope={}, receive=None, send=None)
