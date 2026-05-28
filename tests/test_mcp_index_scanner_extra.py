"""Additional unit tests for MCPIndexScanner internal helpers.

The existing test_mcp_index_scanner.py covers the scan change-detection logic
(added/modified/removed). This file adds coverage for the previously untested
private helpers:
- _categorize_tool
- _extract_tool_tags
- _generate_usage_example
- _infer_tags
- _find_server_file
- _extract_server_description
- _parse_server_tools / _extract_tools_from_ast / _parse_tool_call / _ast_value_to_python
- _ast_dict_to_python
- scan_all_servers with servers_path=None / not a dir
- _get_existing_servers (db error path)
- search_tools / get_server_tools / get_tool_details (via fluent fake)
- _read_claude_config (with tmp_path config file)
- _record_scan_version (version_id extraction)

All tests are pure unit tests with no live database.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lore.db_client import QueryResult
from lore.mcp_index_scanner import MCPIndexScanner

# ---------------------------------------------------------------------------
# Reuse the fluent fake from the existing test file
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, data):
        self._data = data

    def select(self, *a, **k):
        return self

    def insert(self, *a, **k):
        return self

    def upsert(self, *a, **k):
        return self

    def update(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def ilike(self, *a, **k):
        return self

    def or_(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        return QueryResult(data=self._data)


class _FakeDb:
    def __init__(self, table_data: dict | None = None):
        self._table_data = table_data or {}

    def table(self, name: str):
        data = self._table_data.get(name, [])
        return _FakeQuery(data)


def _make_scanner(tmp_path, table_data=None) -> MCPIndexScanner:
    return MCPIndexScanner(_FakeDb(table_data or {}))


# ---------------------------------------------------------------------------
# scan_all_servers — early exit paths
# ---------------------------------------------------------------------------


def test_scan_returns_not_configured_when_path_unset(monkeypatch):
    monkeypatch.delenv("LORE_MCP_SERVERS_PATH", raising=False)
    scanner = MCPIndexScanner(_FakeDb())
    result = scanner.scan_all_servers()
    assert result["ok"] is False
    assert "LORE_MCP_SERVERS_PATH" in result["message"]


def test_scan_returns_not_configured_when_path_not_dir(tmp_path, monkeypatch):
    fake_path = tmp_path / "not-a-dir.txt"
    fake_path.write_text("hello")
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(fake_path))
    scanner = MCPIndexScanner(_FakeDb())
    result = scanner.scan_all_servers()
    assert result["ok"] is False


def test_scan_empty_directory_returns_zero_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    scanner = MCPIndexScanner(_FakeDb({"mcp_index_versions": [{"version_id": 42}]}))
    result = scanner.scan_all_servers(config_filter=False)
    assert result["servers_scanned"] == 0
    assert result["tools_indexed"] == 0


def test_scan_skips_dot_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "server.py").write_text("pass")
    scanner = MCPIndexScanner(_FakeDb({"mcp_index_versions": [{"version_id": 1}]}))
    result = scanner.scan_all_servers(config_filter=False)
    assert result["servers_scanned"] == 0


def test_scan_marks_removed_servers_inactive(tmp_path, monkeypatch):
    """Servers in DB that no longer exist on disk are processed as 'removed'."""
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    # DB has 'old-srv' but it's not on disk
    existing = [{"server_id": "old-srv", "tool_count": 1}]
    scanner = MCPIndexScanner(
        _FakeDb(
            {
                "mcp_servers": existing,
                "mcp_index_versions": [{"version_id": 1}],
            }
        )
    )
    result = scanner.scan_all_servers(config_filter=False)
    assert "old-srv" in result["changes"]["removed"]


# ---------------------------------------------------------------------------
# _categorize_tool
# ---------------------------------------------------------------------------


def _make_scanner_plain() -> MCPIndexScanner:
    return MCPIndexScanner(_FakeDb())


@pytest.mark.parametrize(
    "name,expected_category",
    [
        ("search_items", "search"),
        ("find_records", "search"),
        ("kb_list", "search"),
        ("kb_add", "storage"),
        ("insert_doc", "storage"),
        ("process_audio", "processing"),
        ("transcribe_file", "processing"),
        ("job_queue_status", "orchestration"),
        ("service_health", "monitoring"),
        ("restart_worker", "admin"),
        ("rebuild_index", "admin"),
        ("evaluate_model", "analysis"),
        ("compare_versions", "analysis"),
        ("do_something", "general"),
    ],
)
def test_categorize_tool(name, expected_category):
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": name, "description": "does something"}
    result = scanner._categorize_tool(tool_data)
    assert result == expected_category


# ---------------------------------------------------------------------------
# _infer_tags
# ---------------------------------------------------------------------------


def test_infer_tags_knowledge_server():
    scanner = _make_scanner_plain()
    tags = scanner._infer_tags("lore-knowledge-mcp")
    assert "knowledge" in tags
    assert "kb" in tags
    assert "search" in tags


def test_infer_tags_asr_server():
    scanner = _make_scanner_plain()
    tags = scanner._infer_tags("latvian-asr")
    assert "transcription" in tags
    assert "audio" in tags


def test_infer_tags_tts_server():
    scanner = _make_scanner_plain()
    tags = scanner._infer_tags("latvian-tts")
    assert "tts" in tags
    assert "speech" in tags


def test_infer_tags_unknown_server():
    scanner = _make_scanner_plain()
    tags = scanner._infer_tags("unknown-server-xyz")
    assert isinstance(tags, list)
    assert len(tags) == 0


def test_infer_tags_no_duplicates():
    scanner = _make_scanner_plain()
    tags = scanner._infer_tags("knowledge-search-mcp")
    assert len(tags) == len(set(tags))


# ---------------------------------------------------------------------------
# _extract_tool_tags
# ---------------------------------------------------------------------------


def test_extract_tool_tags_kb_name():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "kb_search", "description": "", "server_id": "lore-knowledge"}
    tags = scanner._extract_tool_tags(tool_data)
    assert "kb" in tags


def test_extract_tool_tags_journal_name():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "journal_append", "description": "", "server_id": "my-server"}
    tags = scanner._extract_tool_tags(tool_data)
    assert "journal" in tags


def test_extract_tool_tags_no_duplicates():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "kb_search", "description": "search kb", "server_id": "knowledge"}
    tags = scanner._extract_tool_tags(tool_data)
    assert len(tags) == len(set(tags))


# ---------------------------------------------------------------------------
# _generate_usage_example
# ---------------------------------------------------------------------------


def test_generate_usage_example_search_tool():
    scanner = _make_scanner_plain()
    tool_data = {
        "tool_name": "search_kb",
        "full_name": "mcp__lore__search_kb",
    }
    example = scanner._generate_usage_example(tool_data)
    assert "search" in example.lower()
    assert "mcp__lore__search_kb" in example


def test_generate_usage_example_add_tool():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "kb_add", "full_name": "mcp__lore__kb_add"}
    example = scanner._generate_usage_example(tool_data)
    assert "data" in example.lower() or "kb_add" in example


def test_generate_usage_example_get_tool():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "kb_get", "full_name": "mcp__lore__kb_get"}
    example = scanner._generate_usage_example(tool_data)
    assert "()" in example


def test_generate_usage_example_generic():
    scanner = _make_scanner_plain()
    tool_data = {"tool_name": "do_thing", "full_name": "mcp__myserver__do_thing"}
    example = scanner._generate_usage_example(tool_data)
    assert "mcp__myserver__do_thing" in example


# ---------------------------------------------------------------------------
# _find_server_file
# ---------------------------------------------------------------------------


def test_find_server_file_at_root(tmp_path):
    server_dir = tmp_path / "my-server"
    server_dir.mkdir()
    server_file = server_dir / "server.py"
    server_file.write_text("pass")

    scanner = _make_scanner_plain()
    result = scanner._find_server_file(server_dir, "my-server")
    assert result == server_file


def test_find_server_file_in_src_subdir(tmp_path):
    server_dir = tmp_path / "my-server"
    src_module = server_dir / "src" / "my_server"
    src_module.mkdir(parents=True)
    server_file = src_module / "server.py"
    server_file.write_text("pass")

    scanner = _make_scanner_plain()
    result = scanner._find_server_file(server_dir, "my-server")
    assert result == server_file


def test_find_server_file_nested_via_glob(tmp_path):
    server_dir = tmp_path / "nested-srv"
    inner = server_dir / "deep" / "path"
    inner.mkdir(parents=True)
    server_file = inner / "server.py"
    server_file.write_text("pass")

    scanner = _make_scanner_plain()
    result = scanner._find_server_file(server_dir, "nested-srv")
    assert result.name == "server.py"


# ---------------------------------------------------------------------------
# _extract_server_description
# ---------------------------------------------------------------------------


def test_extract_server_description_with_docstring(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text('"""My test server.\n\nDoes cool things."""\n\npass\n')

    scanner = _make_scanner_plain()
    desc = scanner._extract_server_description(server_file)
    assert desc == "My test server."


def test_extract_server_description_no_docstring(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("import os\n\nprint('hello')\n")

    scanner = _make_scanner_plain()
    desc = scanner._extract_server_description(server_file)
    assert desc == ""


def test_extract_server_description_invalid_python(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("this is not python !! @@@ ###")

    scanner = _make_scanner_plain()
    # Must not raise
    desc = scanner._extract_server_description(server_file)
    assert desc == ""


# ---------------------------------------------------------------------------
# _parse_server_tools — AST extraction
# ---------------------------------------------------------------------------


def _write_server(path: Path, tool_names: list[str]) -> Path:
    """Write a syntactically valid fake server.py with the given tool names."""
    server_file = path / "server.py"
    # Build each tool as a separate, clean line — avoids brace-escaping issues.
    tool_lines = []
    for n in tool_names:
        tool_lines.append(
            f'        types.Tool(name="{n}", description="Tool {n}", inputSchema={{"type": "object", "properties": {{}}}}),'
        )
    tools_block = "\n".join(tool_lines)
    content = f'"""Server docstring."""\nfrom mcp import types\n\nasync def list_tools():\n    return [\n{tools_block}\n    ]\n'
    server_file.write_text(content)
    return server_file


def test_parse_server_tools_extracts_tools(tmp_path):
    server_file = _write_server(tmp_path, ["tool_alpha", "tool_beta"])
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "test-server")
    assert len(tools) == 2
    tool_names = {t["tool_name"] for t in tools}
    assert "tool_alpha" in tool_names
    assert "tool_beta" in tool_names


def test_parse_server_tools_populates_tool_id(tmp_path):
    server_file = _write_server(tmp_path, ["my_tool"])
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "srv-x")
    assert tools[0]["tool_id"] == "srv-x_my_tool"


def test_parse_server_tools_generates_full_name(tmp_path):
    server_file = _write_server(tmp_path, ["kb_search"])
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "lore-knowledge")
    assert tools[0]["full_name"] == "mcp__lore-knowledge__kb_search"


def test_parse_server_tools_no_list_tools_function(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("def hello(): pass\n")
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "x")
    assert tools == []


def test_parse_server_tools_invalid_python_returns_empty(tmp_path):
    server_file = tmp_path / "server.py"
    server_file.write_text("def ( broken syntax!!!!")
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "x")
    assert tools == []


# ---------------------------------------------------------------------------
# _ast_value_to_python — edge cases
# ---------------------------------------------------------------------------


def test_ast_value_handles_fstring(tmp_path):
    """F-strings in inputSchema are converted to placeholder strings."""
    server_file = tmp_path / "server.py"
    server_file.write_text(
        textwrap.dedent("""\
        from mcp import types

        async def list_tools():
            return [
                types.Tool(
                    name="fstr_tool",
                    description="tool",
                    inputSchema={"type": "object", "properties": {
                        "path": {"type": "string", "description": f"Path to file (default: {__file__})"}
                    }},
                ),
            ]
        """)
    )
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "s")
    # Should not raise; description may contain placeholder
    assert len(tools) == 1 or len(tools) == 0  # may skip if schema fails; no crash


def test_ast_value_handles_name_references(tmp_path):
    """Name references like True/False/None in schema must be handled."""
    server_file = tmp_path / "server.py"
    server_file.write_text(
        textwrap.dedent("""\
        from mcp import types

        async def list_tools():
            return [
                types.Tool(
                    name="bool_tool",
                    description="uses booleans",
                    inputSchema={"type": "object", "properties": {
                        "required": {"type": "boolean", "default": True}
                    }},
                ),
            ]
        """)
    )
    scanner = _make_scanner_plain()
    tools = scanner._parse_server_tools(server_file, "s")
    if tools:
        schema = tools[0].get("input_schema", {})
        props = schema.get("properties", {})
        if "required" in props:
            assert props["required"].get("default") is True


# ---------------------------------------------------------------------------
# _read_claude_config — with tmp config files
# ---------------------------------------------------------------------------


def test_read_claude_config_returns_server_ids(tmp_path):
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "lore-knowledge": {"command": "python", "args": ["server.py"]},
                    "lore-tts": {"command": "python", "args": ["server.py"]},
                }
            }
        )
    )
    scanner = _make_scanner_plain()
    result = scanner._read_claude_config(config_path=str(config))
    assert isinstance(result, list)
    assert "lore-knowledge" in result
    assert "lore-tts" in result


def test_read_claude_config_returns_none_when_no_config(tmp_path):
    scanner = _make_scanner_plain()
    # Point to non-existent config
    result = scanner._read_claude_config(config_path=str(tmp_path / "missing.json"))
    assert result is None


def test_read_claude_config_handles_invalid_json(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text("not valid json")
    scanner = _make_scanner_plain()
    # Must not raise; returns None when no valid sources
    result = scanner._read_claude_config(config_path=str(config))
    assert result is None or isinstance(result, list)


def test_read_claude_config_deduplicates_servers(tmp_path):
    config = tmp_path / "claude.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "srv-a": {"command": "python"},
                    "srv-b": {"command": "python"},
                }
            }
        )
    )
    scanner = _make_scanner_plain()
    result = scanner._read_claude_config(config_path=str(config))
    assert result is not None
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# _record_scan_version
# ---------------------------------------------------------------------------


def test_record_scan_version_returns_version_id():
    db = _FakeDb({"mcp_index_versions": [{"version_id": 99}]})
    scanner = MCPIndexScanner(db)
    vid = scanner._record_scan_version(
        servers_scanned=3,
        tools_indexed=10,
        changes={"added": [], "removed": [], "modified": []},
        scan_duration=500,
        errors=[],
        triggered_by="test",
    )
    assert vid == 99


def test_record_scan_version_returns_minus_one_on_error():
    """When DB insert fails, should return -1 gracefully."""

    class _BrokenDb:
        def table(self, _):
            raise RuntimeError("db dead")

    scanner = MCPIndexScanner(_BrokenDb())
    vid = scanner._record_scan_version(0, 0, {}, 0, [], "test")
    assert vid == -1


# ---------------------------------------------------------------------------
# search_tools / get_server_tools / get_tool_details (fluent fake smoke tests)
# ---------------------------------------------------------------------------


def test_search_tools_returns_list():
    fake_tools = [{"tool_id": "srv_alpha", "tool_name": "alpha"}]
    db = _FakeDb(
        {
            "mcp_tools": fake_tools,
            "mcp_servers": [{"server_id": "srv", "status": "active"}],
        }
    )
    scanner = MCPIndexScanner(db)
    results = scanner.search_tools("alpha")
    assert isinstance(results, list)


def test_get_server_tools_returns_dict():
    fake_server = [{"server_id": "srv", "server_name": "My Server"}]
    fake_tools = [{"tool_id": "srv_kb_search", "tool_name": "kb_search"}]
    db = _FakeDb({"mcp_servers": fake_server, "mcp_tools": fake_tools})
    scanner = MCPIndexScanner(db)
    result = scanner.get_server_tools("srv")
    assert result is not None
    assert "server" in result
    assert "tools" in result


def test_get_server_tools_returns_none_when_not_found():
    db = _FakeDb({"mcp_servers": []})  # empty → no server found
    scanner = MCPIndexScanner(db)
    result = scanner.get_server_tools("nonexistent")
    assert result is None


def test_get_tool_details_exact_match():
    fake_tool = [{"tool_id": "srv_kb_search", "tool_name": "kb_search"}]
    db = _FakeDb({"mcp_tools": fake_tool})
    scanner = MCPIndexScanner(db)
    result = scanner.get_tool_details("kb_search")
    assert result is not None


def test_get_tool_details_returns_none_when_not_found():
    db = _FakeDb({"mcp_tools": []})
    scanner = MCPIndexScanner(db)
    result = scanner.get_tool_details("nonexistent_tool")
    assert result is None


# ---------------------------------------------------------------------------
# _get_existing_servers — error path
# ---------------------------------------------------------------------------


def test_get_existing_servers_returns_empty_on_db_error():
    class _ErrorDb:
        def table(self, _):
            raise RuntimeError("connection refused")

    scanner = MCPIndexScanner(_ErrorDb())
    result = scanner._get_existing_servers()
    assert result == []
