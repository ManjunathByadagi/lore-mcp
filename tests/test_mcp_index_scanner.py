"""Unit tests for MCPIndexScanner change detection (P1-7).

Pure unit tests — no live database. A small fluent fake stands in for the
db_client query builder so ``scan_all_servers`` can be exercised in isolation.
The scanner reads prior server rows from ``mcp_servers`` (each carrying a
``tool_count``) and writes via upsert/insert; the fake accepts the writes and
serves the configured prior rows on reads.

These tests cover the previously-unimplemented ``changes["modified"]`` list
(was always ``[]`` with a bare TODO): a server present in both the prior and
current scan whose tool_count changed must be reported as modified.
"""

from __future__ import annotations

import textwrap

from lore.mcp_index_scanner import MCPIndexScanner

# ---------------------------------------------------------------------------
# Fluent fake db: db.table("mcp_servers").select("*").execute() returns the
# configured prior rows; upsert/insert/update/eq are accepted as no-ops.
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, data: list[dict]):
        self._data = data

    def select(self, *_a, **_k):
        return self

    def insert(self, *_a, **_k):
        return self

    def upsert(self, *_a, **_k):
        return self

    def update(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def execute(self):
        from lore.db_client import QueryResult

        return QueryResult(data=self._data)


class _FakeScannerDb:
    """Serves prior ``mcp_servers`` rows; absorbs all writes.

    ``existing`` is the list of prior server rows (each a dict with at least
    ``server_id`` and ``tool_count``). The ``mcp_index_versions`` insert returns
    a row so ``_record_scan_version`` yields a version id.
    """

    def __init__(self, existing: list[dict]):
        self._existing = existing

    def table(self, name: str):
        if name == "mcp_servers":
            return _FakeQuery(self._existing)
        if name == "mcp_index_versions":
            return _FakeQuery([{"version_id": 1}])
        return _FakeQuery([])


def _make_server_dir(root, server_id: str, tool_names: list[str]) -> None:
    """Create a fake MCP server dir with a ``server.py`` exposing list_tools."""
    server_dir = root / server_id
    server_dir.mkdir(parents=True, exist_ok=True)
    tool_lines = "\n".join(
        f'        types.Tool(name="{n}", description="d", '
        f'inputSchema={{"type": "object", "properties": {{}}}}),'
        for n in tool_names
    )
    source = textwrap.dedent(
        f'''\
        """Fake {server_id} server docstring."""
        from mcp import types


        async def list_tools():
            return [
        {tool_lines}
            ]
        '''
    )
    (server_dir / "server.py").write_text(source)


def test_scan_reports_added_for_new_server(tmp_path, monkeypatch):
    """A server with no prior row is reported under changes['added']."""
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    _make_server_dir(tmp_path, "srv-a", ["alpha", "beta"])

    scanner = MCPIndexScanner(_FakeScannerDb(existing=[]))
    result = scanner.scan_all_servers(config_filter=False)

    assert result["servers_scanned"] == 1
    assert result["tools_indexed"] == 2
    assert result["changes"]["added"] == ["srv-a"]
    assert result["changes"]["modified"] == []
    assert result["changes"]["removed"] == []


def test_scan_reports_modified_when_tool_count_changes(tmp_path, monkeypatch):
    """A server present in both scans with a changed tool_count is 'modified'."""
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    # Current scan finds two tools; prior row recorded only one -> modified.
    _make_server_dir(tmp_path, "srv-a", ["alpha", "beta"])

    scanner = MCPIndexScanner(_FakeScannerDb(existing=[{"server_id": "srv-a", "tool_count": 1}]))
    result = scanner.scan_all_servers(config_filter=False)

    assert result["changes"]["modified"] == ["srv-a"]
    assert result["changes"]["added"] == []  # not new — it existed before
    assert result["changes"]["removed"] == []


def test_scan_unchanged_tool_count_is_not_modified(tmp_path, monkeypatch):
    """A server whose tool_count matches the prior scan is NOT reported."""
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    _make_server_dir(tmp_path, "srv-a", ["alpha", "beta"])

    scanner = MCPIndexScanner(_FakeScannerDb(existing=[{"server_id": "srv-a", "tool_count": 2}]))
    result = scanner.scan_all_servers(config_filter=False)

    assert result["changes"]["modified"] == []
    assert result["changes"]["added"] == []


def test_scan_unknown_prior_tool_count_is_not_a_false_positive(tmp_path, monkeypatch):
    """A prior row with no tool_count must not be guessed as modified."""
    monkeypatch.setenv("LORE_MCP_SERVERS_PATH", str(tmp_path))
    _make_server_dir(tmp_path, "srv-a", ["alpha"])

    # Prior row predates tool_count recording (None) — we must not fabricate a diff.
    scanner = MCPIndexScanner(_FakeScannerDb(existing=[{"server_id": "srv-a", "tool_count": None}]))
    result = scanner.scan_all_servers(config_filter=False)

    assert result["changes"]["modified"] == []
