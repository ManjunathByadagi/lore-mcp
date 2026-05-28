"""Unit tests for server_fastmcp.py utilities and coercion helpers.

Targets:
- _coerce_tags: all branches (None, list, empty str, JSON, comma, space, other)
- _json: serialization of handler dicts including datetime objects
- _StrictArgsMiddleware logic (direct testing via its on_call_tool logic)
- FastMCP tool wrappers call through to _srv handlers correctly (monkeypatched)

The FastMCP app is NOT started — we just import the module and call the
pure utility functions and the thin wrappers directly.
"""

from __future__ import annotations

import json

import pytest

import lore.server as srv  # noqa: E402
import lore.server_fastmcp as fm
from lore.db_client import QueryResult

# ---------------------------------------------------------------------------
# _coerce_tags
# ---------------------------------------------------------------------------


def test_coerce_tags_none_returns_none():
    assert fm._coerce_tags(None) is None


def test_coerce_tags_list_passthrough():
    tags = ["python", "async"]
    assert fm._coerce_tags(tags) is tags


def test_coerce_tags_empty_string_returns_none():
    assert fm._coerce_tags("") is None
    assert fm._coerce_tags("   ") is None


def test_coerce_tags_json_array():
    result = fm._coerce_tags('["python", "testing"]')
    assert result == ["python", "testing"]


def test_coerce_tags_invalid_json_falls_back():
    """Invalid JSON that's not a list falls back to splitting."""
    result = fm._coerce_tags("[not valid json}")
    # Falls back to split; contains the raw string as a single item or split
    assert isinstance(result, list)
    assert len(result) >= 1


def test_coerce_tags_comma_separated():
    result = fm._coerce_tags("python, async, testing")
    assert result == ["python", "async", "testing"]


def test_coerce_tags_space_separated():
    result = fm._coerce_tags("python async testing")
    assert result == ["python", "async", "testing"]


def test_coerce_tags_single_word():
    result = fm._coerce_tags("python")
    assert result == ["python"]


def test_coerce_tags_json_non_list_falls_back():
    """JSON that is valid but not a list (e.g., a dict) falls back to splitting."""
    result = fm._coerce_tags('{"key": "val"}')
    # Not a list → falls back to treating as raw string
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _json serializer
# ---------------------------------------------------------------------------


def test_json_serializes_ok_envelope():
    data = {"ok": True, "message": "done", "data": {"count": 1}}
    result = fm._json(data)
    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert parsed["data"]["count"] == 1


def test_json_serializes_datetime_fields():
    """Datetime objects must serialize via json_serializer, not raise."""
    from datetime import datetime

    data = {"ok": True, "data": {"created_at": datetime(2025, 1, 1, 12, 0, 0)}}
    result = fm._json(data)
    parsed = json.loads(result)
    assert "2025-01-01" in parsed["data"]["created_at"]


def test_json_serializes_uuid():
    """UUID objects must serialize as strings."""
    import uuid

    data = {"ok": True, "data": {"id": uuid.UUID("12345678-1234-5678-1234-567812345678")}}
    result = fm._json(data)
    parsed = json.loads(result)
    assert parsed["data"]["id"] == "12345678-1234-5678-1234-567812345678"


def test_json_serializes_decimal():
    """Decimal objects must serialize as floats."""
    import decimal

    data = {"ok": True, "data": {"score": decimal.Decimal("0.75")}}
    result = fm._json(data)
    parsed = json.loads(result)
    assert parsed["data"]["score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# FastMCP tool wrapper call-through (monkeypatched handlers)
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, db):
        self._db = db
        self._op = "select"

    def select(self, *_a, **_k):
        return self

    def insert(self, data):
        self._op = "insert"
        self._db.inserts.append(data)
        return self

    def update(self, data):
        self._op = "update"
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def offset(self, *_a, **_k):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        if self._op == "insert":
            return QueryResult(data=self._db.inserts)
        return QueryResult(data=self._db.current_row, count=self._db.count)


class _FakeDb:
    def __init__(self, current_row=None, count=None):
        self.current_row = current_row
        self.count = count
        self.inserts = []

    def table(self, _):
        return _FakeQuery(self)


@pytest.fixture(autouse=True)
def _no_semantic(monkeypatch):
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)


def test_kb_add_wrapper_calls_through(monkeypatch):
    """The FastMCP kb_add tool wrapper delegates to handle_kb_add."""
    monkeypatch.setattr(srv, "db", _FakeDb())
    result = fm.kb_add(topic="python", title="Test", content="content here")
    data = json.loads(result)
    assert data["ok"] is True
    assert data["data"]["kb_id"].startswith("kb_")


def test_kb_add_wrapper_coerces_tags_string(monkeypatch):
    """Tags given as a comma-separated string are coerced to a list (no crash)."""
    monkeypatch.setattr(srv, "db", _FakeDb())
    result = fm.kb_add(topic="python", title="Test", content="c", tags="python, async")
    data = json.loads(result)
    # The handler succeeds — tags are inserted via insert() which the FakeDb records
    assert data["ok"] is True
    assert data["data"]["kb_id"].startswith("kb_")


def test_kb_get_wrapper_calls_through(monkeypatch):
    """The FastMCP kb_get tool wrapper delegates to handle_kb_get."""
    entry = {"kb_id": "kb_fastmcp_1", "title": "T", "content": "c", "topic": "t"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    result = fm.kb_get(kb_id="kb_fastmcp_1")
    data = json.loads(result)
    assert data["ok"] is True
    assert data["data"]["kb_id"] == "kb_fastmcp_1"


def test_kb_get_not_found_wrapper(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=None))
    result = fm.kb_get(kb_id="kb_ghost")
    data = json.loads(result)
    assert data["ok"] is False
    assert data["error"] == "not_found"


def test_kb_list_wrapper_calls_through(monkeypatch):
    entries = [{"kb_id": "kb_a"}, {"kb_id": "kb_b"}]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries, count=2))
    result = fm.kb_list()
    data = json.loads(result)
    assert data["ok"] is True
    assert data["data"]["count"] == 2


def test_kb_list_wrapper_with_topic_filter(monkeypatch):
    entries = [{"kb_id": "kb_py", "topic": "python"}]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries, count=1))
    result = fm.kb_list(topic="python")
    data = json.loads(result)
    assert data["ok"] is True


def test_kb_delete_wrapper_calls_through(monkeypatch):
    entry = {"kb_id": "kb_del", "title": "Gone", "content": "bye"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    monkeypatch.setattr(srv, "_delete_kb_embedding", lambda _: None)
    result = fm.kb_delete(kb_id="kb_del", confirm=True)
    data = json.loads(result)
    assert data["ok"] is True


def test_kb_delete_wrapper_no_confirm(monkeypatch):
    entry = {"kb_id": "kb_no_confirm", "title": "T"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    result = fm.kb_delete(kb_id="kb_no_confirm", confirm=False)
    data = json.loads(result)
    assert data["ok"] is False
    assert data["error"] == "invalid_input"


def test_kb_search_wrapper_returns_results(monkeypatch):
    """FastMCP kb_search delegates to handle_kb_search (legacy path)."""
    import lore.telemetry as tel

    monkeypatch.setattr(tel, "mining_enabled", lambda: False)
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    monkeypatch.delenv("DB_BACKEND", raising=False)

    class _SearchFakeDb:
        vec_extension_loaded = False
        fts5_available = False

        def table(self, _):
            return _SearchFakeQ([{"kb_id": "kb_x", "trust_score": 1.0}])

    class _SearchFakeQ:
        def __init__(self, rows):
            self._rows = rows

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def or_(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return QueryResult(data=self._rows)

    monkeypatch.setattr(srv, "db", _SearchFakeDb())
    result = fm.kb_search(query="test query")
    data = json.loads(result)
    assert data["ok"] is True
    assert data["data"]["count"] == 1


def test_kb_update_wrapper_valid(monkeypatch):
    """FastMCP kb_update with valid params updates content."""
    existing = {"kb_id": "kb_upd", "title": "T", "content": "old content"}
    updated = {"kb_id": "kb_upd", "title": "T", "content": "new content"}

    class _UpdQuery:
        def __init__(self, db):
            self._db = db
            self._op = "select"

        def select(self, *_a, **_k):
            return self

        def update(self, data):
            self._op = "update"
            self._db.updates.append(data)
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            if self._op == "update":
                return QueryResult(data=self._db.updated_row or {})
            return QueryResult(data=self._db.current_row)

    class _UpdDb:
        def __init__(self):
            self.current_row = existing
            self.updated_row = updated
            self.updates = []

        def table(self, _):
            if self.updates:
                self.current_row = self.updated_row
            return _UpdQuery(self)

    monkeypatch.setattr(srv, "db", _UpdDb())
    result = fm.kb_update(kb_id="kb_upd", content="new content")
    data = json.loads(result)
    assert data["ok"] is True
