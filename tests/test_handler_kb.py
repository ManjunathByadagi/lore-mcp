"""Unit tests for KB handler functions: kb_add, kb_get, kb_list, kb_delete.

Uses the same fluent-fake pattern as test_server_handlers.py: a small
_FakeDb stand-in replaces the global `srv.db` so handlers execute without
a live database.  Real assertion-level checks on the ResponseEnvelope
(ok, error, data keys).
"""

from __future__ import annotations

import pytest

import lore.server as srv
from lore.db_client import QueryResult

# ---------------------------------------------------------------------------
# Fake query builder (extended from test_server_handlers.py pattern)
# ---------------------------------------------------------------------------


class _FakeQuery:
    """Chainable fake that records inserts, selects, deletes, and updates."""

    def __init__(self, db: _FakeDb):
        self._db = db
        self._op = "select"
        self._filters: dict = {}

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, data):
        self._op = "insert"
        self._db.inserts.append(data)
        return self

    def update(self, data):
        self._op = "update"
        self._db.updates.append(data)
        return self

    def delete(self):
        self._op = "delete"
        self._db.deleted = True
        return self

    def eq(self, col, val):
        self._filters[col] = val
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
            return QueryResult(data=[self._db.inserts[-1]] if self._db.inserts else [])
        if self._op == "delete":
            if self._db.delete_raises:
                raise self._db.delete_raises
            return QueryResult(data=self._db.deleted_row or [])
        if self._op == "update":
            return QueryResult(data=self._db.current_row or {})
        # select / maybe_single
        return QueryResult(data=self._db.current_row, count=self._db.count)


class _FakeDb:
    def __init__(
        self,
        current_row=None,
        count: int | None = None,
        deleted_row=None,
        delete_raises=None,
    ):
        self.current_row = current_row  # returned by every SELECT
        self.count = count
        self.inserts: list = []
        self.updates: list = []
        self.deleted = False
        self.deleted_row = deleted_row or []
        self.delete_raises = delete_raises

    def table(self, _name):
        return _FakeQuery(self)


# ---------------------------------------------------------------------------
# Shared fixture: disable semantic write path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_semantic(monkeypatch):
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)


# ---------------------------------------------------------------------------
# handle_kb_add
# ---------------------------------------------------------------------------


def test_kb_add_success(monkeypatch):
    """kb_add with valid params returns ok=True and a kb_id."""
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="python", title="Async guide", content="content here")
    assert resp["ok"] is True
    assert resp["data"]["kb_id"].startswith("kb_")
    assert resp["data"]["topic"] == "python"


def test_kb_add_stores_author_and_source_type(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(
        topic="testing",
        title="QA notes",
        content="test content",
        author="Alice",
        source_type="manual",
    )
    assert resp["ok"] is True
    assert resp["data"]["author"] == "Alice"
    assert resp["data"]["source_type"] == "manual"


def test_kb_add_trust_score_stored(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="t", title="T", content="c", trust_score=0.7)
    assert resp["ok"] is True
    assert resp["data"]["trust_score"] == pytest.approx(0.7)


def test_kb_add_trust_score_none_defaults_to_one(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="t", title="T", content="c", trust_score=None)
    assert resp["ok"] is True
    assert resp["data"]["trust_score"] == 1.0


def test_kb_add_invalid_trust_score_too_high(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="t", title="T", content="c", trust_score=1.5)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_add_invalid_trust_score_negative(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="t", title="T", content="c", trust_score=-0.1)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_add_invalid_trust_score_bool_rejected(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    # bool subclasses int; float(True)==1.0 would silently pass without explicit check
    resp = srv.handle_kb_add(topic="t", title="T", content="c", trust_score=True)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_add_db_error_returns_envelope(monkeypatch):
    """A DB exception during insert surfaces as unexpected_exception, not a raise."""

    class _BrokenDb:
        def table(self, _name):
            raise RuntimeError("disk full")

    monkeypatch.setattr(srv, "db", _BrokenDb())
    resp = srv.handle_kb_add(topic="t", title="T", content="c")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_kb_add_embedded_false_when_semantic_off(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(srv, "db", fake)
    resp = srv.handle_kb_add(topic="t", title="T", content="c")
    assert resp["data"]["embedded"] is False


# ---------------------------------------------------------------------------
# handle_kb_get
# ---------------------------------------------------------------------------


def test_kb_get_success(monkeypatch):
    entry = {"kb_id": "kb_42", "title": "My Entry", "content": "c", "topic": "t"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    resp = srv.handle_kb_get(kb_id="kb_42")
    assert resp["ok"] is True
    assert resp["data"]["kb_id"] == "kb_42"
    assert "My Entry" in resp["message"]


def test_kb_get_not_found(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=None))
    resp = srv.handle_kb_get(kb_id="kb_nope")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_kb_get_empty_data_returns_not_found(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row={}))
    resp = srv.handle_kb_get(kb_id="kb_empty")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_kb_get_db_error_returns_envelope(monkeypatch):
    class _Broken:
        def table(self, _):
            raise RuntimeError("connection error")

    monkeypatch.setattr(srv, "db", _Broken())
    resp = srv.handle_kb_get(kb_id="kb_x")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# handle_kb_list
# ---------------------------------------------------------------------------


def test_kb_list_success(monkeypatch):
    entries = [
        {"kb_id": "kb_1", "topic": "t", "title": "A"},
        {"kb_id": "kb_2", "topic": "t", "title": "B"},
    ]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries, count=2))
    resp = srv.handle_kb_list()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 2
    assert resp["data"]["total_count"] == 2
    assert resp["data"]["has_more"] is False


def test_kb_list_with_topic_filter(monkeypatch):
    entries = [{"kb_id": "kb_1", "topic": "python", "title": "A"}]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries, count=1))
    resp = srv.handle_kb_list(topic="python")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1


def test_kb_list_limit_clamped_to_max(monkeypatch):
    """limit > 500 must be clamped to 500."""
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[], count=0))
    resp = srv.handle_kb_list(limit=9999)
    assert resp["ok"] is True
    assert resp["data"]["limit"] == 500


def test_kb_list_limit_clamped_to_min(monkeypatch):
    """limit < 1 must be clamped to 1."""
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[], count=0))
    resp = srv.handle_kb_list(limit=0)
    assert resp["ok"] is True
    assert resp["data"]["limit"] == 1


def test_kb_list_offset_clamped_to_zero(monkeypatch):
    """Negative offset must be clamped to 0."""
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[], count=0))
    resp = srv.handle_kb_list(offset=-5)
    assert resp["ok"] is True
    assert resp["data"]["offset"] == 0


def test_kb_list_has_more_true_when_more_pages(monkeypatch):
    entries = [{"kb_id": f"kb_{i}"} for i in range(10)]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries, count=50))
    resp = srv.handle_kb_list(limit=10, offset=0)
    assert resp["data"]["has_more"] is True
    assert resp["data"]["total_count"] == 50


def test_kb_list_unknown_kwarg_returns_invalid_input(monkeypatch):
    """Hallucinated filters must be caught before DB access."""
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_kb_list(created_at__gte="2026-01-01")
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"
    assert "created_at__gte" in resp["message"]


def test_kb_list_db_error_returns_envelope(monkeypatch):
    class _Broken:
        def table(self, _):
            raise RuntimeError("timeout")

    monkeypatch.setattr(srv, "db", _Broken())
    resp = srv.handle_kb_list()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# handle_kb_delete
# ---------------------------------------------------------------------------


def test_kb_delete_success(monkeypatch):
    entry = {"kb_id": "kb_del", "title": "Gone", "content": "bye"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    # _delete_kb_embedding is a no-op for non-semantic setups; patch to avoid
    monkeypatch.setattr(srv, "_delete_kb_embedding", lambda _kb_id: None)
    resp = srv.handle_kb_delete(kb_id="kb_del", confirm=True)
    assert resp["ok"] is True
    assert resp["data"]["kb_id"] == "kb_del"
    assert "Gone" in resp["message"]


def test_kb_delete_no_kb_id_returns_invalid_input(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_kb_delete()
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_delete_without_confirm_returns_error(monkeypatch):
    entry = {"kb_id": "kb_del2", "title": "T"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    resp = srv.handle_kb_delete(kb_id="kb_del2", confirm=False)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"
    assert "confirm" in resp["message"].lower()


def test_kb_delete_not_found(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=None))
    monkeypatch.setattr(srv, "_delete_kb_embedding", lambda _: None)
    resp = srv.handle_kb_delete(kb_id="kb_ghost", confirm=True)
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_kb_delete_entry_id_fallback(monkeypatch):
    """entry_id is accepted when kb_id is absent (BUG-3 compat)."""
    entry = {"kb_id": "kb_old", "title": "Legacy", "content": "c"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    monkeypatch.setattr(srv, "_delete_kb_embedding", lambda _: None)
    resp = srv.handle_kb_delete(entry_id="kb_old", confirm=True)
    assert resp["ok"] is True


def test_kb_delete_db_error_returns_envelope(monkeypatch):
    class _Broken:
        def table(self, _):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(srv, "db", _Broken())
    resp = srv.handle_kb_delete(kb_id="kb_x", confirm=True)
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# _validate_trust_score (directly tested for completeness)
# ---------------------------------------------------------------------------


def test_validate_trust_score_none_passes():
    assert srv._validate_trust_score(None) is None


def test_validate_trust_score_valid_passes():
    assert srv._validate_trust_score(0.5) is None
    assert srv._validate_trust_score(0.0) is None
    assert srv._validate_trust_score(1.0) is None


def test_validate_trust_score_bool_rejected():
    result = srv._validate_trust_score(True)
    assert result is not None
    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_validate_trust_score_out_of_range():
    result = srv._validate_trust_score(1.5)
    assert result is not None
    assert result["ok"] is False

    result2 = srv._validate_trust_score(-0.1)
    assert result2 is not None
    assert result2["ok"] is False


def test_validate_trust_score_non_numeric():
    result = srv._validate_trust_score("high")
    assert result is not None
    assert result["ok"] is False
    assert result["error"] == "invalid_input"
