"""Unit tests for server handler bug fixes found during QA (BUG-1/2/3/7/8).

Pure unit tests — no live database. A small fluent fake stands in for the
db_client query builder so the KB CRUD handlers can be exercised in isolation.
The PostgreSQL round-trip coverage lives in tests/integration/.
"""

from __future__ import annotations

import psycopg2
import pytest

import lore.server as srv
from lore.db_client import QueryResult


# ---------------------------------------------------------------------------
# Fluent fake db: db.table(...).select(...).eq(...).maybe_single().execute()
# and .update(...).eq(...).execute() / .delete().eq(...).execute().
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, db: "_FakeDb"):
        self._db = db
        self._is_update = False

    def select(self, *_a, **_k):
        return self

    def update(self, data):
        self._is_update = True
        self._db.updates.append(data)
        return self

    def delete(self):
        self._db.deleted = True
        return self

    def eq(self, *_a, **_k):
        return self

    def maybe_single(self):
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        if self._is_update:
            if self._db.update_raises is not None:
                raise self._db.update_raises
            return QueryResult(data=self._db.updated_row or {})
        # A read: return the (current) row for this entry.
        return QueryResult(data=self._db.current_row)


class _FakeDb:
    """Configurable fake supporting the KB CRUD handler call chains.

    ``current_row`` is returned by every read (select...maybe_single...execute).
    After an update the handler re-reads, so ``updated_row`` (if set) is returned
    on subsequent reads. ``update_raises`` lets a test simulate a SQL error.
    """

    def __init__(self, current_row=None, updated_row=None, update_raises=None):
        self.current_row = current_row if current_row is not None else {}
        self.updated_row = updated_row
        self.update_raises = update_raises
        self.updates: list[dict] = []
        self.deleted = False

    def table(self, _name):
        # Once an update has been applied, point reads at the updated row.
        if self.updated_row is not None and self.updates:
            self.current_row = self.updated_row
        return _FakeQuery(self)


@pytest.fixture(autouse=True)
def _no_semantic(monkeypatch):
    """Disable the embed-on-write path so handlers don't touch embeddings."""
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)


# ---------------------------------------------------------------------------
# BUG-1: kb_update with an unsupported column -> clean invalid_input, not raise
# ---------------------------------------------------------------------------


def test_kb_update_metadata_returns_clean_error(monkeypatch):
    """Passing metadata (no such column) must surface a clean invalid_input
    error rather than leaking the raw psycopg2 ProgrammingError."""
    existing = {"kb_id": "kb_1", "title": "T", "content": "c"}
    # The update execute() raises UndefinedColumn (a psycopg2.ProgrammingError
    # subclass) — mirroring what Postgres does for an unknown column.
    err = psycopg2.errors.UndefinedColumn("column \"metadata\" does not exist")
    fake = _FakeDb(current_row=existing, update_raises=err)
    monkeypatch.setattr(srv, "db", fake)

    # metadata is no longer a declared param; simulate the value reaching SQL
    # by updating a (pretend-unknown) field via the supported `content` path.
    resp = srv.handle_kb_update(kb_id="kb_1", content="new content")
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"
    assert resp["message"] == "Field not supported"


def test_kb_update_metadata_param_removed_from_schema():
    """BUG-1 Fix A: the kb_update inputSchema must not expose `metadata`."""
    schema = srv._TOOL_SCHEMA_MAP["kb_update"]
    assert "metadata" not in schema["properties"]


def test_kb_update_metadata_rejected_at_call_tool_boundary():
    """End-to-end BUG-1: a kb_update call carrying `metadata` is rejected as a
    clean invalid_input at the handler level (**kwargs check), never raising an
    exception that leaks as unexpected_exception.

    Note: additionalProperties:False was removed from the schema because FastMCP
    intercepts schema rejections before call_tool executes, producing a raw
    -32603 transport error instead of our clean envelope. The handler's **kwargs
    check catches unsupported fields and returns the clean invalid_input envelope.
    """
    import asyncio
    import json

    out = asyncio.run(srv.call_tool("kb_update", {"kb_id": "kb_1", "metadata": {"foo": "bar"}}))
    payload = json.loads(out[0].text)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    # It is a validation error, NOT an unexpected_exception.
    assert payload["error"] != "unexpected_exception"


# ---------------------------------------------------------------------------
# BUG-3: kb_update / kb_delete accept kb_id (not just entry_id) + title update
# ---------------------------------------------------------------------------


def test_kb_update_accepts_kb_id_field(monkeypatch):
    existing = {"kb_id": "kb_42", "title": "Old", "content": "c"}
    updated = {"kb_id": "kb_42", "title": "Old", "content": "new"}
    fake = _FakeDb(current_row=existing, updated_row=updated)
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_kb_update(kb_id="kb_42", content="new")
    assert resp["ok"] is True
    assert resp["data"]["kb_id"] == "kb_42"
    assert "content" in resp["data"]["updated_fields"]


def test_kb_update_entry_id_fallback_still_works(monkeypatch):
    """Backward compat: entry_id is accepted when kb_id is absent."""
    existing = {"kb_id": "kb_7", "title": "T", "content": "c"}
    updated = {"kb_id": "kb_7", "title": "T", "content": "z"}
    fake = _FakeDb(current_row=existing, updated_row=updated)
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_kb_update(entry_id="kb_7", content="z")
    assert resp["ok"] is True
    assert resp["data"]["kb_id"] == "kb_7"


def test_kb_update_missing_id_returns_invalid_input(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_kb_update(content="x")
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_update_title_is_updatable(monkeypatch):
    existing = {"kb_id": "kb_9", "title": "Old Title", "content": "c"}
    updated = {"kb_id": "kb_9", "title": "New Title", "content": "c"}
    fake = _FakeDb(current_row=existing, updated_row=updated)
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_kb_update(kb_id="kb_9", title="New Title")
    assert resp["ok"] is True
    assert "title" in resp["data"]["updated_fields"]
    # The title field was actually written to the update payload.
    assert fake.updates[0]["title"] == "New Title"


def test_kb_update_title_in_schema():
    schema = srv._TOOL_SCHEMA_MAP["kb_update"]
    assert "title" in schema["properties"]
    assert schema["required"] == ["kb_id"]


def test_kb_delete_accepts_kb_id_field(monkeypatch):
    existing = {"kb_id": "kb_del", "title": "Doomed"}
    fake = _FakeDb(current_row=existing)
    monkeypatch.setattr(srv, "db", fake)
    monkeypatch.setattr(srv, "_delete_kb_embedding", lambda _id: None)

    resp = srv.handle_kb_delete(kb_id="kb_del", confirm=True)
    assert resp["ok"] is True
    assert resp["data"]["kb_id"] == "kb_del"
    assert fake.deleted is True


def test_kb_delete_schema_uses_kb_id():
    schema = srv._TOOL_SCHEMA_MAP["kb_delete"]
    assert "kb_id" in schema["properties"]
    assert "entry_id" not in schema["properties"]
    assert schema["required"] == ["kb_id"]


# ---------------------------------------------------------------------------
# BUG-2: deduplicate_results without a content field keys on kb_id
# ---------------------------------------------------------------------------


def test_deduplicate_results_without_content_uses_kb_id():
    """Search results (no `content`, only kb_id/title/topic/score) must NOT all
    collapse to one. Distinct kb_ids are kept; a repeated kb_id is removed."""
    results = [
        {"kb_id": "kb_a", "title": "Alpha", "topic": "t", "score": 0.9},
        {"kb_id": "kb_b", "title": "Beta", "topic": "t", "score": 0.8},
        {"kb_id": "kb_a", "title": "Alpha", "topic": "t", "score": 0.7},  # dup id
    ]
    resp = srv.handle_deduplicate_results(results)
    assert resp["ok"] is True
    kept = resp["data"]["results"]
    kept_ids = [r["kb_id"] for r in kept]
    assert kept_ids == ["kb_a", "kb_b"]  # one kb_a removed, kb_b kept
    assert resp["data"]["removed_count"] == 1
    assert resp["data"]["unique_count"] == 2


def test_deduplicate_results_distinct_ids_all_kept():
    """The primary regression: distinct kb_ids must never be de-duped together."""
    results = [{"kb_id": f"kb_{i}", "title": "x", "topic": "t"} for i in range(5)]
    resp = srv.handle_deduplicate_results(results)
    assert resp["data"]["unique_count"] == 5
    assert resp["data"]["removed_count"] == 0


def test_deduplicate_results_still_dedupes_on_content():
    """Items WITH content are de-duped on normalized text as before."""
    results = [
        {"content": "Same Text"},
        {"content": "same text"},  # case/space-insensitive dup
        {"content": "different"},
    ]
    resp = srv.handle_deduplicate_results(results)
    assert resp["data"]["unique_count"] == 2
    assert resp["data"]["removed_count"] == 1


# ---------------------------------------------------------------------------
# BUG-7: log_retrieval_feedback rejects out-of-range user_feedback_score
# ---------------------------------------------------------------------------


def _enable_mining(monkeypatch):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.setenv("DB_BACKEND", "local")


@pytest.mark.parametrize("bad_score", [999, 0, -1, 6])
def test_log_feedback_score_out_of_range(monkeypatch, bad_score):
    _enable_mining(monkeypatch)
    # update_retrieval_feedback must never be reached for a bad score.
    monkeypatch.setattr(
        srv.telemetry,
        "update_retrieval_feedback",
        lambda **_k: pytest.fail("DB update should not run for out-of-range score"),
    )
    resp = srv.handle_log_retrieval_feedback("qry_x", user_feedback_score=bad_score)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"
    assert "between 1 and 5" in resp["message"]


@pytest.mark.parametrize("good_score", [1, 3, 5])
def test_log_feedback_score_in_range_succeeds(monkeypatch, good_score):
    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "update_retrieval_feedback", lambda **_k: 1)
    resp = srv.handle_log_retrieval_feedback("qry_ok", user_feedback_score=good_score)
    assert resp["ok"] is True
    assert resp["data"]["updated"] == 1


def test_log_feedback_score_schema_bounds():
    schema = srv._TOOL_SCHEMA_MAP["log_retrieval_feedback"]
    score = schema["properties"]["user_feedback_score"]
    assert score["minimum"] == 1
    assert score["maximum"] == 5


# ---------------------------------------------------------------------------
# BUG-8: kb_search with top_k=0 returns invalid_input (not 1 result)
# ---------------------------------------------------------------------------


def test_kb_search_top_k_zero_returns_error(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "")  # legacy lexical path
    resp = srv.handle_kb_search("anything", top_k=0)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"
    assert "at least 1" in resp["message"]


def test_kb_search_top_k_schema_minimum():
    schema = srv._TOOL_SCHEMA_MAP["kb_search"]
    assert schema["properties"]["top_k"]["minimum"] == 1
