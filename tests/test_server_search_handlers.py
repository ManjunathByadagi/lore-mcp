"""Unit tests for handle_kb_search, filter helpers, and server utilities.

Targets the uncovered paths in server.py:
- _filter_by_min_score (all branches: None, fts, hybrid, missing field)
- _filter_by_min_trust_score (all branches)
- handle_kb_search legacy lexical path (most reachable without Postgres/embeddings)
- handle_kb_search degraded-mode path (semantic requested but unavailable)
- handle_kb_search top_k validation edge cases
- handle_kb_search exception path
- _semantic_write_enabled branches
- _backend_kind branches
- _coerce_arguments array/object coercion
- call_tool routing for unknown tool + backfill_query_embeddings
- _finalize_search_response with mining disabled (no-op)

Uses the same fluent-fake pattern: a _FakeDb injects predictable rows so
the legacy lexical FTS path returns controlled data.
"""

from __future__ import annotations

import os

import pytest

import lore.server as srv
from lore.db_client import QueryResult

# ---------------------------------------------------------------------------
# Fake DB for kb_search legacy lexical path
# ---------------------------------------------------------------------------


class _FakeOrQuery:
    """Chainable fluent fake that supports or_(), limit(), eq() and execute()."""

    def __init__(self, rows: list[dict], count: int | None = None, raise_on_or: bool = False):
        self._rows = rows
        self._count = count
        self._raise_on_or = raise_on_or

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def or_(self, *_a, **_k):
        if self._raise_on_or:
            raise RuntimeError("wfts error simulated")
        return self

    def limit(self, *_a, **_k):
        return self

    def execute(self):
        return QueryResult(data=self._rows, count=self._count)


class _FakeSearchDb:
    """DB stub returning fixed rows from the legacy lexical search path."""

    def __init__(self, rows: list[dict] | None = None, raise_on_or: bool = False):
        self._rows = rows or []
        self._raise_on_or = raise_on_or

    def table(self, _name):
        return _FakeOrQuery(self._rows, raise_on_or=self._raise_on_or)

    # Make sure semantic checks fail gracefully
    vec_extension_loaded: bool = False
    fts5_available: bool = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_mining(monkeypatch):
    """Disable telemetry mining so _finalize_search_response is a no-op."""
    import lore.telemetry as tel

    monkeypatch.setattr(tel, "mining_enabled", lambda: False)


@pytest.fixture(autouse=True)
def _no_semantic_env(monkeypatch):
    """Ensure LORE_SEMANTIC_SEARCH is off; DB_BACKEND unset so we hit legacy path."""
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    monkeypatch.delenv("DB_BACKEND", raising=False)


# ---------------------------------------------------------------------------
# _filter_by_min_score
# ---------------------------------------------------------------------------


def test_filter_by_min_score_none_returns_unchanged():
    results = [{"kb_id": "a", "score": 0.1}, {"kb_id": "b", "score": 0.9}]
    out = srv._filter_by_min_score(results, "fts", None)
    assert out is results  # same object, no filtering


def test_filter_by_min_score_fts_uses_score_field():
    results = [
        {"kb_id": "high", "score": 0.8},
        {"kb_id": "low", "score": 0.2},
    ]
    out = srv._filter_by_min_score(results, "fts", 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "high"


def test_filter_by_min_score_hybrid_uses_rrf_score_field():
    results = [
        {"kb_id": "good", "rrf_score": 0.7},
        {"kb_id": "bad", "rrf_score": 0.1},
    ]
    out = srv._filter_by_min_score(results, "hybrid", 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "good"


def test_filter_by_min_score_semantic_uses_score_field():
    results = [
        {"kb_id": "ok", "score": 0.6},
        {"kb_id": "nope", "score": 0.3},
    ]
    out = srv._filter_by_min_score(results, "semantic", 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "ok"


def test_filter_by_min_score_missing_field_defaults_to_zero():
    """Results with no score field default to 0.0 and are excluded when min_score > 0."""
    results = [
        {"kb_id": "no_score"},
        {"kb_id": "has_score", "score": 0.9},
    ]
    out = srv._filter_by_min_score(results, "fts", 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "has_score"


def test_filter_by_min_score_zero_threshold_passes_all():
    """min_score=0.0 keeps results that have score >= 0 (includes no-score rows)."""
    results = [{"kb_id": "x", "score": 0.0}, {"kb_id": "y"}]
    out = srv._filter_by_min_score(results, "fts", 0.0)
    # Both have effective score 0.0 which >= 0.0
    assert len(out) == 2


def test_filter_by_min_score_empty_list():
    out = srv._filter_by_min_score([], "fts", 0.5)
    assert out == []


# ---------------------------------------------------------------------------
# _filter_by_min_trust_score
# ---------------------------------------------------------------------------


def test_filter_by_min_trust_score_none_returns_unchanged():
    results = [{"kb_id": "x", "trust_score": 0.1}]
    out = srv._filter_by_min_trust_score(results, None)
    assert out is results


def test_filter_by_min_trust_score_excludes_low_confidence():
    results = [
        {"kb_id": "trusted", "trust_score": 0.9},
        {"kb_id": "untrusted", "trust_score": 0.3},
    ]
    out = srv._filter_by_min_trust_score(results, 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "trusted"


def test_filter_by_min_trust_score_missing_defaults_to_one():
    """Legacy rows without trust_score are treated as fully trusted (1.0)."""
    results = [
        {"kb_id": "legacy"},  # no trust_score
        {"kb_id": "low", "trust_score": 0.2},
    ]
    out = srv._filter_by_min_trust_score(results, 0.5)
    assert len(out) == 1
    assert out[0]["kb_id"] == "legacy"


def test_filter_by_min_trust_score_exact_boundary():
    results = [{"kb_id": "exactly_half", "trust_score": 0.5}]
    out = srv._filter_by_min_trust_score(results, 0.5)
    assert len(out) == 1  # >= not >


def test_filter_by_min_trust_score_empty_list():
    out = srv._filter_by_min_trust_score([], 0.5)
    assert out == []


# ---------------------------------------------------------------------------
# handle_kb_search — legacy lexical path
# ---------------------------------------------------------------------------


def test_kb_search_legacy_returns_results(monkeypatch):
    """Default (no semantic env) hits the legacy LIKE path and returns ok=True."""
    rows = [{"kb_id": "kb_1", "title": "Python tips", "topic": "python", "trust_score": 1.0}]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="python")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1
    assert resp["data"]["search_mode"] == "fts"


def test_kb_search_legacy_empty_results(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))
    resp = srv.handle_kb_search(query="nonexistent topic xyz123")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 0


def test_kb_search_degraded_mode_when_semantic_requested(monkeypatch):
    """semantic=True with no embeddings degrades to lexical and sets degraded flag."""
    rows = [{"kb_id": "kb_1", "title": "T", "trust_score": 1.0}]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    # LORE_SEMANTIC_SEARCH unset → wants_vectors=True but no embedder
    resp = srv.handle_kb_search(query="test", semantic=True)
    assert resp["ok"] is True
    assert resp["data"].get("degraded") is True
    assert "degraded_reason" in resp["data"]


def test_kb_search_degraded_mode_hybrid(monkeypatch):
    """hybrid=True with no embeddings degrades to lexical and sets degraded flag."""
    rows = []
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="test", hybrid=True)
    assert resp["ok"] is True
    assert resp["data"].get("degraded") is True


def test_kb_search_degraded_mode_search_mode_semantic(monkeypatch):
    """search_mode='semantic' without LORE_SEMANTIC_SEARCH degrades."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))
    resp = srv.handle_kb_search(query="test", search_mode="semantic")
    assert resp["ok"] is True
    assert resp["data"].get("degraded") is True


def test_kb_search_top_k_zero_returns_error(monkeypatch):
    """top_k=0 must return invalid_input."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb())
    resp = srv.handle_kb_search(query="test", top_k=0)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_search_top_k_negative_returns_error(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeSearchDb())
    resp = srv.handle_kb_search(query="test", top_k=-5)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_kb_search_top_k_clamped_to_200(monkeypatch):
    """top_k > 200 is clamped; the response must still succeed."""
    rows = [{"kb_id": "x", "trust_score": 1.0}]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="test", top_k=9999)
    assert resp["ok"] is True  # clamped, not rejected


def test_kb_search_top_k_invalid_type_defaults(monkeypatch):
    """Non-numeric top_k is coerced to default 20 (no crash)."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))
    resp = srv.handle_kb_search(query="test", top_k="bad")
    assert resp["ok"] is True  # defaults to 20, runs legacy path


def test_kb_search_min_trust_score_filters_results(monkeypatch):
    """min_trust_score filters are applied before returning."""
    rows = [
        {"kb_id": "good", "trust_score": 0.9},
        {"kb_id": "bad", "trust_score": 0.2},
    ]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="test", min_trust_score=0.5)
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1
    assert resp["data"]["results"][0]["kb_id"] == "good"


def test_kb_search_min_score_excludes_zero_score(monkeypatch):
    """Legacy path carries no per-row score; min_score > 0 excludes all rows."""
    rows = [{"kb_id": "no_score", "trust_score": 1.0}]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="test", min_score=0.5)
    assert resp["ok"] is True
    assert resp["data"]["count"] == 0  # no-score rows default to 0.0


def test_kb_search_or_fallback_on_wfts_error(monkeypatch):
    """If or_(wfts) raises, handler falls back to plfts and still succeeds."""
    rows = [{"kb_id": "kb_fallback", "trust_score": 1.0}]
    # First call to or_ raises; second (plfts fallback) returns rows

    class _FallbackOrQuery:
        def __init__(self):
            self._call_count = 0
            self._rows = rows

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def or_(self, *_a, **_k):
            self._call_count += 1
            if self._call_count == 1:
                raise RuntimeError("wfts error simulated")
            return self  # second call (plfts) works

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return QueryResult(data=self._rows)

    class _FallbackDb:
        vec_extension_loaded = False
        fts5_available = False

        def table(self, _name):
            return _FallbackOrQuery()

    monkeypatch.setattr(srv, "db", _FallbackDb())
    resp = srv.handle_kb_search(query="test")
    assert resp["ok"] is True


def test_kb_search_db_exception_returns_envelope(monkeypatch):
    """An unexpected exception from the DB is caught and returned as error."""

    class _BrokenDb:
        vec_extension_loaded = False
        fts5_available = False

        def table(self, _name):
            raise RuntimeError("connection lost")

    monkeypatch.setattr(srv, "db", _BrokenDb())
    resp = srv.handle_kb_search(query="test")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_kb_search_search_mode_fts_explicit(monkeypatch):
    """search_mode='fts' explicit does the same as the default legacy path."""
    rows = [{"kb_id": "explicit_fts", "trust_score": 1.0}]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))
    resp = srv.handle_kb_search(query="test", search_mode="fts")
    assert resp["ok"] is True
    assert resp["data"]["search_mode"] == "fts"


# ---------------------------------------------------------------------------
# _semantic_write_enabled
# ---------------------------------------------------------------------------


def test_semantic_write_enabled_false_when_env_off(monkeypatch):
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    assert srv._semantic_write_enabled() is False


def test_semantic_write_enabled_false_when_env_set_false(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    assert srv._semantic_write_enabled() is False


def test_semantic_write_enabled_false_when_no_db(monkeypatch):
    """Even with env=true, returns False when db lacks vec_extension_loaded."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("DB_BACKEND", "sqlite")

    class _NoVecDb:
        vec_extension_loaded = False

    monkeypatch.setattr(srv, "db", _NoVecDb())
    assert srv._semantic_write_enabled() is False


def test_semantic_write_enabled_true_for_sqlite(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("DB_BACKEND", "sqlite")

    class _VecDb:
        vec_extension_loaded = True

    monkeypatch.setattr(srv, "db", _VecDb())
    assert srv._semantic_write_enabled() is True


def test_semantic_write_enabled_true_for_postgres(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("DB_BACKEND", "postgres")

    class _VecDb:
        vec_extension_loaded = True

    monkeypatch.setattr(srv, "db", _VecDb())
    assert srv._semantic_write_enabled() is True


def test_semantic_write_enabled_true_for_postgresql_alias(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    monkeypatch.setenv("DB_BACKEND", "postgresql")

    class _VecDb:
        vec_extension_loaded = True

    monkeypatch.setattr(srv, "db", _VecDb())
    assert srv._semantic_write_enabled() is True


# ---------------------------------------------------------------------------
# _backend_kind
# ---------------------------------------------------------------------------


def test_backend_kind_sqlite(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    assert srv._backend_kind() == "sqlite"


def test_backend_kind_postgres(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgres")
    assert srv._backend_kind() == "postgres"


def test_backend_kind_local(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "local")
    assert srv._backend_kind() == "postgres"


def test_backend_kind_postgresql(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "postgresql")
    assert srv._backend_kind() == "postgres"


def test_backend_kind_empty(monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    assert srv._backend_kind() == ""


def test_backend_kind_unknown(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "supabase")
    assert srv._backend_kind() == ""


# ---------------------------------------------------------------------------
# _coerce_arguments
# ---------------------------------------------------------------------------


def test_coerce_arguments_empty_schema():
    args = {"tags": '["a","b"]'}
    result = srv._coerce_arguments(args, {})
    assert result is args  # returned unchanged


def test_coerce_arguments_json_array_coerced():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": '["python", "testing"]'}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["python", "testing"]


def test_coerce_arguments_comma_separated_array():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": "python, testing, async"}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["python", "testing", "async"]


def test_coerce_arguments_space_separated_array():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": "python testing async"}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["python", "testing", "async"]


def test_coerce_arguments_object_coerced():
    schema = {"properties": {"config": {"type": "object"}}}
    args = {"config": '{"key": "value"}'}
    result = srv._coerce_arguments(args, schema)
    assert result["config"] == {"key": "value"}


def test_coerce_arguments_already_list_passthrough():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": ["a", "b"]}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["a", "b"]


def test_coerce_arguments_missing_field_skipped():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"title": "No tags here"}
    result = srv._coerce_arguments(args, schema)
    assert "tags" not in result


# ---------------------------------------------------------------------------
# call_tool routing: unknown tool + backfill_query_embeddings
# ---------------------------------------------------------------------------


def test_call_tool_unknown_tool_returns_not_found():
    """An unknown tool name must return not_found, not raise."""
    import asyncio
    import json

    out = asyncio.run(srv.call_tool("no_such_tool_xyz", {}))
    payload = json.loads(out[0].text)
    assert payload["ok"] is False
    assert payload["error"] == "not_found"
    assert "no_such_tool_xyz" in payload["message"]


def test_call_tool_validation_error_returned_cleanly():
    """A jsonschema validation error (wrong type) returns invalid_input."""
    import asyncio
    import json

    # kb_add requires topic/title/content (strings); pass wrong type
    out = asyncio.run(srv.call_tool("kb_add", {"topic": 123, "title": "T", "content": "c"}))
    payload = json.loads(out[0].text)
    # jsonschema should flag topic as not a string
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
