"""Unit tests for ``journal_search`` (Issue #15).

Pure unit tests — no live database. A small fluent fake stands in for the
db_client query builder so the SQLite/lexical path can be exercised in
isolation, and a separate fake connection covers the PostgreSQL FTS path.
PostgreSQL round-trip coverage lives in tests/integration/.
"""

from __future__ import annotations

import pytest

import lore.server as srv
from lore.db_client import QueryResult


# ---------------------------------------------------------------------------
# Fluent fake db for the SQLite / lexical (ILIKE) path. Records the filters and
# modifiers the handler applies, and returns configured rows.
# ---------------------------------------------------------------------------


class _FakeSearchQuery:
    def __init__(self, db: "_FakeSearchDb"):
        self._db = db

    def select(self, *_a, **_k):
        return self

    def eq(self, column, value):
        self._db.filters.append(("eq", column, value))
        return self

    def gte(self, column, value):
        self._db.filters.append(("gte", column, value))
        return self

    def lte(self, column, value):
        self._db.filters.append(("lte", column, value))
        return self

    def ilike(self, column, pattern):
        self._db.filters.append(("ilike", column, pattern))
        return self

    def like(self, column, pattern):
        self._db.filters.append(("like", column, pattern))
        return self

    def order(self, column, desc=False):
        self._db.orders.append((column, desc))
        return self

    def limit(self, count):
        self._db.applied_limit = count
        return self

    def execute(self):
        rows = [dict(r) for r in self._db.rows]
        return QueryResult(data=rows)


class _FakeSearchDb:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.filters: list[tuple] = []
        self.orders: list[tuple] = []
        self.applied_limit = None

    def table(self, _name):
        return _FakeSearchQuery(self)


@pytest.fixture(autouse=True)
def _sqlite_backend(monkeypatch):
    """Default every test to the SQLite/lexical path unless it overrides."""
    monkeypatch.setenv("DB_BACKEND", "sqlite")


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_journal_search_basic_returns_required_fields(monkeypatch):
    """A matching entry comes back with all required fields and a score."""
    rows = [
        {
            "entry_id": "jrnl_1",
            "date": "2026-05-26",
            "entry_type": "session_summary",
            "tags": ["hermes", "memory"],
            "content": "Hermes memory provider recall session summary",
        }
    ]
    fake = _FakeSearchDb(rows=rows)
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_journal_search("memory")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1
    result = resp["data"]["results"][0]
    for field in ("entry_id", "date", "entry_type", "tags", "content", "score"):
        assert field in result, f"missing field {field}"
    assert result["entry_id"] == "jrnl_1"
    assert result["tags"] == ["hermes", "memory"]


def test_journal_search_score_present_in_all_results(monkeypatch):
    """Every returned row carries a numeric score."""
    rows = [
        {"entry_id": "a", "date": "2026-05-01", "entry_type": "daily",
         "tags": [], "content": "alpha beta gamma"},
        {"entry_id": "b", "date": "2026-05-02", "entry_type": "daily",
         "tags": [], "content": "beta beta delta"},
    ]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))

    resp = srv.handle_journal_search("beta")
    assert resp["ok"] is True
    for r in resp["data"]["results"]:
        assert "score" in r
        assert isinstance(r["score"], (int, float))


def test_journal_search_ranks_by_score_descending(monkeypatch):
    """Higher term-frequency content ranks above lower (relevance ordering)."""
    rows = [
        {"entry_id": "low", "date": "2026-05-02", "entry_type": "daily",
         "tags": [], "content": "beta once"},
        {"entry_id": "high", "date": "2026-05-01", "entry_type": "daily",
         "tags": [], "content": "beta beta beta thrice"},
    ]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))

    resp = srv.handle_journal_search("beta")
    ids = [r["entry_id"] for r in resp["data"]["results"]]
    # "high" (3 hits) should sort before "low" (1 hit) despite later date order.
    assert ids == ["high", "low"]


def test_journal_search_empty_result_count_zero(monkeypatch):
    """No matching rows -> empty results, count=0, ok=True."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))

    resp = srv.handle_journal_search("nonexistent-term")
    assert resp["ok"] is True
    assert resp["data"]["results"] == []
    assert resp["data"]["count"] == 0


# ---------------------------------------------------------------------------
# Limit handling
# ---------------------------------------------------------------------------


def test_journal_search_limit_respected(monkeypatch):
    """limit is passed into the query and bounds the returned rows."""
    rows = [
        {"entry_id": f"e{i}", "date": "2026-05-01", "entry_type": "daily",
         "tags": [], "content": "match term here"}
        for i in range(5)
    ]
    fake = _FakeSearchDb(rows=rows)
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_journal_search("term", limit=3)
    assert resp["ok"] is True
    # The query carried the limit...
    assert fake.applied_limit == 3
    # ...and the post-scoring slice respects it too.
    assert resp["data"]["count"] == 3
    assert len(resp["data"]["results"]) == 3


def test_journal_search_limit_clamped_to_max(monkeypatch):
    """limit above 200 is clamped to 200."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    resp = srv.handle_journal_search("term", limit=9999)
    assert resp["ok"] is True
    assert fake.applied_limit == 200


def test_journal_search_limit_zero_returns_invalid_input(monkeypatch):
    """limit below 1 is meaningless and rejected."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))

    resp = srv.handle_journal_search("term", limit=0)
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


# ---------------------------------------------------------------------------
# Filters (entry_type, date_from, date_to)
# ---------------------------------------------------------------------------


def test_journal_search_entry_type_filter_applied(monkeypatch):
    """entry_type filter is pushed into the query as an eq() on entry_type."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("term", entry_type="milestone")
    assert ("eq", "entry_type", "milestone") in fake.filters


def test_journal_search_no_entry_type_filter_when_omitted(monkeypatch):
    """Omitting entry_type does not add an entry_type eq filter."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("term")
    assert not any(f[0] == "eq" and f[1] == "entry_type" for f in fake.filters)


def test_journal_search_date_from_filter_applied(monkeypatch):
    """date_from is pushed into the query as a gte() on date."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("term", date_from="2026-01-01")
    assert ("gte", "date", "2026-01-01") in fake.filters


def test_journal_search_date_to_filter_applied(monkeypatch):
    """date_to is pushed into the query as an lte() on date."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("term", date_to="2026-12-31")
    assert ("lte", "date", "2026-12-31") in fake.filters


def test_journal_search_date_range_both_bounds(monkeypatch):
    """date_from and date_to together apply both gte and lte filters."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("term", date_from="2026-01-01", date_to="2026-06-30")
    assert ("gte", "date", "2026-01-01") in fake.filters
    assert ("lte", "date", "2026-06-30") in fake.filters


def test_journal_search_applies_ilike_on_content(monkeypatch):
    """The lexical path matches on content via a case-insensitive ILIKE."""
    fake = _FakeSearchDb(rows=[])
    monkeypatch.setattr(srv, "db", fake)

    srv.handle_journal_search("memory")
    ilikes = [f for f in fake.filters if f[0] == "ilike" and f[1] == "content"]
    assert len(ilikes) == 1
    assert ilikes[0][2] == "%memory%"


# ---------------------------------------------------------------------------
# Backward compatibility — missing optional params don't error
# ---------------------------------------------------------------------------


def test_journal_search_only_query_arg_does_not_error(monkeypatch):
    """Calling with just `query` (all optional params omitted) succeeds."""
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=[]))
    resp = srv.handle_journal_search("anything")
    assert resp["ok"] is True
    assert "results" in resp["data"]


# ---------------------------------------------------------------------------
# PostgreSQL FTS path (mocked connection/cursor)
# ---------------------------------------------------------------------------


class _FakePgCursor:
    def __init__(self, conn: "_FakePgConn"):
        self._conn = conn
        self.description = None
        self._fetch: list[tuple] = []

    def execute(self, sql, params):
        self._conn.executed.append((sql, list(params)))
        if self._conn.fts_raises and "ts_rank_cd" in sql:
            raise RuntimeError("text search configuration error")
        # FTS query (has ts_rank_cd) returns scored rows; ILIKE fallback returns
        # rows without a score column.
        if "ts_rank_cd" in sql:
            self.description = [
                ("entry_id",), ("date",), ("entry_type",),
                ("tags",), ("content",), ("score",),
            ]
            self._fetch = self._conn.fts_rows
        else:
            self.description = [
                ("entry_id",), ("date",), ("entry_type",),
                ("tags",), ("content",),
            ]
            self._fetch = self._conn.ilike_rows

    def fetchall(self):
        return self._fetch

    def close(self):
        pass


class _FakePgConn:
    def __init__(self, fts_rows=None, ilike_rows=None, fts_raises=False):
        self.fts_rows = fts_rows or []
        self.ilike_rows = ilike_rows or []
        self.fts_raises = fts_raises
        self.executed: list[tuple] = []

    def cursor(self):
        return _FakePgCursor(self)


class _FakePgDb:
    def __init__(self, conn: _FakePgConn):
        self._conn = conn

    def _get_connection(self):
        return self._conn


def test_journal_search_postgres_fts_path(monkeypatch):
    """PostgreSQL backend uses the ranked FTS query and returns scored rows."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    conn = _FakePgConn(
        fts_rows=[
            ("jrnl_1", "2026-05-26", "session_summary",
             ["hermes"], "hermes memory recall", 1.23),
        ]
    )
    monkeypatch.setattr(srv, "db", _FakePgDb(conn))

    resp = srv.handle_journal_search("memory")
    assert resp["ok"] is True
    assert resp["data"]["backend"] == "postgres"
    assert resp["data"]["count"] == 1
    row = resp["data"]["results"][0]
    assert row["entry_id"] == "jrnl_1"
    assert row["score"] == 1.23
    for field in ("entry_id", "date", "entry_type", "tags", "content", "score"):
        assert field in row
    # The executed SQL must be the FTS variant.
    assert any("ts_rank_cd" in sql for sql, _ in conn.executed)


def test_journal_search_postgres_filters_in_sql(monkeypatch):
    """entry_type and date bounds become SQL predicates + params on the PG path."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    conn = _FakePgConn(fts_rows=[])
    monkeypatch.setattr(srv, "db", _FakePgDb(conn))

    srv.handle_journal_search(
        "memory", entry_type="milestone", date_from="2026-01-01", date_to="2026-06-30"
    )
    fts_sql, fts_params = next((s, p) for s, p in conn.executed if "ts_rank_cd" in s)
    assert "entry_type = %s" in fts_sql
    assert "date >= %s" in fts_sql
    assert "date <= %s" in fts_sql
    assert "milestone" in fts_params
    assert "2026-01-01" in fts_params
    assert "2026-06-30" in fts_params


def test_journal_search_postgres_degrades_to_ilike(monkeypatch):
    """If the FTS query raises, the PG path falls back to ILIKE and still scores."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    conn = _FakePgConn(
        ilike_rows=[
            ("jrnl_2", "2026-04-01", "daily", [], "plain memory note"),
        ],
        fts_raises=True,
    )
    monkeypatch.setattr(srv, "db", _FakePgDb(conn))

    resp = srv.handle_journal_search("memory")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1
    row = resp["data"]["results"][0]
    assert row["entry_id"] == "jrnl_2"
    # Fallback attaches a heuristic score (not from SQL).
    assert "score" in row
    assert row["score"] > 0
    # Both the FTS attempt and the ILIKE fallback ran.
    assert any("ts_rank_cd" in sql for sql, _ in conn.executed)
    assert any("ILIKE" in sql for sql, _ in conn.executed)


# ---------------------------------------------------------------------------
# Schema / registration
# ---------------------------------------------------------------------------


def test_journal_search_in_schema_map():
    """journal_search is registered with the expected input schema."""
    schema = srv._TOOL_SCHEMA_MAP["journal_search"]
    props = schema["properties"]
    assert "query" in props
    assert props["query"]["type"] == "string"
    assert props["limit"]["default"] == 20
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 200
    assert "entry_type" in props
    assert "date_from" in props
    assert "date_to" in props
    assert schema["required"] == ["query"]


def test_journal_search_registered_in_tool_definitions():
    """journal_search appears in the advertised tool list."""
    names = [t.name for t in srv._TOOL_DEFINITIONS]
    assert "journal_search" in names


def test_journal_search_routed_in_call_tool(monkeypatch):
    """call_tool dispatches journal_search to the handler end-to-end.

    Driven via asyncio.run rather than the pytest-asyncio marker so the suite
    needs no asyncio_mode config (the project runs with --strict-markers).
    """
    import asyncio
    import json

    monkeypatch.setenv("DB_BACKEND", "sqlite")
    rows = [
        {"entry_id": "jrnl_x", "date": "2026-05-26", "entry_type": "daily",
         "tags": ["t"], "content": "routing memory test"},
    ]
    monkeypatch.setattr(srv, "db", _FakeSearchDb(rows=rows))

    out = asyncio.run(srv.call_tool("journal_search", {"query": "memory"}))
    payload = json.loads(out[0].text)
    assert payload["ok"] is True
    assert payload["data"]["count"] == 1
    assert payload["data"]["results"][0]["entry_id"] == "jrnl_x"
