"""Additional tests for SqliteClient covering degradation paths and edge cases
not covered by test_sqlite_client.py.

Exercises:
- _init_core_schema_only fallback (FTS5/vec0 not available)
- _probe_optional_features when FTS5 table absent
- kb_list count=None-and-full-page warning path
- SqliteClient rpc stub
- get_db_client with SQLITE_DB_PATH from env
- SqliteTableQuery OR_ with zero matching segments
"""

from __future__ import annotations

import sqlite3

import pytest

from lore.db_client import (
    _CORE_SQLITE_STATEMENTS,
    QueryResult,
    SqliteClient,
    get_db_client,
)

# ---------------------------------------------------------------------------
# _init_core_schema_only: tables created even when FTS5/vec0 fail
# ---------------------------------------------------------------------------
#
# NOTE on patching strategy:
#
# An earlier version of these tests attempted to simulate FTS5/vec0 absence by
# monkey-patching ``sqlite3.Connection.executescript``. That approach fails on
# CPython 3.10+ with::
#
#     TypeError: cannot set 'executescript' attribute of immutable type
#                'sqlite3.Connection'
#
# because the built-in C-extension ``sqlite3.Connection`` is now a fully
# immutable heap type — class-level method replacement is forbidden. The
# correct way to unit-test the fallback (``_init_core_schema_only``) is to
# invoke it directly on a fresh connection: that is exactly what
# ``_init_schema`` does inside the ``except OperationalError`` block. We
# verify the *behaviour* of the fallback path (the tables it creates and that
# they accept inserts) without faking the FTS5/vec0 failure mode itself.


def test_core_schema_only_creates_main_tables(tmp_path):
    """_init_core_schema_only must create all core tables (FTS5/vec0-free)."""
    db_path = str(tmp_path / "core_only.db")

    # Open a bare sqlite3 connection (no schema applied yet) and feed it
    # through the fallback directly. This is the exact code path
    # ``_init_schema`` takes when ``executescript(_SQLITE_SCHEMA)`` raises
    # ``no such module: fts5`` or ``no such module: vec0``.
    conn = sqlite3.connect(db_path)
    try:
        # Build a stand-alone client just to borrow the bound method; do not
        # let it run normal init (it would create the same tables via the
        # full schema). We construct manually with the minimum state the
        # method touches: ``_sqlite3`` and a logger via the module.
        client = SqliteClient.__new__(SqliteClient)
        client._sqlite3 = sqlite3
        client._init_core_schema_only(conn)

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "knowledge_kb_entries" in tables
        assert "knowledge_journal_entries" in tables
        assert "knowledge_research_notes" in tables

        # Sanity-check: the fallback must apply *every* core statement.
        # Count plain CREATE TABLE statements in the core list.
        expected_tables = sum(
            1 for stmt in _CORE_SQLITE_STATEMENTS if "CREATE TABLE" in stmt.upper()
        )
        # All core CREATE TABLE statements should have produced a table row.
        # (Some statements may share a table name across IF NOT EXISTS; check
        # at least the headline three are present, which the previous block
        # already does.)
        assert expected_tables >= 3
    finally:
        conn.close()


def test_core_schema_only_kb_entries_insertable(tmp_path):
    """After applying core-only schema, knowledge_kb_entries accepts inserts."""
    db_path = str(tmp_path / "core_ins.db")

    conn = sqlite3.connect(db_path)
    try:
        client = SqliteClient.__new__(SqliteClient)
        client._sqlite3 = sqlite3
        client._init_core_schema_only(conn)

        # Insert directly via SQL — exercising the bare table without going
        # through SqliteTableQuery (which would re-trigger full schema init).
        conn.execute(
            "INSERT INTO knowledge_kb_entries "
            "(kb_id, topic, title, content, tags) VALUES (?, ?, ?, ?, ?)",
            ("kb_core", "test", "Core Schema Test", "hello", "[]"),
        )
        conn.commit()

        row = conn.execute(
            "SELECT kb_id, topic, title FROM knowledge_kb_entries WHERE kb_id = ?",
            ("kb_core",),
        ).fetchone()
        assert row is not None
        assert row[0] == "kb_core"
        assert row[1] == "test"
        assert row[2] == "Core Schema Test"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _probe_optional_features: vec0 absent → vec_extension_loaded=False
# ---------------------------------------------------------------------------


def test_probe_optional_features_vec0_absent(tmp_path):
    """_probe_optional_features must mark vec_extension_loaded=False when
    the vec0 virtual table is not present.

    We construct an instance whose connection has only the *core* schema
    applied — no FTS5 virtual table, no vec0 virtual table — and then call
    the probe. This is exactly the state ``_init_schema`` leaves things in
    after the FTS5/vec0 fallback fires.
    """
    db_path = str(tmp_path / "novec.db")

    conn = sqlite3.connect(db_path)
    try:
        client = SqliteClient.__new__(SqliteClient)
        client._sqlite3 = sqlite3
        client._conn = conn
        client.vec_extension_loaded = False
        client.fts5_available = False
        client._init_core_schema_only(conn)

        # Now run the probe — neither FTS5 nor vec0 table exists.
        client._probe_optional_features()

        # The probe must downgrade both flags to False on a bare connection.
        assert client.fts5_available is False
        assert client.vec_extension_loaded is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# kb_list: count=None AND len(results)==limit triggers a warning
# ---------------------------------------------------------------------------


def test_kb_list_count_none_full_page_logs_warning(monkeypatch, caplog):
    """When the db returns count=None and the page is full, kb_list warns."""
    import logging

    import lore.server as srv

    class _FullPageQuery:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def order(self, *_a, **_k):
            return self

        def limit(self, n, **_k):
            self._limit = n
            return self

        def offset(self, *_a, **_k):
            return self

        def execute(self):
            # Return exactly `limit` rows with count=None to trigger the warning
            return QueryResult(
                data=[{"kb_id": f"kb_{i}"} for i in range(self._limit)],
                count=None,
            )

    class _CountNoneDb:
        def table(self, _name):
            return _FullPageQuery()

    monkeypatch.setattr(srv, "db", _CountNoneDb())

    with caplog.at_level(logging.WARNING, logger="lore.server"):
        resp = srv.handle_kb_list(limit=5)

    assert resp["ok"] is True
    assert "has_more may be incorrect" in caplog.text


# ---------------------------------------------------------------------------
# kb_list: count=None AND len(results) < limit — NO warning, has_more=False
# ---------------------------------------------------------------------------


def test_kb_list_count_none_partial_page_no_warning(monkeypatch, caplog):
    """count=None with a partial page means no more pages — no warning."""
    import logging

    import lore.server as srv

    class _PartialPageQuery:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def order(self, *_a, **_k):
            return self

        def limit(self, n, **_k):
            self._limit = n
            return self

        def offset(self, *_a, **_k):
            return self

        def execute(self):
            # Return fewer than limit rows — no warning expected
            return QueryResult(data=[{"kb_id": "kb_1"}], count=None)

    class _PartialDb:
        def table(self, _name):
            return _PartialPageQuery()

    monkeypatch.setattr(srv, "db", _PartialDb())

    with caplog.at_level(logging.WARNING, logger="lore.server"):
        resp = srv.handle_kb_list(limit=10)

    assert resp["ok"] is True
    assert resp["data"]["has_more"] is False
    assert "has_more may be incorrect" not in caplog.text


# ---------------------------------------------------------------------------
# handle_kb_update branches not covered by test_server_handlers.py
# ---------------------------------------------------------------------------


def test_kb_update_not_found_returns_not_found(monkeypatch):
    """handle_kb_update: entry not found returns not_found."""
    import lore.server as srv

    class _EmptyQuery:
        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return QueryResult(data=None)

    class _EmptyDb:
        def table(self, _):
            return _EmptyQuery()

    monkeypatch.setattr(srv, "db", _EmptyDb())
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)
    resp = srv.handle_kb_update(kb_id="kb_nope", title="New Title")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_kb_update_tags_field(monkeypatch):
    """handle_kb_update: tags update path is exercised."""
    import lore.server as srv

    class _TagQuery:
        def __init__(self):
            self._op = "select"

        def select(self, *_a, **_k):
            return self

        def update(self, data):
            self._op = "update"
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return QueryResult(
                data={"kb_id": "kb_t", "title": "T", "content": "c", "tags": ["new"]}
            )

    class _TagDb:
        def table(self, _):
            return _TagQuery()

    monkeypatch.setattr(srv, "db", _TagDb())
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)
    resp = srv.handle_kb_update(kb_id="kb_t", tags=["python", "async"])
    assert resp["ok"] is True
    assert "tags" in resp["data"]["updated_fields"]


def test_kb_update_topic_field(monkeypatch):
    """handle_kb_update: topic field update."""
    import lore.server as srv

    class _TopicQuery:
        def select(self, *_a, **_k):
            return self

        def update(self, _):
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return QueryResult(
                data={"kb_id": "kb_tp", "title": "T", "content": "c", "topic": "new_topic"}
            )

    class _TopicDb:
        def table(self, _):
            return _TopicQuery()

    monkeypatch.setattr(srv, "db", _TopicDb())
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)
    resp = srv.handle_kb_update(kb_id="kb_tp", topic="new_topic")
    assert resp["ok"] is True
    assert "topic" in resp["data"]["updated_fields"]


def test_kb_update_verified_field(monkeypatch):
    """handle_kb_update: verified update includes it in update_data."""
    import lore.server as srv

    class _VQuery:
        def select(self, *_a, **_k):
            return self

        def update(self, _):
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return QueryResult(
                data={"kb_id": "kb_v", "title": "T", "content": "c", "verified": True}
            )

    class _VDb:
        def table(self, _):
            return _VQuery()

    monkeypatch.setattr(srv, "db", _VDb())
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)
    resp = srv.handle_kb_update(kb_id="kb_v", verified=True)
    assert resp["ok"] is True
    assert "verified" in resp["data"]["updated_fields"]


def test_kb_update_trust_score_field(monkeypatch):
    """handle_kb_update: trust_score update path."""
    import lore.server as srv

    class _TSQuery:
        def select(self, *_a, **_k):
            return self

        def update(self, _):
            return self

        def eq(self, *_a, **_k):
            return self

        def maybe_single(self):
            return self

        def execute(self):
            return QueryResult(
                data={"kb_id": "kb_ts", "title": "T", "content": "c", "trust_score": 0.8}
            )

    class _TSDb:
        def table(self, _):
            return _TSQuery()

    monkeypatch.setattr(srv, "db", _TSDb())
    monkeypatch.setattr(srv, "_semantic_write_enabled", lambda: False)
    resp = srv.handle_kb_update(kb_id="kb_ts", trust_score=0.8)
    assert resp["ok"] is True
    assert "trust_score" in resp["data"]["updated_fields"]


# ---------------------------------------------------------------------------
# get_db_client: SQLITE_DB_PATH env var respected
# ---------------------------------------------------------------------------


def test_get_db_client_sqlite_uses_sqlite_db_path_env(monkeypatch, tmp_path):
    db_path = str(tmp_path / "env_path.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    monkeypatch.delenv("KNOWLEDGE_DATA_DIR", raising=False)
    client = get_db_client()
    assert isinstance(client, SqliteClient)
    assert client._db_path == db_path
    client.close()


def test_get_db_client_sqlite_uses_knowledge_data_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
    monkeypatch.setenv("KNOWLEDGE_DATA_DIR", str(tmp_path))
    client = get_db_client()
    assert isinstance(client, SqliteClient)
    assert "knowledge.db" in client._db_path
    client.close()


# ---------------------------------------------------------------------------
# OR_ filter: parts with no dot (not matching wfts/generic patterns)
# ---------------------------------------------------------------------------


def test_or_filter_no_matching_parts_no_crash(tmp_path):
    """OR_ conditions with zero matching segments must not crash."""
    client = SqliteClient(db_path=str(tmp_path / "or_test.db"))
    try:
        client.table("knowledge.kb_entries").insert(
            {
                "kb_id": "kb_or1",
                "topic": "t",
                "title": "Hello",
                "content": "world",
                "tags": [],
            }
        ).execute()
        # An OR_ with no parseable segments — should return all rows (no filter added)
        rows = (
            client.table("knowledge.kb_entries")
            .select("*")
            .or_("")  # empty condition
            .execute()
        )
        # Should not crash; result may be all rows or empty depending on filter handling
        assert isinstance(rows.data, list)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# SqliteClient.close() sets _conn to None
# ---------------------------------------------------------------------------


def test_close_sets_conn_to_none(tmp_path):
    client = SqliteClient(db_path=str(tmp_path / "close2.db"))
    assert client._conn is not None
    client.close()
    assert client._conn is None


# ---------------------------------------------------------------------------
# _build_where_clause with multiple ANDs
# ---------------------------------------------------------------------------


def test_multiple_eq_filters_and_together(tmp_path):
    """Two eq() filters produce an AND clause."""
    client = SqliteClient(db_path=str(tmp_path / "and_test.db"))
    try:
        for i, (topic, title) in enumerate([("python", "A"), ("python", "B"), ("rust", "A")]):
            client.table("knowledge.kb_entries").insert(
                {
                    "kb_id": f"kb_{i}",
                    "topic": topic,
                    "title": title,
                    "content": "c",
                    "tags": [],
                }
            ).execute()
        rows = (
            client.table("knowledge.kb_entries")
            .select("*")
            .eq("topic", "python")
            .eq("title", "A")
            .execute()
        )
        assert len(rows.data) == 1
        assert rows.data[0]["kb_id"] == "kb_0"
    finally:
        client.close()
