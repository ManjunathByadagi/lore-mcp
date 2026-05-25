"""Unit tests for retrieval telemetry (Issue #5, Phase 1).

Pure unit tests — no live database. The PostgreSQL round-trip / FK / 5-path
coverage lives in tests/integration/test_telemetry_postgres.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest import mock

import pytest

from lore import telemetry

# ---------------------------------------------------------------------------
# mining_enabled() flag matrix
# ---------------------------------------------------------------------------


def test_mining_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LORE_HARD_NEGATIVE_MINING", raising=False)
    monkeypatch.setenv("DB_BACKEND", "local")
    assert telemetry.mining_enabled() is False


@pytest.mark.parametrize("backend", ["local", "postgres", "postgresql"])
def test_mining_enabled_requires_both_flag_and_pg_backend(monkeypatch, backend):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.setenv("DB_BACKEND", backend)
    assert telemetry.mining_enabled() is True


def test_mining_disabled_when_flag_true_but_sqlite(monkeypatch):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    assert telemetry.mining_enabled() is False


def test_mining_disabled_when_flag_true_but_supabase(monkeypatch):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.setenv("DB_BACKEND", "supabase")
    assert telemetry.mining_enabled() is False


def test_mining_disabled_when_pg_backend_but_flag_off(monkeypatch):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")
    assert telemetry.mining_enabled() is False


@pytest.mark.parametrize("flag", ["TRUE", "True", "  true  ", "tRuE"])
def test_mining_flag_case_and_whitespace_insensitive(monkeypatch, flag):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", flag)
    monkeypatch.setenv("DB_BACKEND", "local")
    assert telemetry.mining_enabled() is True


@pytest.mark.parametrize("flag", ["1", "yes", "on", "truthy", ""])
def test_mining_flag_only_exact_true_enables(monkeypatch, flag):
    """Anything other than 'true' (case-insensitive) must NOT enable mining."""
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", flag)
    monkeypatch.setenv("DB_BACKEND", "local")
    assert telemetry.mining_enabled() is False


# ---------------------------------------------------------------------------
# generate_query_id()
# ---------------------------------------------------------------------------


def test_generate_query_id_format():
    qid = telemetry.generate_query_id()
    assert re.match(r"^qry_[0-9a-f]{12}$", qid), qid


def test_generate_query_id_unique():
    ids = {telemetry.generate_query_id() for _ in range(1000)}
    assert len(ids) == 1000


# ---------------------------------------------------------------------------
# TELEMETRY_PG_SCHEMA shape + migration parity
# ---------------------------------------------------------------------------


def test_schema_contains_all_create_if_not_exists_statements():
    schema = telemetry.TELEMETRY_PG_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS knowledge.retrieval_telemetry" in schema
    index_creates = re.findall(r"CREATE INDEX IF NOT EXISTS", schema)
    assert len(index_creates) == 3
    # Four CREATE ... IF NOT EXISTS statements total (1 table + 3 indexes).
    assert len(re.findall(r"CREATE (?:TABLE|INDEX) IF NOT EXISTS", schema)) == 4


def test_schema_matches_migration_file():
    """The DDL embedded in code must be byte-for-byte identical to migration 005."""
    migration = Path(__file__).resolve().parents[1] / "migrations" / "005_retrieval_telemetry.sql"
    text = migration.read_text()
    # Strip the leading comment/documentation header; the SQL begins at the
    # first CREATE statement.
    sql_start = text.index("CREATE TABLE IF NOT EXISTS")
    migration_ddl = text[sql_start:].strip()
    code_ddl = telemetry.TELEMETRY_PG_SCHEMA.strip()
    assert migration_ddl == code_ddl


def test_ensure_telemetry_schema_executes_each_statement_individually():
    """psycopg2 only runs the first statement of a multi-statement string, so
    ensure_telemetry_schema must call cursor.execute once per non-empty base
    statement (1 table + 3 indexes = 4) — a string test alone can't catch this.

    Phase 2 (Issue #5) appends one more statement (the notes DDL); that trailing
    statement is asserted separately in test_ensure_telemetry_schema_applies_notes_ddl,
    so here we only pin the 4 base statements as the executed prefix.
    """

    class _FakeCursor:
        def __init__(self):
            self.executed: list[str] = []

        def execute(self, sql):
            self.executed.append(sql)

        def close(self):
            pass

    class _FakeConn:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    expected = [s.strip() for s in telemetry.TELEMETRY_PG_SCHEMA.split(";") if s.strip()]
    assert len(expected) == 4
    cursor = _FakeCursor()
    telemetry.ensure_telemetry_schema(_FakeConn(cursor))

    # The 4 base statements are executed first, each as a single, semicolon-free
    # DDL statement (the multi-statement bug-fix invariant).
    assert cursor.executed[:4] == expected
    assert all(";" not in stmt for stmt in cursor.executed)


# ---------------------------------------------------------------------------
# write_retrieval_telemetry_async() backend guard (Fix 3)
# ---------------------------------------------------------------------------


def test_write_async_noop_for_non_postgres_db():
    """Non-PostgreSQL db objects must be a no-op (returns None), never AttributeError."""

    class FakeSqliteDb:  # not a LocalPostgresClient
        pass

    result = telemetry.write_retrieval_telemetry_async(
        query_id="qry_abc123abc123",
        query_text="q",
        topic=None,
        search_mode="fts",
        retrieved_document_ids=["a"],
        result_count=1,
        session_id=None,
        parent_query_id=None,
        required_requery=False,
        caller_agent=None,
        model_version="0.0.0",
        db=FakeSqliteDb(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# _write_row() swallows exceptions (Fix: best-effort telemetry)
# ---------------------------------------------------------------------------


def test_write_row_swallows_connection_error(caplog):
    """A bad connection must be logged at WARNING, never raised."""
    import logging

    with caplog.at_level(logging.WARNING, logger="lore.telemetry"):
        # Unroutable conn_params → psycopg2.connect raises → swallowed.
        telemetry._write_row(
            conn_params={
                "host": "127.0.0.1",
                "port": 1,  # nothing listening
                "dbname": "nope",
                "user": "nope",
                "password": "nope",
            },
            query_id="qry_dead00beef00",
            query_text="q",
            topic=None,
            search_mode="fts",
            retrieved_document_ids=[],
            result_count=0,
            session_id=None,
            parent_query_id=None,
            required_requery=False,
            caller_agent=None,
            model_version="0.0.0",
        )
    assert any("retrieval telemetry" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# handle_kb_search integration with telemetry (mining on/off)
# ---------------------------------------------------------------------------


class _FakeBuilder:
    """Minimal fluent stand-in for db.table(...).select(...).or_(...).limit().execute()."""

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
        from lore.db_client import QueryResult

        return QueryResult(data=self._rows)


class _FakeDb:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeBuilder(self._rows)


def _force_lexical_backend(monkeypatch):
    """Unset DB_BACKEND so handle_kb_search routes to the legacy lexical path."""
    monkeypatch.setenv("DB_BACKEND", "")


def test_kb_search_mining_disabled_has_no_query_id(monkeypatch):
    import lore.server as srv

    _force_lexical_backend(monkeypatch)
    monkeypatch.delenv("LORE_HARD_NEGATIVE_MINING", raising=False)

    called = {"n": 0}

    def _spy(**_kwargs):
        called["n"] += 1
        return None

    monkeypatch.setattr(srv.telemetry, "write_retrieval_telemetry_async", _spy)
    monkeypatch.setattr(srv, "db", _FakeDb([{"kb_id": "1", "title": "t", "topic": "x"}]))

    resp = srv.handle_kb_search("hello")
    assert resp["ok"] is True
    assert "query_id" not in resp["data"]
    assert called["n"] == 0


def test_kb_search_mining_enabled_sets_query_id_and_spies_args(monkeypatch):
    import lore.search as _search
    import lore.server as srv

    # mining_enabled() requires a PostgreSQL backend. With DB_BACKEND=local and
    # requested_mode "fts" (the default), handle_kb_search routes through the
    # PostgreSQL FTS-only path, which we stub so no real DB is needed.
    monkeypatch.setenv("DB_BACKEND", "local")
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)

    rows = [
        {"kb_id": "kb-1", "title": "a", "topic": "t"},
        {"kb_id": "kb-2", "title": "b", "topic": "t"},
    ]
    monkeypatch.setattr(_search, "fts_search_postgres", lambda *a, **k: list(rows))

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(srv.telemetry, "write_retrieval_telemetry_async", _spy)
    monkeypatch.setattr(srv, "db", _FakeDb(rows))

    resp = srv.handle_kb_search(
        "needle",
        topic="t",
        session_id="sess-9",
        parent_query_id="qry_parent00000",
        required_requery=True,
        caller_agent="tester",
    )

    assert resp["ok"] is True
    qid = resp["data"]["query_id"]
    assert re.match(r"^qry_[0-9a-f]{12}$", qid)

    # Spy received the same query_id and the forwarded telemetry context.
    assert captured["query_id"] == qid
    assert captured["query_text"] == "needle"
    assert captured["topic"] == "t"
    assert captured["session_id"] == "sess-9"
    assert captured["parent_query_id"] == "qry_parent00000"
    assert captured["required_requery"] is True
    assert captured["caller_agent"] == "tester"
    assert captured["retrieved_document_ids"] == ["kb-1", "kb-2"]
    assert captured["result_count"] == 2
    assert captured["db"] is srv.db
    # model_version is the live lore version, not None.
    assert captured["model_version"]


# ===========================================================================
# Phase 2 (Issue #5): notes migration parity, schema DDL, read/update helpers,
# and the three analysis-tool handlers.
# ===========================================================================


# ---------------------------------------------------------------------------
# Migration 006 parity + ensure_telemetry_schema applies the notes DDL
# ---------------------------------------------------------------------------


def _normalize_sql(text: str) -> str:
    """Collapse whitespace and drop a trailing semicolon for byte-comparison."""
    return " ".join(text.split()).rstrip(";").strip()


def test_notes_migration_matches_constant():
    """migration 006 SQL (sans comments) must match TELEMETRY_NOTES_DDL."""
    migration = Path(__file__).resolve().parents[1] / "migrations" / "006_telemetry_notes.sql"
    text = migration.read_text()
    sql_start = text.index("ALTER TABLE")
    migration_ddl = _normalize_sql(text[sql_start:])
    assert migration_ddl == _normalize_sql(telemetry.TELEMETRY_NOTES_DDL)
    # The constant is the exact single-line ALTER ... statement (with trailing ;).
    assert telemetry.TELEMETRY_NOTES_DDL == (
        "ALTER TABLE knowledge.retrieval_telemetry ADD COLUMN IF NOT EXISTS notes TEXT;"
    )


def test_ensure_telemetry_schema_applies_notes_ddl():
    """ensure_telemetry_schema runs the 4 base statements + the notes DDL last."""

    class _FakeCursor:
        def __init__(self):
            self.executed: list[str] = []

        def execute(self, sql):
            self.executed.append(sql)

        def close(self):
            pass

    class _FakeConn:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    cursor = _FakeCursor()
    telemetry.ensure_telemetry_schema(_FakeConn(cursor))

    base = [s.strip() for s in telemetry.TELEMETRY_PG_SCHEMA.split(";") if s.strip()]
    # 1 table + 3 indexes + 1 notes column + 1 query_embedding column (Phase 4a).
    assert len(cursor.executed) == len(base) + 2 == 6
    # ensure_telemetry_schema strips the trailing ';' before executing so
    # psycopg2 receives a single semicolon-free statement (multi-statement invariant).
    # The notes DDL precedes the query_embedding DDL; the latter is applied last.
    assert cursor.executed[-2] == telemetry.TELEMETRY_NOTES_DDL.rstrip(";").strip()
    assert cursor.executed[-1] == telemetry.TELEMETRY_QUERY_EMBEDDING_DDL.rstrip(";").strip()
    assert all(";" not in stmt for stmt in cursor.executed)


# ---------------------------------------------------------------------------
# Fake psycopg2 connection plumbing for read/update helper tests
# ---------------------------------------------------------------------------


class _FakeRWCursor:
    """Records executed SQL/params; returns canned fetch results and rowcount."""

    def __init__(self, *, fetchall=None, fetchone=None, rowcount=0):
        self._fetchall = fetchall if fetchall is not None else []
        self._fetchone = fetchone
        self.rowcount = rowcount
        self.executed: list[tuple] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall

    def fetchone(self):
        return self._fetchone

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.autocommit = False
        self.closed = False

    def cursor(self, *_a, **_k):
        return self._cursor

    def close(self):
        self.closed = True


@pytest.fixture
def pg_db():
    """A LocalPostgresClient instance (no live connection; constructor is inert)."""
    from lore.db_client import LocalPostgresClient

    return LocalPostgresClient(host="h", port=1, database="d", user="u", password="p")


def _patch_connect(monkeypatch, conn):
    """Make psycopg2.connect return our fake connection."""
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", lambda **_k: conn)


# ---------------------------------------------------------------------------
# Backend guard: non-PostgreSQL db => None from all three helpers
# ---------------------------------------------------------------------------


class _FakeSqliteDb:  # not a LocalPostgresClient
    pass


def test_update_feedback_none_for_non_pg():
    assert (
        telemetry.update_retrieval_feedback(
            query_id="qry_x", user_feedback_score=1, notes=None, db=_FakeSqliteDb()
        )
        is None
    )


def test_fetch_telemetry_none_for_non_pg():
    assert (
        telemetry.fetch_retrieval_telemetry(
            query_id=None, session_id="s", topic=None, limit=50, db=_FakeSqliteDb()
        )
        is None
    )


def test_fetch_stats_none_for_non_pg():
    assert telemetry.fetch_telemetry_stats(session_id=None, topic=None, db=_FakeSqliteDb()) is None


# ---------------------------------------------------------------------------
# update_retrieval_feedback: truncation + rowcount passthrough
# ---------------------------------------------------------------------------


def test_update_feedback_truncates_notes(monkeypatch, pg_db):
    cur = _FakeRWCursor(rowcount=1)
    _patch_connect(monkeypatch, _FakeConn(cur))

    long_notes = "x" * (telemetry.MAX_NOTES_LEN + 500)
    rc = telemetry.update_retrieval_feedback(
        query_id="qry_abc", user_feedback_score=None, notes=long_notes, db=pg_db
    )
    assert rc == 1
    # Second positional param is the (truncated) notes value.
    _sql, params = cur.executed[0]
    assert len(params[1]) == telemetry.MAX_NOTES_LEN
    assert params[0] is None  # score left untouched
    assert params[2] == "qry_abc"


def test_update_feedback_returns_rowcount_zero_when_not_found(monkeypatch, pg_db):
    cur = _FakeRWCursor(rowcount=0)
    _patch_connect(monkeypatch, _FakeConn(cur))
    rc = telemetry.update_retrieval_feedback(
        query_id="qry_missing", user_feedback_score=5, notes=None, db=pg_db
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# clamp_read_limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (9999, 500),  # above MAX_READ_LIMIT -> clamped
        ("x", 50),  # non-int -> DEFAULT_READ_LIMIT
        (0, 1),  # below 1 -> clamped to 1
        (-3, 1),
        (None, 50),
        (50, 50),
        ("100", 100),  # numeric string coerced
    ],
)
def test_clamp_read_limit(value, expected):
    assert telemetry.clamp_read_limit(value) == expected


def test_fetch_telemetry_clamps_limit_in_sql(monkeypatch, pg_db):
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_retrieval_telemetry(
        query_id=None, session_id="sess", topic=None, limit=9999, db=pg_db
    )
    _sql, params = cur.executed[0]
    assert params == ("sess", 500)


def test_fetch_telemetry_query_id_precedence(monkeypatch, pg_db):
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_retrieval_telemetry(
        query_id="qry_1", session_id="sess", topic="t", limit=50, db=pg_db
    )
    sql, params = cur.executed[0]
    assert "WHERE query_id = %s" in sql
    assert params == ("qry_1",)


def test_fetch_telemetry_topic_branch(monkeypatch, pg_db):
    """topic selector (no query_id/session_id) filters by topic, newest-first."""
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_retrieval_telemetry(
        query_id=None, session_id=None, topic="qa_test", limit=10, db=pg_db
    )
    sql, params = cur.executed[0]
    assert "WHERE topic = %s" in sql
    assert "ORDER BY created_at DESC LIMIT" in sql
    assert params == ("qa_test", 10)


def test_fetch_telemetry_all_none_branch(monkeypatch, pg_db):
    """All selectors None => recent rows: no WHERE clause, newest-first LIMIT."""
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_retrieval_telemetry(
        query_id=None, session_id=None, topic=None, limit=10, db=pg_db
    )
    sql, params = cur.executed[0]
    assert "WHERE" not in sql
    assert "ORDER BY created_at DESC LIMIT" in sql
    assert params == (10,)


def test_update_feedback_score_only(monkeypatch, pg_db):
    """COALESCE partial update with score only leaves notes param as None."""
    cur = _FakeRWCursor(rowcount=1)
    _patch_connect(monkeypatch, _FakeConn(cur))
    rc = telemetry.update_retrieval_feedback(
        query_id="qry_x", user_feedback_score=3, notes=None, db=pg_db
    )
    assert rc == 1
    _sql, params = cur.executed[0]
    assert params == (3, None, "qry_x")


def test_write_row_swallows_execute_error(caplog):
    """A cursor.execute() failure after a successful connect must be logged at
    WARNING, never raised (best-effort telemetry)."""
    import logging

    import psycopg2

    class _BoomCursor:
        def execute(self, *_a, **_k):
            raise psycopg2.IntegrityError("duplicate key")

        def close(self):
            pass

    class _BoomConn:
        autocommit = False

        def set_client_encoding(self, *_a, **_k):
            pass

        def cursor(self):
            return _BoomCursor()

        def close(self):
            pass

    with caplog.at_level(logging.WARNING, logger="lore.telemetry"):
        with mock.patch.object(psycopg2, "connect", lambda **_k: _BoomConn()):
            telemetry._write_row(
                conn_params={
                    "host": "h",
                    "port": 5432,
                    "dbname": "d",
                    "user": "u",
                    "password": "p",
                },
                query_id="qry_dup0000000",
                query_text="q",
                topic=None,
                search_mode="fts",
                retrieved_document_ids=[],
                result_count=0,
                session_id=None,
                parent_query_id=None,
                required_requery=False,
                caller_agent=None,
                model_version="0.0.0",
            )
    assert any("retrieval telemetry" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Handler: handle_log_retrieval_feedback
# ---------------------------------------------------------------------------


def _enable_mining(monkeypatch):
    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "true")
    monkeypatch.setenv("DB_BACKEND", "local")


def test_log_feedback_noop_when_both_args_none(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)

    called = {"n": 0}

    def _spy(**_k):
        called["n"] += 1
        return 1

    monkeypatch.setattr(srv.telemetry, "update_retrieval_feedback", _spy)
    resp = srv.handle_log_retrieval_feedback("qry_x")
    assert resp["ok"] is True
    assert resp["data"]["noop"] is True
    assert resp["data"]["query_id"] == "qry_x"
    assert called["n"] == 0  # no DB round-trip


def test_log_feedback_mining_off_skipped(monkeypatch):
    import lore.server as srv

    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")
    resp = srv.handle_log_retrieval_feedback("qry_x", user_feedback_score=3)
    assert resp["ok"] is True
    assert resp["data"]["skipped"] is True


def test_log_feedback_not_found(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "update_retrieval_feedback", lambda **_k: 0)
    resp = srv.handle_log_retrieval_feedback("qry_missing", user_feedback_score=3)
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_log_feedback_success(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "update_retrieval_feedback", lambda **_k: 1)
    resp = srv.handle_log_retrieval_feedback("qry_ok", user_feedback_score=4, notes="good")
    assert resp["ok"] is True
    assert resp["data"]["updated"] == 1
    assert resp["data"]["query_id"] == "qry_ok"


def test_log_feedback_backend_unavailable_returns_error(monkeypatch):
    """When update_retrieval_feedback returns None (non-PG backend), the handler
    returns an UNEXPECTED_EXCEPTION error, not a false-success.

    Regression guard for Fix 1 (commit f01a1c2): None == 0 is False, so a single
    rowcount check would let None fall through to {updated: None}. The explicit
    `rows_affected is None` branch must short-circuit to an error first.
    """
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "update_retrieval_feedback", lambda **_k: None)
    resp = srv.handle_log_retrieval_feedback("qry_x", user_feedback_score=3)
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# Handler: handle_get_retrieval_telemetry / handle_get_telemetry_stats
# ---------------------------------------------------------------------------


def test_get_telemetry_empty_session_is_ok_not_error(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "fetch_retrieval_telemetry", lambda **_k: [])
    resp = srv.handle_get_retrieval_telemetry(session_id="sess-empty")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 0
    assert resp["data"]["rows"] == []


def test_get_telemetry_mining_off_skipped(monkeypatch):
    import lore.server as srv

    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")
    resp = srv.handle_get_retrieval_telemetry(session_id="sess")
    assert resp["ok"] is True
    assert resp["data"]["skipped"] is True


def test_get_telemetry_stats_mining_off_skipped(monkeypatch):
    import lore.server as srv

    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")
    resp = srv.handle_get_telemetry_stats()
    assert resp["ok"] is True
    assert resp["data"]["skipped"] is True


def test_get_telemetry_stats_success(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    fake_stats = {"total": 3, "with_feedback": 1, "requeries": 0, "with_notes": 1}
    monkeypatch.setattr(srv.telemetry, "fetch_telemetry_stats", lambda **_k: fake_stats)
    resp = srv.handle_get_telemetry_stats(topic="t")
    assert resp["ok"] is True
    assert resp["data"]["stats"] == fake_stats


# ===========================================================================
# Phase 3 (Issue #5): hard negative pairs — deterministic pair_id, schema DDL,
# migration parity, backend guards, limit clamping, and the two handlers.
# ===========================================================================


# ---------------------------------------------------------------------------
# Deterministic pair_id (Correction 1: SHA-256, not secrets.token_hex)
# ---------------------------------------------------------------------------


def test_hard_negative_pair_id_is_deterministic():
    """Same (query_text, doc_id) must always yield the same 32-hex pair_id."""
    a = telemetry.hard_negative_pair_id("how do I X?", "kb-42")
    b = telemetry.hard_negative_pair_id("how do I X?", "kb-42")
    assert a == b
    assert re.match(r"^[0-9a-f]{32}$", a), a


def test_hard_negative_pair_id_differs_by_input():
    """Different inputs must yield different pair_ids (no collisions on basics)."""
    base = telemetry.hard_negative_pair_id("q", "d")
    assert base != telemetry.hard_negative_pair_id("q", "d2")
    assert base != telemetry.hard_negative_pair_id("q2", "d")


def test_hard_negative_pair_id_matches_sha256_contract():
    """pair_id must be SHA-256 of 'query:doc' truncated to 32 hex chars."""
    import hashlib

    expected = hashlib.sha256(b"q:d").hexdigest()[:32]
    assert telemetry.hard_negative_pair_id("q", "d") == expected


# ---------------------------------------------------------------------------
# HARD_NEGATIVE_PG_SCHEMA shape + migration 007 parity
# ---------------------------------------------------------------------------


def test_hn_schema_contains_table_and_four_indexes():
    schema = telemetry.HARD_NEGATIVE_PG_SCHEMA
    assert "CREATE TABLE IF NOT EXISTS knowledge.hard_negative_pairs" in schema
    assert "ON DELETE RESTRICT" in schema  # Correction 3
    index_creates = re.findall(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS", schema)
    assert len(index_creates) == 4
    # 1 table + 4 indexes = 5 CREATE statements total.
    statements = [s.strip() for s in schema.split(";") if s.strip()]
    assert len(statements) == 5


def test_hn_schema_matches_migration_file():
    """The DDL embedded in code must be byte-for-byte identical to migration 007."""
    migration = (
        Path(__file__).resolve().parents[1] / "migrations" / "007_hard_negative_pairs.sql"
    )
    text = migration.read_text()
    sql_start = text.index("CREATE TABLE IF NOT EXISTS")
    migration_ddl = text[sql_start:].strip()
    code_ddl = telemetry.HARD_NEGATIVE_PG_SCHEMA.strip()
    assert migration_ddl == code_ddl


def test_ensure_hard_negative_schema_executes_each_statement_individually():
    """psycopg2 only runs the first statement of a multi-statement string, so
    ensure_hard_negative_schema must execute each of the 5 statements (1 table +
    4 indexes) individually, with no embedded semicolons."""

    class _FakeCursor:
        def __init__(self):
            self.executed: list[str] = []

        def execute(self, sql):
            self.executed.append(sql)

        def close(self):
            pass

    class _FakeConn:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

    expected = [s.strip() for s in telemetry.HARD_NEGATIVE_PG_SCHEMA.split(";") if s.strip()]
    assert len(expected) == 5
    cursor = _FakeCursor()
    telemetry.ensure_hard_negative_schema(_FakeConn(cursor))

    assert cursor.executed == expected
    assert all(";" not in stmt for stmt in cursor.executed)


def test_ensure_hard_negative_schema_swallows_errors(caplog):
    """A cursor.execute() failure must be logged at WARNING, never raised."""
    import logging

    class _BoomCursor:
        def execute(self, *_a, **_k):
            raise RuntimeError("boom")

        def close(self):
            pass

    class _BoomConn:
        def cursor(self):
            return _BoomCursor()

    with caplog.at_level(logging.WARNING, logger="lore.telemetry"):
        telemetry.ensure_hard_negative_schema(_BoomConn())
    assert any("ensure_hard_negative_schema failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Backend guards: non-PostgreSQL db => None from refresh/fetch
# ---------------------------------------------------------------------------


def test_refresh_hard_negative_pairs_none_for_non_pg():
    assert (
        telemetry.refresh_hard_negative_pairs(since=None, dry_run=False, db=_FakeSqliteDb())
        is None
    )


def test_fetch_hard_negatives_none_for_non_pg():
    assert (
        telemetry.fetch_hard_negatives(
            signal_type=None, limit=100, doc_id=None, query_text_like=None, db=_FakeSqliteDb()
        )
        is None
    )


def test_refresh_dry_run_rolls_back_not_commits(monkeypatch, pg_db):
    """dry_run=True must ROLLBACK the transaction, never COMMIT."""
    rolled_back = []
    committed = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **kw):
            pass

        def executemany(self, *a, **kw):
            pass

        def fetchone(self):
            return (0,)  # for the COUNT(*) queries

        def fetchall(self):
            return []

    class FakeConn:
        autocommit = True  # set to False by the function

        def set_client_encoding(self, *a, **kw):
            pass

        def cursor(self, **kw):
            return FakeCursor()

        def rollback(self):
            rolled_back.append(1)

        def commit(self):
            committed.append(1)

        def close(self):
            pass

    _patch_connect(monkeypatch, FakeConn())

    result = telemetry.refresh_hard_negative_pairs(since=None, dry_run=True, db=pg_db)
    assert result["dry_run"] is True
    assert len(rolled_back) == 1, "rollback() must be called exactly once"
    assert len(committed) == 0, "commit() must never be called on dry_run"


# ---------------------------------------------------------------------------
# fetch_hard_negatives: filter clauses + parameterized ILIKE
# ---------------------------------------------------------------------------


def test_fetch_hard_negatives_no_filters(monkeypatch, pg_db):
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_hard_negatives(
        signal_type=None, limit=50, doc_id=None, query_text_like=None, db=pg_db
    )
    sql, params = cur.executed[0]
    assert "WHERE" not in sql
    assert "ORDER BY occurrence_count DESC, last_seen_at DESC LIMIT %s" in sql
    assert params == (50,)


def test_fetch_hard_negatives_all_filters_parameterized(monkeypatch, pg_db):
    cur = _FakeRWCursor(fetchall=[])
    _patch_connect(monkeypatch, _FakeConn(cur))
    telemetry.fetch_hard_negatives(
        signal_type="explicit", limit=10, doc_id="kb-7", query_text_like="needle", db=pg_db
    )
    sql, params = cur.executed[0]
    assert "signal_type = %s" in sql
    assert "doc_id = %s" in sql
    assert "query_text ILIKE %s" in sql
    # ILIKE value is wrapped in %...% as a bind param (no f-string interpolation).
    assert params == ("explicit", "kb-7", "%needle%", 10)


# ---------------------------------------------------------------------------
# Handler: handle_refresh_hard_negatives
# ---------------------------------------------------------------------------


def test_refresh_hard_negatives_mining_off_skipped(monkeypatch):
    import lore.server as srv

    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")
    resp = srv.handle_refresh_hard_negatives()
    assert resp["ok"] is True
    assert resp["data"]["skipped"] is True
    assert resp["data"]["reason"] == "mining_disabled"


def test_refresh_hard_negatives_backend_unavailable_returns_error(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "refresh_hard_negative_pairs", lambda **_k: None)
    resp = srv.handle_refresh_hard_negatives()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_refresh_hard_negatives_success(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    fake = {
        "inserted": 2, "updated": 1, "total_pairs": 3,
        "processed_telemetry_rows": 4, "since": "all", "dry_run": False,
    }
    monkeypatch.setattr(srv.telemetry, "refresh_hard_negative_pairs", lambda **_k: fake)
    resp = srv.handle_refresh_hard_negatives()
    assert resp["ok"] is True
    assert resp["data"] == fake
    assert "Processed 4 telemetry rows" in resp["message"]


def test_refresh_hard_negatives_dry_run_message(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    fake = {
        "inserted": 0, "updated": 0, "total_pairs": 0,
        "processed_telemetry_rows": 0, "since": "all", "dry_run": True,
    }
    monkeypatch.setattr(srv.telemetry, "refresh_hard_negative_pairs", lambda **_k: fake)
    resp = srv.handle_refresh_hard_negatives(dry_run=True)
    assert resp["ok"] is True
    assert resp["message"].startswith("[dry-run] ")


# ---------------------------------------------------------------------------
# Handler: handle_get_hard_negatives
# ---------------------------------------------------------------------------


def test_get_hard_negatives_invalid_signal_type(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    resp = srv.handle_get_hard_negatives(signal_type="bogus")
    assert resp["ok"] is False
    assert resp["error"] == "invalid_input"


def test_get_hard_negatives_does_not_gate_on_mining(monkeypatch):
    """get_hard_negatives reads historical pairs even when mining is OFF."""
    import lore.server as srv

    monkeypatch.setenv("LORE_HARD_NEGATIVE_MINING", "false")
    monkeypatch.setenv("DB_BACKEND", "local")

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return [{"pair_id": "abc", "doc_id": "kb-1"}]

    monkeypatch.setattr(srv.telemetry, "fetch_hard_negatives", _spy)
    resp = srv.handle_get_hard_negatives(signal_type="explicit")
    # Mining is OFF but the read still succeeds (no skipped flag).
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1
    assert "skipped" not in resp["data"]
    assert captured["signal_type"] == "explicit"


def test_get_hard_negatives_all_maps_to_none_filter(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(srv.telemetry, "fetch_hard_negatives", _spy)
    resp = srv.handle_get_hard_negatives(signal_type="all")
    assert resp["ok"] is True
    assert captured["signal_type"] is None  # "all" => no filter


@pytest.mark.parametrize(
    "given,expected",
    [
        (-3, 1),       # truthy but below 1 -> floored to 1
        (1, 1),        # exact lower bound passes through
        (1001, 1000),  # above MAX_HN_LIMIT -> clamped
        (9999, 1000),  # well above ceiling -> clamped
        (None, 100),   # falsy -> DEFAULT_HN_LIMIT
        (0, 100),      # 0 is falsy -> DEFAULT_HN_LIMIT (handler uses `if limit`)
        (50, 50),
    ],
)
def test_get_hard_negatives_limit_clamping(monkeypatch, given, expected):
    """Limit is clamped to [1, MAX_HN_LIMIT]; falsy (None/0) falls back to default.

    Note: the handler's `int(limit) if limit else DEFAULT_HN_LIMIT` treats 0 as
    falsy, so limit=0 yields the default (100), not the floor (1). A negative
    limit is truthy and is floored to 1 by the surrounding max(1, ...).
    """
    import lore.server as srv

    _enable_mining(monkeypatch)
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(srv.telemetry, "fetch_hard_negatives", _spy)
    srv.handle_get_hard_negatives(limit=given)
    assert captured["limit"] == expected


def test_get_hard_negatives_backend_unavailable_returns_error(monkeypatch):
    import lore.server as srv

    _enable_mining(monkeypatch)
    monkeypatch.setattr(srv.telemetry, "fetch_hard_negatives", lambda **_k: None)
    resp = srv.handle_get_hard_negatives()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# fetch_hard_negatives: graceful empty result when pairs table not yet created
# ---------------------------------------------------------------------------


def test_fetch_hard_negatives_missing_table_returns_empty(monkeypatch, pg_db):
    """When the hard_negative_pairs table does not exist (UndefinedTable), the
    function must return [] rather than propagating the raw psycopg2 exception."""
    import psycopg2
    import psycopg2.errors

    class _UndefinedTableCursor:
        def execute(self, *_a, **_k):
            raise psycopg2.errors.UndefinedTable(
                'relation "knowledge.hard_negative_pairs" does not exist'
            )

        def fetchall(self):  # pragma: no cover — never reached
            return []

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class _UndefinedTableConn:
        autocommit = False

        def set_client_encoding(self, *_a, **_k):
            pass

        def cursor(self, **_k):
            return _UndefinedTableCursor()

        def close(self):
            pass

    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _UndefinedTableConn())
    result = telemetry.fetch_hard_negatives(
        signal_type=None, limit=100, doc_id=None, query_text_like=None, db=pg_db
    )
    assert result == []


# ===========================================================================
# Phase 4 (Issue #5): query embedding storage + re-ranking.
#   - reranking_enabled() flag (independent of mining)
#   - migration 008 parity (column added; HNSW index NOT auto-applied)
#   - _write_row threads query_embedding through the INSERT
#   - backfill_query_embeddings (dry-run, limit, encode-error tolerance)
#   - fetch_reranking_bad_docs best-effort empty-on-error
#   - server._apply_reranking_penalties reorder semantics
# ===========================================================================


# ---------------------------------------------------------------------------
# reranking_enabled(): decoupled from mining_enabled()
# ---------------------------------------------------------------------------


def test_reranking_enabled_independent_of_mining(monkeypatch):
    """LORE_RERANKING_ENABLED enables re-ranking without LORE_HARD_NEGATIVE_MINING.

    The two flags are independent: re-ranking reads hard_negative_pairs (which
    may be populated by another process) and must not require mining to be on.
    """
    monkeypatch.setenv("LORE_RERANKING_ENABLED", "true")
    monkeypatch.delenv("LORE_HARD_NEGATIVE_MINING", raising=False)
    monkeypatch.setenv("DB_BACKEND", "local")
    assert telemetry.reranking_enabled() is True
    # Mining is independently False (its flag is unset).
    assert telemetry.mining_enabled() is False


def test_reranking_disabled_by_default(monkeypatch):
    """No env vars set => reranking_enabled() is False."""
    monkeypatch.delenv("LORE_RERANKING_ENABLED", raising=False)
    assert telemetry.reranking_enabled() is False


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "Yes", "yes"])
def test_reranking_enabled_accepts_truthy_flags(monkeypatch, flag):
    monkeypatch.setenv("LORE_RERANKING_ENABLED", flag)
    assert telemetry.reranking_enabled() is True


# ---------------------------------------------------------------------------
# Migration 008 parity: column added, HNSW index NOT auto-applied
# ---------------------------------------------------------------------------


def test_migration_008_parity():
    """migration 008 adds the query_embedding column and must NOT create the
    HNSW index (it is deferred to the backfill_query_embeddings tool)."""
    migration = (
        Path(__file__).resolve().parents[1] / "migrations" / "008_query_embedding.sql"
    )
    text = migration.read_text()
    assert "ADD COLUMN IF NOT EXISTS query_embedding halfvec(384)" in text
    # The HNSW index is documented only as a comment; the executable DDL (the
    # ALTER TABLE) must not contain an active CREATE INDEX statement.
    sql_start = text.index("ALTER TABLE")
    active_sql = "\n".join(
        line for line in text[sql_start:].splitlines() if not line.strip().startswith("--")
    )
    assert "CREATE INDEX" not in active_sql.upper()
    # The constant in code matches the column-add intent.
    assert "ADD COLUMN IF NOT EXISTS query_embedding halfvec(384)" in (
        telemetry.TELEMETRY_QUERY_EMBEDDING_DDL
    )


# ---------------------------------------------------------------------------
# _write_row threads query_embedding through the INSERT
# ---------------------------------------------------------------------------


class _CaptureCursor:
    """Records the SQL + params of the most recent execute()."""

    def __init__(self):
        self.sql = None
        self.params = None

    def execute(self, sql, params=None):
        self.sql = sql
        self.params = params

    def close(self):
        pass


class _CaptureConn:
    autocommit = False

    def __init__(self, cursor):
        self._cursor = cursor

    def set_client_encoding(self, *_a, **_k):
        pass

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_write_row_with_embedding(monkeypatch):
    """_write_row includes query_embedding in the INSERT and passes the value."""
    import psycopg2

    cur = _CaptureCursor()
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _CaptureConn(cur))

    emb = [0.1] * 384
    telemetry._write_row(
        conn_params={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
        query_id="qry_emb000000",
        query_text="q",
        topic=None,
        search_mode="fts",
        retrieved_document_ids=["a"],
        result_count=1,
        session_id=None,
        parent_query_id=None,
        required_requery=False,
        caller_agent=None,
        model_version="0.0.0",
        query_embedding=emb,
    )
    assert "query_embedding" in cur.sql
    # The embedding is the last positional param.
    assert cur.params[-1] == emb


def test_write_row_without_embedding(monkeypatch):
    """_write_row defaults query_embedding to None and passes None through."""
    import psycopg2

    cur = _CaptureCursor()
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _CaptureConn(cur))

    telemetry._write_row(
        conn_params={"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"},
        query_id="qry_noemb00000",
        query_text="q",
        topic=None,
        search_mode="fts",
        retrieved_document_ids=[],
        result_count=0,
        session_id=None,
        parent_query_id=None,
        required_requery=False,
        caller_agent=None,
        model_version="0.0.0",
    )
    assert "query_embedding" in cur.sql
    assert cur.params[-1] is None


# ---------------------------------------------------------------------------
# backfill_query_embeddings: dry-run, limit, encode-error tolerance
# ---------------------------------------------------------------------------


class _BackfillCursor:
    """Fakes the SELECT-batch / UPDATE flow of backfill_query_embeddings.

    The first SELECT returns ``rows`` (as (query_id, query_text) tuples); any
    subsequent SELECT returns [] so the LIMIT/OFFSET loop terminates. UPDATE
    statements are recorded in ``updates``.
    """

    def __init__(self, rows):
        self._rows = rows
        self._select_calls = 0
        self.updates: list[tuple] = []

    def execute(self, sql, params=None):
        upper = sql.strip().upper()
        if upper.startswith("SELECT"):
            self._select_calls += 1
            self._last = self._rows if self._select_calls == 1 else []
        elif upper.startswith("UPDATE"):
            self.updates.append(params)
        else:
            self._last = []

    def fetchall(self):
        return self._last


class _BackfillConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        pass


def test_backfill_dry_run_does_not_write(monkeypatch, pg_db):
    """dry_run=True computes embeddings but issues no UPDATE and reports updated=0."""
    import psycopg2

    cur = _BackfillCursor([("qry_1", "text one"), ("qry_2", "text two")])
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _BackfillConn(cur))

    result = telemetry.backfill_query_embeddings(
        pg_db,
        encode_fn=lambda _t: [0.0] * 384,
        batch_size=32,
        limit=1000,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["processed"] == 2
    assert result["updated"] == 0
    assert result["index_built"] is False
    assert cur.updates == []  # no UPDATE executed under dry_run


def test_backfill_respects_limit(monkeypatch, pg_db):
    """With limit=10, no more than 10 rows are processed even if more exist."""
    import psycopg2

    # Return a full 10-row batch on the first SELECT; the loop should stop once
    # remaining hits 0 (limit=10, batch_size=10), never issuing a second SELECT.
    rows = [(f"qry_{i}", f"text {i}") for i in range(10)]
    cur = _BackfillCursor(rows)
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _BackfillConn(cur))

    result = telemetry.backfill_query_embeddings(
        pg_db,
        encode_fn=lambda _t: [0.0] * 384,
        batch_size=10,
        limit=10,
        dry_run=False,
    )
    assert result["processed"] <= 10
    assert result["processed"] == 10
    assert result["updated"] == 10


def test_backfill_skips_encode_error(monkeypatch, pg_db):
    """An encode_fn failure on one row is counted as skipped, not a crash."""
    import psycopg2

    cur = _BackfillCursor([("qry_1", "ok"), ("qry_2", "boom")])
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _BackfillConn(cur))

    calls = {"n": 0}

    def _encode(_text):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("encode failed")
        return [0.0] * 384

    result = telemetry.backfill_query_embeddings(
        pg_db,
        encode_fn=_encode,
        batch_size=32,
        limit=1000,
        dry_run=False,
    )
    assert result["processed"] == 2
    assert result["updated"] == 1
    assert result["skipped"] == 1


class _LimitRecordingCursor:
    """Records the LIMIT value of every SELECT so we can assert it is never 0.

    Always returns [] from fetchall(), so the backfill loop terminates after a
    single SELECT regardless of batch_size. The point of this fake is purely to
    capture the LIMIT bind parameter the loop computes.
    """

    def __init__(self):
        self.select_limits: list = []

    def execute(self, sql, params=None):
        if sql.strip().upper().startswith("SELECT") and params:
            self.select_limits.append(params[0])  # LIMIT bind value

    def fetchall(self):
        return []


def test_backfill_batch_size_clamped_to_minimum(monkeypatch, pg_db):
    """batch_size=0 must never reach the loop as LIMIT 0 (no empty-batch spin).

    Routes through the real handler so the defensive max(1, min(batch_size, 200))
    clamp is exercised end-to-end. A recording cursor captures every SELECT LIMIT
    value; with the clamp in place the loop fetches with LIMIT >= 1 (here 1) and
    terminates on the empty result, never issuing LIMIT 0.
    """
    import psycopg2

    import lore.server as srv

    cur = _LimitRecordingCursor()
    monkeypatch.setattr(psycopg2, "connect", lambda **_k: _BackfillConn(cur))
    # Reach the loop: mining must be enabled and a PG db must be wired in.
    monkeypatch.setattr(srv.telemetry, "mining_enabled", lambda: True)
    monkeypatch.setattr(srv, "db", pg_db)
    # The handler imports encode_text from .embeddings at call time.
    import lore.embeddings as emb

    monkeypatch.setattr(emb, "encode_text", lambda _t: [0.0] * 384, raising=False)

    resp = srv.handle_backfill_query_embeddings({"batch_size": 0, "limit": 1000})

    assert resp["ok"] is True
    # The clamp guarantees LIMIT >= 1 on every SELECT; LIMIT 0 would spin/no-op.
    assert cur.select_limits, "expected at least one SELECT to be issued"
    assert 0 not in cur.select_limits
    assert all(lim >= 1 for lim in cur.select_limits)


# ---------------------------------------------------------------------------
# fetch_reranking_bad_docs: best-effort, returns [] on any error
# ---------------------------------------------------------------------------


def test_fetch_reranking_bad_docs_returns_empty_on_error(monkeypatch, pg_db):
    """A psycopg2 failure must yield [] (re-ranking is best-effort)."""
    import psycopg2

    def _boom(**_k):
        raise psycopg2.OperationalError("connection refused")

    monkeypatch.setattr(psycopg2, "connect", _boom)
    result = telemetry.fetch_reranking_bad_docs([0.0] * 384, pg_db, 0.15)
    assert result == []


# ---------------------------------------------------------------------------
# server._apply_reranking_penalties: bad docs moved to end, not excluded
# ---------------------------------------------------------------------------


def test_apply_reranking_penalties_moves_bad_to_end(monkeypatch):
    import lore.server as srv

    monkeypatch.setattr(srv.telemetry, "fetch_reranking_bad_docs", lambda *_a, **_k: ["doc2"])
    results = [{"kb_id": "doc1"}, {"kb_id": "doc2"}, {"kb_id": "doc3"}]
    out = srv._apply_reranking_penalties(results, [0.0] * 384, db=object())
    assert out == [{"kb_id": "doc1"}, {"kb_id": "doc3"}, {"kb_id": "doc2"}]


def test_apply_reranking_penalties_empty_bad_list(monkeypatch):
    import lore.server as srv

    monkeypatch.setattr(srv.telemetry, "fetch_reranking_bad_docs", lambda *_a, **_k: [])
    results = [{"kb_id": "doc1"}, {"kb_id": "doc2"}, {"kb_id": "doc3"}]
    out = srv._apply_reranking_penalties(results, [0.0] * 384, db=object())
    assert out == results
