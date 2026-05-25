"""Unit tests for retrieval telemetry (Issue #5, Phase 1).

Pure unit tests — no live database. The PostgreSQL round-trip / FK / 5-path
coverage lives in tests/integration/test_telemetry_postgres.py.
"""

from __future__ import annotations

import re
from pathlib import Path

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
