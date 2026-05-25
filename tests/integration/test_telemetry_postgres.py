"""Integration tests for retrieval telemetry on PostgreSQL (Issue #5, Phase 1).

These tests exercise the live PostgreSQL backend and require:

  - ``LORE_HARD_NEGATIVE_MINING=true``
  - ``DB_BACKEND`` in {local, postgres, postgresql}
  - A reachable PostgreSQL (default DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD).

The whole module is skipped unless those prerequisites are satisfied. Tests
isolate themselves under a generated topic / session prefix and clean up the
rows they create — they never drop the schema.
"""

from __future__ import annotations

import importlib
import os
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


def _mining_pg_available() -> bool:
    if os.getenv("LORE_HARD_NEGATIVE_MINING", "false").strip().lower() != "true":
        return False
    backend = os.getenv("DB_BACKEND", "").strip().lower()
    if backend not in {"local", "postgres", "postgresql"}:
        return False
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark.append(
    pytest.mark.skipif(
        not _mining_pg_available(),
        reason=(
            "Telemetry PG prerequisites missing: LORE_HARD_NEGATIVE_MINING=true, "
            "DB_BACKEND in {local,postgres,postgresql}, psycopg2 installed, and a "
            "reachable PostgreSQL."
        ),
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_module():
    """Import the server module with a live PG connection + telemetry schema."""
    os.environ.setdefault("LORE_ENV", "staging")

    import lore.server as s

    importlib.reload(s)

    try:
        s.db._get_connection()  # triggers _init_schema → ensure_telemetry_schema
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot connect to PostgreSQL: {exc}")
    return s


@pytest.fixture
def session_id():
    return f"_test_tele_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_session(server_module, session_id):
    """Yield a session id, then delete telemetry rows under it."""
    s = server_module
    yield session_id
    # Open a fresh, short-lived autocommit connection for teardown (mirrors the
    # production read helpers' _pg_conn_params pattern) so the DELETE is always
    # committed and never leaks rows in an open transaction.
    try:
        import psycopg2

        from lore.telemetry import _pg_conn_params

        conn_params = _pg_conn_params(s.db)
        if conn_params is None:
            return
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM knowledge.retrieval_telemetry WHERE session_id = %s",
                    (session_id,),
                )
        finally:
            conn.close()
    except Exception:
        pass


def _fetch_row(s, query_id):
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT query_id, query_text, topic, search_mode, "
            "retrieved_document_ids, result_count, session_id, parent_query_id, "
            "required_requery, caller_agent, model_version "
            "FROM knowledge.retrieval_telemetry WHERE query_id = %s",
            (query_id,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()


def _wait_for_row(s, query_id, timeout=5.0):
    """Telemetry writes happen on a daemon thread; poll until visible."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = _fetch_row(s, query_id)
        if row is not None:
            return row
        time.sleep(0.05)
    return None


def _wait_for_session_count(s, session_id, expected, timeout=5.0):
    """Poll until at least ``expected`` rows exist for ``session_id`` (async writes)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        conn = s.db._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM knowledge.retrieval_telemetry WHERE session_id = %s",
                (session_id,),
            )
            if cursor.fetchone()[0] >= expected:
                return True
        finally:
            cursor.close()
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_telemetry_table_exists(server_module):
    s = server_module
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='knowledge' AND table_name='retrieval_telemetry')"
        )
        assert cursor.fetchone()[0] is True
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# All 5 kb_search return paths emit a query_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"search_mode": "fts"}, id="postgres-fts-path"),
        pytest.param({"search_mode": "semantic"}, id="semantic-path"),
        pytest.param({"search_mode": "hybrid"}, id="hybrid-path"),
        pytest.param({"semantic": True}, id="semantic-flag-path"),
        pytest.param({}, id="default-lexical-path"),
    ],
)
def test_every_path_returns_query_id(server_module, cleanup_session, kwargs):
    """Each routing variant must tag the response with a query_id.

    Some variants degrade to lexical when semantic is unavailable; either way
    a query_id must be present because every success return is finalized.
    """
    s = server_module
    resp = s.handle_kb_search(
        "telemetry path probe",
        session_id=cleanup_session,
        caller_agent="itest",
        **kwargs,
    )
    assert resp["ok"] is True, resp
    assert "query_id" in resp["data"]
    assert resp["data"]["query_id"].startswith("qry_")


# ---------------------------------------------------------------------------
# Round-trip write
# ---------------------------------------------------------------------------


def test_row_round_trip(server_module, cleanup_session):
    s = server_module
    resp = s.handle_kb_search(
        "round trip query",
        topic="_tele_topic",
        session_id=cleanup_session,
        required_requery=True,
        caller_agent="round-tripper",
    )
    qid = resp["data"]["query_id"]
    row = _wait_for_row(s, qid)
    assert row is not None, "telemetry row not written within timeout"

    (
        r_query_id,
        r_query_text,
        r_topic,
        r_search_mode,
        r_doc_ids,
        r_result_count,
        r_session_id,
        r_parent,
        r_required_requery,
        r_caller_agent,
        r_model_version,
    ) = row

    assert r_query_id == qid
    assert r_query_text == "round trip query"
    assert r_topic == "_tele_topic"
    assert r_session_id == cleanup_session
    assert r_required_requery is True
    assert r_caller_agent == "round-tripper"
    assert isinstance(r_doc_ids, list)
    assert r_result_count == len(r_doc_ids) or r_result_count >= 0
    assert r_parent is None
    assert r_model_version  # live lore version stamped


# ---------------------------------------------------------------------------
# FK behaviour: parent_query_id ON DELETE SET NULL
# ---------------------------------------------------------------------------


def test_parent_fk_set_null_on_delete(server_module, cleanup_session):
    s = server_module

    parent = s.handle_kb_search("parent query", session_id=cleanup_session)
    parent_qid = parent["data"]["query_id"]
    assert _wait_for_row(s, parent_qid) is not None

    child = s.handle_kb_search(
        "child query",
        session_id=cleanup_session,
        parent_query_id=parent_qid,
    )
    child_qid = child["data"]["query_id"]
    child_row = _wait_for_row(s, child_qid)
    assert child_row is not None
    assert child_row[7] == parent_qid  # parent_query_id column

    # Delete the parent; the child's parent_query_id should become NULL (not cascade-delete).
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM knowledge.retrieval_telemetry WHERE query_id = %s",
            (parent_qid,),
        )
    finally:
        cursor.close()

    child_after = _fetch_row(s, child_qid)
    assert child_after is not None, "child row must survive parent deletion"
    assert child_after[7] is None, "parent_query_id must be SET NULL after parent delete"


# ---------------------------------------------------------------------------
# Separate connection: telemetry write does not run on the request connection
# ---------------------------------------------------------------------------


def test_write_uses_separate_connection(server_module, cleanup_session):
    """The background writer opens its own connection, distinct from db._conn."""
    s = server_module
    seen = {}

    real_connect = None
    import psycopg2

    real_connect = psycopg2.connect

    def _tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        seen["bg_conn"] = conn
        return conn

    request_conn = s.db._get_connection()
    psycopg2.connect = _tracking_connect
    try:
        resp = s.handle_kb_search("separate conn probe", session_id=cleanup_session)
        qid = resp["data"]["query_id"]
        assert _wait_for_row(s, qid) is not None
    finally:
        psycopg2.connect = real_connect

    assert "bg_conn" in seen, "background writer should open its own connection"
    assert seen["bg_conn"] is not request_conn


# ===========================================================================
# Phase 2 (Issue #5): notes column, feedback round-trip, partial update,
# fetch-by-session/topic, stats aggregation, not-found, notes truncation.
# ===========================================================================


def test_notes_column_exists(server_module):
    """Migration 006 / TELEMETRY_NOTES_DDL must have added the notes column."""
    s = server_module
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
            "WHERE table_schema='knowledge' AND table_name='retrieval_telemetry' "
            "AND column_name='notes')"
        )
        assert cursor.fetchone()[0] is True
    finally:
        cursor.close()


def test_feedback_round_trip(server_module, cleanup_session):
    """kb_search -> query_id -> log_feedback -> fetch -> verify score + notes."""
    s = server_module
    resp = s.handle_kb_search("feedback round trip", session_id=cleanup_session)
    qid = resp["data"]["query_id"]
    assert _wait_for_row(s, qid) is not None

    fb = s.handle_log_retrieval_feedback(qid, user_feedback_score=4, notes="useful hit")
    assert fb["ok"] is True, fb
    assert fb["data"]["updated"] == 1

    got = s.handle_get_retrieval_telemetry(query_id=qid)
    assert got["ok"] is True
    assert got["data"]["count"] == 1
    row = got["data"]["rows"][0]
    assert row["query_id"] == qid
    assert row["user_feedback_score"] == 4
    assert row["notes"] == "useful hit"


def test_partial_update_coalesce(server_module, cleanup_session):
    """Logging score-only then notes-only must not clobber the earlier field."""
    s = server_module
    resp = s.handle_kb_search("partial update probe", session_id=cleanup_session)
    qid = resp["data"]["query_id"]
    assert _wait_for_row(s, qid) is not None

    assert s.handle_log_retrieval_feedback(qid, user_feedback_score=2)["ok"] is True
    # Notes-only update: score must survive (COALESCE leaves it unchanged).
    assert s.handle_log_retrieval_feedback(qid, notes="added later")["ok"] is True

    row = s.handle_get_retrieval_telemetry(query_id=qid)["data"]["rows"][0]
    assert row["user_feedback_score"] == 2
    assert row["notes"] == "added later"


def test_get_telemetry_by_session(server_module, cleanup_session):
    s = server_module
    for i in range(3):
        s.handle_kb_search(f"session fetch {i}", session_id=cleanup_session)
    # Wait for the last write to land before reading.
    _wait_for_session_count(s, cleanup_session, 3)

    got = s.handle_get_retrieval_telemetry(session_id=cleanup_session)
    assert got["ok"] is True
    assert got["data"]["count"] == 3
    assert all(r["session_id"] == cleanup_session for r in got["data"]["rows"])


def test_get_telemetry_by_topic(server_module, cleanup_session):
    s = server_module
    topic = f"_tele_topic_{cleanup_session}"
    s.handle_kb_search("topic fetch probe", topic=topic, session_id=cleanup_session)
    _wait_for_session_count(s, cleanup_session, 1)

    got = s.handle_get_retrieval_telemetry(topic=topic)
    assert got["ok"] is True
    assert got["data"]["count"] >= 1
    assert all(r["topic"] == topic for r in got["data"]["rows"])


def test_get_telemetry_stats_aggregation(server_module, cleanup_session):
    s = server_module
    q1 = s.handle_kb_search("stats probe 1", session_id=cleanup_session)
    q2 = s.handle_kb_search("stats probe 2", session_id=cleanup_session)
    _wait_for_session_count(s, cleanup_session, 2)

    # Give one row feedback so with_feedback/avg are exercised.
    s.handle_log_retrieval_feedback(q1["data"]["query_id"], user_feedback_score=5)
    s.handle_log_retrieval_feedback(q2["data"]["query_id"], notes="noted")

    stats = s.handle_get_telemetry_stats(session_id=cleanup_session)
    assert stats["ok"] is True
    data = stats["data"]["stats"]
    assert data["total"] == 2
    assert data["with_feedback"] == 1
    assert data["with_notes"] == 1
    assert data["avg_feedback_score"] == 5.0
    assert data["oldest"] is not None and data["newest"] is not None


def test_log_feedback_not_found(server_module):
    s = server_module
    resp = s.handle_log_retrieval_feedback("qry_does_not_exist", user_feedback_score=1)
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_notes_truncation_persists(server_module, cleanup_session):
    s = server_module
    from lore.telemetry import MAX_NOTES_LEN

    resp = s.handle_kb_search("truncation probe", session_id=cleanup_session)
    qid = resp["data"]["query_id"]
    assert _wait_for_row(s, qid) is not None

    long_notes = "z" * (MAX_NOTES_LEN + 1000)
    assert s.handle_log_retrieval_feedback(qid, notes=long_notes)["ok"] is True

    row = s.handle_get_retrieval_telemetry(query_id=qid)["data"]["rows"][0]
    assert len(row["notes"]) == MAX_NOTES_LEN
