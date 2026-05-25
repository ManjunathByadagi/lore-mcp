"""Retrieval telemetry foundation for hard negative mining (Issue #5, Phase 1).

This module records each ``kb_search`` invocation into a PostgreSQL table so a
later phase can mine hard negatives (documents that were retrieved but turned
out to be unhelpful) for re-ranking / fine-tuning. Phase 1 only *captures*
telemetry — no mining, scoring, or re-query inference happens here.

Design constraints (deliberately minimal for Phase 1):

  - PostgreSQL only. SQLite/Supabase backends are a hard no-op. Mining is
    gated behind ``LORE_HARD_NEGATIVE_MINING=true`` AND a PostgreSQL backend.
  - Writes happen on a daemon thread using a *separate* connection so a slow
    or failing telemetry write never blocks or corrupts the request path.
  - Every failure is swallowed (logged at WARNING) — telemetry is best-effort
    and must never surface an error to the caller.
  - No connection pooling, no queue-based writer (out of scope for Phase 1).
"""

from __future__ import annotations

import logging
import os
import secrets
import threading
from typing import Any

logger = logging.getLogger(__name__)


def mining_enabled() -> bool:
    """Whether retrieval telemetry capture is active.

    Requires BOTH:
      - ``LORE_HARD_NEGATIVE_MINING`` set to exactly ``"true"`` (case-insensitive), AND
      - ``DB_BACKEND`` naming a PostgreSQL backend (``local`` / ``postgres`` /
        ``postgresql``) — the same spellings ``get_db_client`` accepts for PG.

    Backend detection mirrors ``server._backend_kind`` but is re-derived here
    to avoid importing from ``server`` (circular import).
    """
    flag = os.getenv("LORE_HARD_NEGATIVE_MINING", "false").strip().lower()
    if flag != "true":
        return False
    backend = os.getenv("DB_BACKEND", "").strip().lower()
    return backend in {"local", "postgres", "postgresql"}


def generate_query_id() -> str:
    """Return a fresh, unguessable query id of the form ``qry_<12 hex>``."""
    return "qry_" + secrets.token_hex(6)


# DDL for the telemetry table. Kept byte-for-byte identical to
# migrations/005_retrieval_telemetry.sql (verified by a unit test).
TELEMETRY_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge.retrieval_telemetry (
    query_id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    topic TEXT,
    search_mode TEXT,
    retrieved_document_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    result_count INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    parent_query_id TEXT REFERENCES knowledge.retrieval_telemetry(query_id) ON DELETE SET NULL,
    required_requery BOOLEAN NOT NULL DEFAULT FALSE,
    caller_agent TEXT,
    user_feedback_score INTEGER,
    model_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_session ON knowledge.retrieval_telemetry (session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_created ON knowledge.retrieval_telemetry (created_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_telemetry_parent ON knowledge.retrieval_telemetry (parent_query_id);
"""


# Phase 2 (Issue #5): the ``notes`` column is added by a *separate* idempotent
# DDL statement rather than by editing TELEMETRY_PG_SCHEMA (which stays frozen
# and byte-for-byte identical to migration 005). This single statement mirrors
# migrations/006_telemetry_notes.sql (a unit test enforces parity).
TELEMETRY_NOTES_DDL = (
    "ALTER TABLE knowledge.retrieval_telemetry ADD COLUMN IF NOT EXISTS notes TEXT"
)

# Phase 2 bounds (Issue #5):
#   - MAX_NOTES_LEN     caps stored feedback notes (defensive truncation).
#   - MAX_READ_LIMIT    hard ceiling on rows a read tool may return.
#   - DEFAULT_READ_LIMIT default when the caller supplies no/invalid limit.
MAX_NOTES_LEN = 4000
MAX_READ_LIMIT = 500
DEFAULT_READ_LIMIT = 50


def ensure_telemetry_schema(conn) -> None:
    """Create the retrieval_telemetry table + indexes (+ notes column) if absent.

    Idempotent (CREATE ... IF NOT EXISTS / ADD COLUMN IF NOT EXISTS). Called
    from ``LocalPostgresClient._init_schema`` inside an isolated try/except so a
    failure here can never disturb the rest of schema initialisation.
    Uses the *existing* request connection (autocommit) for the one-time DDL.
    """
    cursor = conn.cursor()
    try:
        # psycopg2's cursor.execute() runs only the first statement in a
        # multi-statement string, so split on ';' and execute each non-empty
        # statement individually (otherwise the 3 CREATE INDEX statements are
        # silently dropped on a fresh database). The Phase 2 notes column
        # (migration 006) is applied as one more statement after the base schema.
        statements = [s.strip() for s in TELEMETRY_PG_SCHEMA.split(";") if s.strip()]
        statements.append(TELEMETRY_NOTES_DDL)
        for stmt in statements:
            cursor.execute(stmt)
    finally:
        cursor.close()


def _pg_conn_params(db: Any) -> dict | None:
    """Extract psycopg2 connection params from a ``LocalPostgresClient``.

    Returns ``None`` for any non-PostgreSQL ``db`` (SQLite/Supabase/test fakes),
    mirroring the backend guard in ``write_retrieval_telemetry_async``. Read
    functions use this to open a *fresh* connection per call rather than sharing
    ``db._conn`` (which is not safe for concurrent use across request threads).
    """
    from lore.db_client import LocalPostgresClient

    if not isinstance(db, LocalPostgresClient):
        return None
    return {
        "host": db.host,
        "port": db.port,
        "dbname": db.database,
        "user": db.user,
        "password": db.password,
    }


def write_retrieval_telemetry_async(
    *,
    query_id: str,
    query_text: str,
    topic: str | None,
    search_mode: str | None,
    retrieved_document_ids: list[str],
    result_count: int,
    session_id: str | None,
    parent_query_id: str | None,
    required_requery: bool,
    caller_agent: str | None,
    model_version: str | None,
    db: Any,
) -> threading.Thread | None:
    """Spawn a daemon thread that writes one telemetry row, then return.

    Returns the started ``Thread`` (so callers/tests can join it) or ``None``
    when the write is skipped. The write is skipped — without raising — when
    ``db`` is not a PostgreSQL client; this guards against ``AttributeError``
    on SQLite/Supabase backends and in tests, and should never be hit when
    ``mining_enabled()`` is honoured by the caller.
    """
    # Local import avoids a module-level cycle (db_client imports nothing from us,
    # but keeping it local matches the isolation pattern used elsewhere).
    from lore.db_client import LocalPostgresClient

    if not isinstance(db, LocalPostgresClient):
        return None

    conn_params = {
        "host": db.host,
        "port": db.port,
        "dbname": db.database,
        "user": db.user,
        "password": db.password,
    }

    thread = threading.Thread(
        target=_write_row,
        kwargs=dict(
            conn_params=conn_params,
            query_id=query_id,
            query_text=query_text,
            topic=topic,
            search_mode=search_mode,
            retrieved_document_ids=retrieved_document_ids,
            result_count=result_count,
            session_id=session_id,
            parent_query_id=parent_query_id,
            required_requery=required_requery,
            caller_agent=caller_agent,
            model_version=model_version,
        ),
        daemon=True,
    )
    thread.start()
    return thread


def _write_row(
    *,
    conn_params: dict,
    query_id: str,
    query_text: str,
    topic: str | None,
    search_mode: str | None,
    retrieved_document_ids: list[str],
    result_count: int,
    session_id: str | None,
    parent_query_id: str | None,
    required_requery: bool,
    caller_agent: str | None,
    model_version: str | None,
) -> None:
    """Insert a single telemetry row on a fresh, dedicated connection.

    Runs on a background daemon thread. Opens its own short-lived connection
    (never the request connection) so it cannot interfere with request-path
    transactions. All exceptions are swallowed and logged at WARNING —
    telemetry must never break the caller.
    """
    import json

    conn = None
    try:
        import psycopg2

        conn = psycopg2.connect(**conn_params)
        conn.set_client_encoding("UTF8")
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO knowledge.retrieval_telemetry (
                    query_id, query_text, topic, search_mode,
                    retrieved_document_ids, result_count, session_id,
                    parent_query_id, required_requery, caller_agent, model_version
                ) VALUES (
                    %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (query_id) DO NOTHING
                """,
                (
                    query_id,
                    query_text,
                    topic,
                    search_mode,
                    json.dumps(list(retrieved_document_ids)),
                    result_count,
                    session_id,
                    parent_query_id,
                    required_requery,
                    caller_agent,
                    model_version,
                ),
            )
        finally:
            cursor.close()
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.warning("Failed to write retrieval telemetry row (non-fatal): %s", exc)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


# ===========================================================================
# Phase 2 (Issue #5): synchronous read / update helpers for the analysis tools.
#
# Unlike the write path these run on the request thread — the caller awaits the
# result — so they must NOT share ``db._conn``. Each opens a *fresh* connection
# (via _pg_conn_params) and closes it within the call. All three return ``None``
# for a non-PostgreSQL backend so handlers can map that to a clean no-op.
# ===========================================================================

# Columns returned by fetch_retrieval_telemetry, in a stable order. ``created_at``
# (TIMESTAMPTZ) is serialised by server.json_serializer — no extra handling here.
_TELEMETRY_READ_COLUMNS = (
    "query_id, query_text, topic, search_mode, retrieved_document_ids, "
    "result_count, session_id, parent_query_id, required_requery, caller_agent, "
    "user_feedback_score, notes, model_version, created_at"
)


def clamp_read_limit(limit: Any) -> int:
    """Coerce an arbitrary ``limit`` into ``1..MAX_READ_LIMIT``.

    Non-integer / missing values fall back to ``DEFAULT_READ_LIMIT``; values are
    then clamped to ``[1, MAX_READ_LIMIT]`` so a hostile or sloppy caller can
    neither exhaust the table nor request a non-positive page.
    """
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = DEFAULT_READ_LIMIT
    return max(1, min(MAX_READ_LIMIT, value))


def update_retrieval_feedback(
    *,
    query_id: str,
    user_feedback_score: int | None,
    notes: str | None,
    db: Any,
) -> int | None:
    """Apply caller feedback to one telemetry row; return rows affected.

    Partial update via COALESCE: a ``None`` argument leaves that column
    unchanged (so the caller may set the score, the notes, or both). A
    consequence — documented as a limitation — is that neither field can be
    *reset* to NULL through this path.

    ``notes`` is defensively truncated to ``MAX_NOTES_LEN`` before storage.
    Returns ``cursor.rowcount`` (0 when ``query_id`` is unknown), or ``None``
    when ``db`` is not a PostgreSQL client.
    """
    import psycopg2

    conn_params = _pg_conn_params(db)
    if conn_params is None:
        return None

    if notes is not None and len(notes) > MAX_NOTES_LEN:
        notes = notes[:MAX_NOTES_LEN]

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE knowledge.retrieval_telemetry
                   SET user_feedback_score = COALESCE(%s, user_feedback_score),
                       notes               = COALESCE(%s, notes)
                 WHERE query_id = %s
                """,
                (user_feedback_score, notes, query_id),
            )
            return cur.rowcount
    finally:
        conn.close()


def fetch_retrieval_telemetry(
    *,
    query_id: str | None,
    session_id: str | None,
    topic: str | None,
    limit: Any,
    db: Any,
) -> list[dict] | None:
    """Read telemetry rows by the first selector provided.

    Selector precedence: ``query_id`` > ``session_id`` > ``topic`` > recent.
    ``query_id`` returns the single matching row (still as a list); the other
    selectors return up to ``limit`` rows ordered newest-first. Returns ``None``
    when ``db`` is not a PostgreSQL client.
    """
    import psycopg2
    import psycopg2.extras

    conn_params = _pg_conn_params(db)
    if conn_params is None:
        return None

    bounded_limit = clamp_read_limit(limit)

    base = f"SELECT {_TELEMETRY_READ_COLUMNS} FROM knowledge.retrieval_telemetry"
    if query_id is not None:
        sql = f"{base} WHERE query_id = %s"
        params: tuple = (query_id,)
    elif session_id is not None:
        sql = f"{base} WHERE session_id = %s ORDER BY created_at DESC LIMIT %s"
        params = (session_id, bounded_limit)
    elif topic is not None:
        sql = f"{base} WHERE topic = %s ORDER BY created_at DESC LIMIT %s"
        params = (topic, bounded_limit)
    else:
        sql = f"{base} ORDER BY created_at DESC LIMIT %s"
        params = (bounded_limit,)

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_telemetry_stats(
    *,
    session_id: str | None,
    topic: str | None,
    db: Any,
) -> dict | None:
    """Aggregate telemetry counts/averages, optionally scoped by session/topic.

    Both filters are optional and AND-combined; a ``None`` filter matches every
    row (``%(x)s IS NULL OR col = %(x)s``). Returns a single stats dict (the
    ``oldest``/``newest`` timestamps serialise via server.json_serializer), or
    ``None`` when ``db`` is not a PostgreSQL client.
    """
    import psycopg2
    import psycopg2.extras

    conn_params = _pg_conn_params(db)
    if conn_params is None:
        return None

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COUNT(*)                                 AS total,
                       COUNT(user_feedback_score)               AS with_feedback,
                       COUNT(*) FILTER (WHERE required_requery)  AS requeries,
                       COUNT(notes)                             AS with_notes,
                       AVG(user_feedback_score)::float           AS avg_feedback_score,
                       MIN(created_at)                          AS oldest,
                       MAX(created_at)                          AS newest
                  FROM knowledge.retrieval_telemetry
                 WHERE (%(session_id)s IS NULL OR session_id = %(session_id)s)
                   AND (%(topic)s      IS NULL OR topic      = %(topic)s)
                """,
                {"session_id": session_id, "topic": topic},
            )
            return dict(cur.fetchone())
    finally:
        conn.close()
