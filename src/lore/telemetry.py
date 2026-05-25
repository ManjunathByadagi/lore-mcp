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


def ensure_telemetry_schema(conn) -> None:
    """Create the retrieval_telemetry table + indexes if absent.

    Idempotent (CREATE ... IF NOT EXISTS). Called from
    ``LocalPostgresClient._init_schema`` inside an isolated try/except so a
    failure here can never disturb the rest of schema initialisation.
    Uses the *existing* request connection (autocommit) for the one-time DDL.
    """
    cursor = conn.cursor()
    try:
        # psycopg2's cursor.execute() runs only the first statement in a
        # multi-statement string, so split on ';' and execute each non-empty
        # statement individually (otherwise the 3 CREATE INDEX statements are
        # silently dropped on a fresh database).
        for stmt in TELEMETRY_PG_SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                cursor.execute(stmt)
    finally:
        cursor.close()


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
