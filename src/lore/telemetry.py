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

import hashlib
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
    "ALTER TABLE knowledge.retrieval_telemetry ADD COLUMN IF NOT EXISTS notes TEXT;"
)

# Phase 4a (Issue #5): the ``query_embedding`` column stores each query's
# embedding alongside its telemetry row so Phase 4b re-ranking can find
# historically-poor docs for queries similar to the current one (ANN cosine).
# Single idempotent statement; mirrors migrations/008_query_embedding.sql (the
# HNSW index is deliberately NOT auto-applied — it is created on demand by the
# backfill_query_embeddings tool to avoid blocking startup on large tables).
TELEMETRY_QUERY_EMBEDDING_DDL = """\
ALTER TABLE knowledge.retrieval_telemetry
    ADD COLUMN IF NOT EXISTS query_embedding halfvec(384);\
"""

# Phase 2 bounds (Issue #5):
#   - MAX_NOTES_LEN     caps stored feedback notes (defensive truncation).
#   - MAX_READ_LIMIT    hard ceiling on rows a read tool may return.
#   - DEFAULT_READ_LIMIT default when the caller supplies no/invalid limit.
MAX_NOTES_LEN = 4000
MAX_READ_LIMIT = 500
DEFAULT_READ_LIMIT = 50


# Phase 3 (Issue #5): DDL for the hard_negative_pairs table + indexes. Kept
# byte-for-byte identical to the SQL body of migrations/007_hard_negative_pairs.sql
# (verified by a unit test). ON DELETE RESTRICT — pairs are training signal, not
# disposable cache; a KB entry deletion is blocked until its pairs are cleared.
HARD_NEGATIVE_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge.hard_negative_pairs (
    pair_id          TEXT PRIMARY KEY,
    query_text       TEXT NOT NULL,
    doc_id           TEXT NOT NULL REFERENCES knowledge.kb_entries(kb_id) ON DELETE RESTRICT,
    signal_type      TEXT NOT NULL CHECK (signal_type IN ('explicit', 'behavioral')),
    source_query_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_hn_pairs_query_doc
    ON knowledge.hard_negative_pairs (query_text, doc_id);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_doc_id
    ON knowledge.hard_negative_pairs (doc_id);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_signal
    ON knowledge.hard_negative_pairs (signal_type);
CREATE INDEX IF NOT EXISTS idx_hn_pairs_last_seen
    ON knowledge.hard_negative_pairs (last_seen_at);
"""

# Phase 3 read bounds: hard ceiling + default for get_hard_negatives.
MAX_HN_LIMIT = 1000
DEFAULT_HN_LIMIT = 100


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
        statements.append(TELEMETRY_NOTES_DDL.rstrip(";").strip())
        # Phase 4a (Issue #5): the query_embedding column (migration 008) is
        # applied as one more statement after the notes column, following the
        # same split-and-loop pattern so a fresh database picks it up at startup.
        for stmt in TELEMETRY_QUERY_EMBEDDING_DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                statements.append(stmt)
        for stmt in statements:
            cursor.execute(stmt)
    finally:
        cursor.close()


def ensure_hard_negative_schema(conn) -> None:
    """Create the hard_negative_pairs table + indexes if absent (Phase 3).

    Mirrors ``ensure_telemetry_schema``: psycopg2 runs only the first statement
    of a multi-statement string, so split on ';' and execute each non-empty
    statement individually (otherwise the 4 index statements are silently
    dropped on a fresh database). Idempotent (CREATE ... IF NOT EXISTS). Wrapped
    in try/except by the caller (db_client) so a failure here can never disturb
    schema initialisation; we additionally swallow + warn here defensively.
    """
    cursor = conn.cursor()
    try:
        statements = [s.strip() for s in HARD_NEGATIVE_PG_SCHEMA.split(";") if s.strip()]
        for stmt in statements:
            cursor.execute(stmt)
    except Exception as exc:  # noqa: BLE001 — schema init is best-effort
        logger.warning("ensure_hard_negative_schema failed: %s", exc)
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
    query_embedding: list | None = None,
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
            query_embedding=query_embedding,
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
    query_embedding: list | None = None,
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
                    parent_query_id, required_requery, caller_agent, model_version,
                    query_embedding
                ) VALUES (
                    %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s::halfvec
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
                    # Phase 4a (Issue #5): query_embedding stored as halfvec(384).
                    # psycopg2 adapts a Python list[float] to a pgvector literal
                    # under the explicit ::halfvec cast; None becomes NULL.
                    query_embedding,
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
    except (TypeError, ValueError, OverflowError):
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

    conn = None
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
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
        if conn is not None:
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
            row = cur.fetchone()
            return dict(row) if row is not None else {
                "total": 0, "with_feedback": 0, "requeries": 0,
                "with_notes": 0, "avg_feedback_score": None,
                "oldest": None, "newest": None
            }
    finally:
        conn.close()


# ===========================================================================
# Phase 3 (Issue #5): hard negative mining — derive (query, doc) pairs from
# telemetry and upsert them into knowledge.hard_negative_pairs, plus a read
# helper. Both run on the request thread and open a *fresh* connection (via
# _pg_conn_params); both return ``None`` for a non-PostgreSQL backend.
# ===========================================================================

# Two source queries over retrieval_telemetry. Each unrolls the JSONB
# retrieved_document_ids array into one row per doc_id and tags it with a
# signal type. The optional ``{since}`` clause is interpolated (not a bind
# param) only when ``since`` is non-None; the value itself is always passed
# as a %(since)s bind parameter to keep it parameterized.
_HN_EXPLICIT_SQL = """
SELECT query_id, query_text,
       jsonb_array_elements_text(retrieved_document_ids) AS doc_id
  FROM knowledge.retrieval_telemetry
 WHERE user_feedback_score <= 2
   AND result_count > 0
   {since}
"""

_HN_BEHAVIORAL_SQL = """
SELECT query_id, query_text,
       jsonb_array_elements_text(retrieved_document_ids) AS doc_id
  FROM knowledge.retrieval_telemetry
 WHERE required_requery = TRUE
   AND result_count > 0
   AND user_feedback_score IS NULL
   {since}
"""

_HN_UPSERT_SQL = """
INSERT INTO knowledge.hard_negative_pairs
    (pair_id, query_text, doc_id, signal_type, source_query_ids,
     occurrence_count, first_seen_at, last_seen_at)
VALUES (%s, %s, %s, %s, %s::jsonb, 1, NOW(), NOW())
ON CONFLICT (pair_id) DO UPDATE SET
    occurrence_count = knowledge.hard_negative_pairs.occurrence_count + 1,
    last_seen_at     = NOW(),
    source_query_ids = (
        SELECT jsonb_agg(DISTINCT v)
        FROM jsonb_array_elements_text(
            knowledge.hard_negative_pairs.source_query_ids || EXCLUDED.source_query_ids
        ) t(v)
    )
"""


def hard_negative_pair_id(query_text: str, doc_id: str) -> str:
    """Deterministic 32-hex pair id derived from (query_text, doc_id).

    Using SHA-256 (not a random token) makes the primary key stable across
    refresh runs, so ``ON CONFLICT (pair_id) DO UPDATE`` fires on every re-run
    instead of inserting duplicates.
    """
    return hashlib.sha256(f"{query_text}:{doc_id}".encode()).hexdigest()[:32]


def refresh_hard_negative_pairs(
    *,
    since: str | None,
    dry_run: bool,
    db: Any,
) -> dict | None:
    """Scan telemetry, derive hard negative pairs, and upsert them.

    ``since`` is an ISO timestamp string (or ``None`` for a full refresh); when
    set, only telemetry rows with ``created_at > since::timestamptz`` are mined.
    Each ``(query_text, doc_id)`` becomes a deterministic ``pair_id`` and is
    batch-upserted (occurrence_count++ and source_query_ids deduped on conflict).

    ``dry_run`` runs the full upsert inside a transaction and ROLLBACKs at the
    end — the returned counts reflect what *would* change, but nothing persists.

    Returns a summary dict, or ``None`` when ``db`` is not a PostgreSQL client.
    """
    import json

    import psycopg2

    conn_params = _pg_conn_params(db)
    if conn_params is None:
        return None

    explicit_sql = _HN_EXPLICIT_SQL.format(
        since="AND created_at > %(since)s::timestamptz" if since else ""
    )
    behavioral_sql = _HN_BEHAVIORAL_SQL.format(
        since="AND created_at > %(since)s::timestamptz" if since else ""
    )

    conn = psycopg2.connect(**conn_params)
    conn.set_client_encoding("UTF8")
    # Transactional (not autocommit) so dry_run can ROLLBACK the whole batch.
    conn.autocommit = False
    try:
        # Count existing pairs first so we can derive inserted vs updated even
        # under a dry-run rollback (occurrence_count deltas aren't surfaced).
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge.hard_negative_pairs")
            total_before = cur.fetchone()[0]

        # One upsert param tuple per (query_text, doc_id) telemetry row. The
        # pair_id is the deterministic SHA-256 of (query_text, doc_id) so that
        # ON CONFLICT fires across refreshes; source_query_ids is seeded with the
        # originating telemetry query_id and deduped on conflict.
        upsert_params: list[tuple] = []
        with conn.cursor() as cur:
            cur.execute(explicit_sql, {"since": since})
            explicit_rows = cur.fetchall()
            cur.execute(behavioral_sql, {"since": since})
            behavioral_rows = cur.fetchall()

        for query_id, query_text, doc_id in explicit_rows:
            upsert_params.append((
                hard_negative_pair_id(query_text, doc_id),
                query_text, doc_id, "explicit", json.dumps([query_id]),
            ))
        for query_id, query_text, doc_id in behavioral_rows:
            upsert_params.append((
                hard_negative_pair_id(query_text, doc_id),
                query_text, doc_id, "behavioral", json.dumps([query_id]),
            ))

        processed = len(explicit_rows) + len(behavioral_rows)

        if upsert_params:
            with conn.cursor() as cur:
                cur.executemany(_HN_UPSERT_SQL, upsert_params)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge.hard_negative_pairs")
            total_after = cur.fetchone()[0]

        inserted = total_after - total_before
        updated = len(upsert_params) - inserted

        if dry_run:
            conn.rollback()
        else:
            conn.commit()

        return {
            "inserted": inserted,
            "updated": max(updated, 0),
            "total_pairs": total_after,
            "processed_telemetry_rows": processed,
            "since": since or "all",
            "dry_run": dry_run,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns returned by fetch_hard_negatives, in a stable order. The TIMESTAMPTZ
# columns serialise via server.json_serializer — no extra handling here.
_HN_READ_COLUMNS = (
    "pair_id, query_text, doc_id, signal_type, source_query_ids, "
    "occurrence_count, first_seen_at, last_seen_at"
)


def fetch_hard_negatives(
    *,
    signal_type: str | None,
    limit: int,
    doc_id: str | None,
    query_text_like: str | None,
    db: Any,
) -> list[dict] | None:
    """Read hard negative pairs with optional filters, busiest pairs first.

    Filters (all optional, AND-combined): ``signal_type`` (exact),
    ``doc_id`` (exact), ``query_text_like`` (case-insensitive substring via
    parameterized ILIKE). Ordered by ``occurrence_count`` then ``last_seen_at``
    descending, capped at ``limit``. Returns ``None`` for a non-PG backend.
    """
    import psycopg2
    import psycopg2.extras

    conn_params = _pg_conn_params(db)
    if conn_params is None:
        return None

    clauses: list[str] = []
    params: list[Any] = []
    if signal_type is not None:
        clauses.append("signal_type = %s")
        params.append(signal_type)
    if doc_id is not None:
        clauses.append("doc_id = %s")
        params.append(doc_id)
    if query_text_like is not None:
        # Escape ILIKE special characters so user input is treated as literals
        escaped = (
            query_text_like
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        like_param = f"%{escaped}%"
        clauses.append("query_text ILIKE %s ESCAPE '\\'")
        params.append(like_param)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = (
        f"SELECT {_HN_READ_COLUMNS} FROM knowledge.hard_negative_pairs"
        f"{where} ORDER BY occurrence_count DESC, last_seen_at DESC LIMIT %s"
    )
    params.append(limit)

    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            try:
                cur.execute(sql, tuple(params))
                return [dict(row) for row in cur.fetchall()]
            except psycopg2.errors.UndefinedTable:
                logger.debug(
                    "hard_negative_pairs table does not exist yet; returning empty list"
                )
                return []
    finally:
        conn.close()


# ===========================================================================
# Phase 4 (Issue #5): query-embedding storage + re-ranking.
#
# Phase 4b re-ranking is INDEPENDENT of mining: it reads from
# hard_negative_pairs (a pre-aggregated quality signal that may have been
# populated by another process or by hand) joined with the Phase 4a
# query_embedding column for ANN similarity. The gate is its own env flag
# (LORE_RERANKING_ENABLED) so it can be toggled without enabling mining.
# Both helpers below open a fresh connection and are strictly best-effort:
# re-ranking must NEVER break the search request path.
# ===========================================================================


def reranking_enabled() -> bool:
    """Returns True if re-ranking is enabled.

    Re-ranking is independent of telemetry mining; it reads from
    hard_negative_pairs which may have been populated by another process or
    manually.

    Requires: LORE_RERANKING_ENABLED=true (or 1 or yes)
    Also requires: PostgreSQL backend (re-ranking is Postgres-only).
    """
    return os.environ.get("LORE_RERANKING_ENABLED", "").lower() in ("1", "true", "yes")


def fetch_reranking_bad_docs(query_embedding: list, db, cosine_threshold: float = 0.15) -> list[str]:
    """Returns doc_ids that were historically poor matches for queries similar to the current one.

    Uses hard_negative_pairs as the source (pre-aggregated quality signal)
    joined with retrieval_telemetry.query_embedding for ANN similarity.

    Returns [] on any error (re-ranking is best-effort; must not break search).
    """
    try:
        import psycopg2

        conn_params = _pg_conn_params(db)
        if conn_params is None:
            return []
        conn = psycopg2.connect(**conn_params)
        try:
            cursor = conn.cursor()
            # Find query_ids from telemetry where the stored query embedding is
            # within cosine_threshold of the current query embedding.
            # Then find doc_ids from hard_negative_pairs that appeared in those query_ids.
            cursor.execute(
                """
                SELECT DISTINCT hnp.doc_id
                FROM knowledge.hard_negative_pairs hnp
                JOIN knowledge.retrieval_telemetry rt
                    ON rt.query_id = ANY(
                        SELECT jsonb_array_elements_text(hnp.source_query_ids)
                    )
                WHERE rt.query_embedding IS NOT NULL
                  AND (rt.query_embedding <=> %s::halfvec) < %s
                LIMIT 500
                """,
                (query_embedding, cosine_threshold),
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()
    except Exception:
        return []


def backfill_query_embeddings(
    db,
    encode_fn,
    batch_size: int = 32,
    limit: int = 1000,
    dry_run: bool = False,
    build_index: bool = False,
) -> dict:
    """Backfill query_embedding for rows in retrieval_telemetry where it is NULL.

    Args:
        db: DB client (for connection params)
        encode_fn: callable(text) -> list[float] (pass encode_text)
        batch_size: rows per batch (default 32)
        limit: max rows to process in this call (default 1000)
        dry_run: if True, compute embeddings but do not write
        build_index: if True, run CREATE INDEX CONCURRENTLY after backfill

    Returns dict with: processed, updated, skipped, index_built
    """
    import psycopg2

    conn_params = _pg_conn_params(db)
    conn = psycopg2.connect(**conn_params)
    processed = 0
    updated = 0
    skipped = 0
    try:
        cursor = conn.cursor()
        # Fetch in batches using LIMIT/OFFSET
        offset = 0
        remaining = limit
        while remaining > 0:
            batch = min(batch_size, remaining)
            cursor.execute(
                """
                SELECT query_id, query_text
                FROM knowledge.retrieval_telemetry
                WHERE query_embedding IS NULL
                ORDER BY created_at ASC
                LIMIT %s OFFSET %s
                """,
                (batch, offset),
            )
            rows = cursor.fetchall()
            if not rows:
                break
            for query_id, query_text in rows:
                processed += 1
                try:
                    embedding = encode_fn(query_text)
                    if not dry_run:
                        cursor.execute(
                            "UPDATE knowledge.retrieval_telemetry "
                            "SET query_embedding = %s::halfvec WHERE query_id = %s",
                            (embedding, query_id),
                        )
                        updated += 1
                except Exception:
                    skipped += 1
            if not dry_run:
                conn.commit()
            offset += len(rows)
            remaining -= len(rows)
    finally:
        conn.close()

    index_built = False
    if build_index and not dry_run and updated > 0:
        # Run CONCURRENTLY — must be outside a transaction block
        idx_conn = psycopg2.connect(**conn_params)
        idx_conn.autocommit = True
        try:
            idx_cursor = idx_conn.cursor()
            idx_cursor.execute(
                """
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_retrieval_telemetry_qemb_hnsw
                ON knowledge.retrieval_telemetry
                USING hnsw (query_embedding halfvec_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            index_built = True
        except Exception:
            pass
        finally:
            idx_conn.close()

    return {
        "processed": processed,
        "updated": updated,
        "skipped": skipped,
        "dry_run": dry_run,
        "index_built": index_built,
    }
