"""Hybrid lexical + semantic search orchestration for lore-mcp.

Supports both SQLite (FTS5 + sqlite-vec) and PostgreSQL (websearch_to_tsquery
+ pgvector HNSW) backends. When ``LORE_SEMANTIC_SEARCH`` is false or the
embedding model / vector extension is unavailable, the search module falls
back to the legacy lexical search via the Supabase-compatible TableQuery
interface — preserving today's behavior.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def semantic_enabled() -> bool:
    """True iff ``LORE_SEMANTIC_SEARCH`` is set to ``true``."""
    return os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower() == "true"


def default_search_mode() -> str:
    """Default ``search_mode`` when caller does not specify one."""
    raw = os.getenv("LORE_SEARCH_MODE_DEFAULT", "hybrid").strip().lower()
    if raw not in {"fts", "semantic", "hybrid"}:
        return "hybrid"
    return raw


def rrf_k() -> int:
    """Reciprocal Rank Fusion smoothing constant. Default 10."""
    raw = os.getenv("LORE_RRF_K", "10").strip() or "10"
    try:
        k = int(raw)
    except ValueError:
        logger.warning("Invalid LORE_RRF_K=%r; using 10", raw)
        return 10
    return max(1, k)


def debug_search() -> bool:
    """Enable verbose per-result debug logging."""
    return os.getenv("LORE_DEBUG_SEARCH", "false").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Public ranking / fusion API
# ---------------------------------------------------------------------------


def candidate_pool_size(top_k: int, corpus_size: int) -> int:
    """Adaptive candidate pool size per Issue #6.

    ``min(corpus_size, 200, max(top_k*5, 50))`` — wide enough to give RRF
    headroom on small KBs without scanning huge corpora on big ones.
    """
    if corpus_size <= 0:
        return max(top_k, 50)
    return max(1, min(corpus_size, 200, max(top_k * 5, 50)))


def reciprocal_rank_fusion(
    rankings: Iterable[list[str]],
    k: int | None = None,
) -> list[tuple[str, float]]:
    """Standard Reciprocal Rank Fusion (Cormack et al. 2009).

    ``rankings`` is an iterable of per-list ranked identifier lists, ordered
    best-first. Returns ``(id, score)`` pairs sorted by descending score.
    Identifiers not present in any list are not returned.

    Empty input yields an empty list. Duplicate identifiers within a single
    ranking only contribute their first (best) rank — guards against
    upstream bugs.
    """
    fusion_k = rrf_k() if k is None else max(1, int(k))
    scores: dict[str, float] = {}

    for ranking in rankings:
        seen_in_list: set[str] = set()
        for rank, ident in enumerate(ranking, start=1):
            if ident in seen_in_list:
                continue
            seen_in_list.add(ident)
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (fusion_k + rank)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Backend-specific search primitives (SQLite)
# ---------------------------------------------------------------------------


def fts5_search_sqlite(
    db_client: Any,
    query: str,
    topic: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Run an FTS5 MATCH against ``knowledge_kb_entries_fts``.

    Returns rows ordered by FTS5 ``bm25()`` score (lower = better match).
    Each row is a dict with ``kb_id``, ``title``, ``content``, ``topic``,
    ``tags``, ``author``, ``source_type``, ``verified``, ``trust_score``, and
    ``score``.
    """
    if not getattr(db_client, "fts5_available", False):
        return []
    conn = db_client._get_connection()

    # FTS5 MATCH expects a query syntax; we keep things simple: parameter binding.
    # bm25() returns the BM25 ranking score (lower is better).
    sql = (
        "SELECT k.kb_id, k.title, k.content, k.topic, k.tags, k.author, "
        "       k.source_type, k.verified, k.trust_score, "
        "       bm25(knowledge_kb_entries_fts) AS score "
        "FROM knowledge_kb_entries_fts f "
        "JOIN knowledge_kb_entries k ON k.rowid = f.rowid "
        "WHERE knowledge_kb_entries_fts MATCH ?"
    )
    params: list[Any] = [query]
    if topic:
        sql += " AND k.topic = ?"
        params.append(topic)
    sql += " ORDER BY score LIMIT ?"
    params.append(int(limit))

    try:
        cur = conn.execute(sql, params)
    except Exception as exc:  # noqa: BLE001
        # FTS5 syntax errors (e.g. unbalanced quotes) — surface as empty result.
        logger.warning("FTS5 query failed for %r: %s", query, exc)
        return []

    col_names = [d[0] for d in cur.description]
    rows: list[dict[str, Any]] = []
    for raw in cur.fetchall():
        row = dict(zip(col_names, raw))
        if isinstance(row.get("tags"), str):
            # Mirror SqliteTableQuery._fetch_rows which parses JSON columns.
            try:
                import json

                row["tags"] = json.loads(row["tags"]) if row["tags"] else []
            except (ValueError, TypeError):
                pass
        rows.append(row)
    return rows


def vector_search_sqlite(
    db_client: Any,
    query_vector: list[float],
    topic: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """K-NN search via sqlite-vec vec0.

    Returns rows ordered by ascending distance (best first). Joins back to
    ``knowledge_kb_entries`` to populate metadata for the response payload.
    """
    if not getattr(db_client, "vec_extension_loaded", False):
        return []
    try:
        import sqlite_vec
    except ImportError:
        logger.warning("sqlite-vec not importable at query time")
        return []

    conn = db_client._get_connection()
    blob = sqlite_vec.serialize_float32(query_vector)

    # vec0 needs `k=?` literal in MATCH/k clause; kNN is selected via
    # `WHERE embedding MATCH ? AND k = ?`.
    sql = (
        "SELECT v.kb_id, v.distance, k.title, k.content, k.topic, k.tags, "
        "       k.author, k.source_type, k.verified, k.trust_score "
        "FROM knowledge_kb_vec_embeddings v "
        "JOIN knowledge_kb_entries k ON k.kb_id = v.kb_id "
        "WHERE v.embedding MATCH ? AND k = ?"
    )
    params: list[Any] = [blob, int(limit)]
    if topic:
        sql += " AND k.topic = ?"
        params.append(topic)
    sql += " ORDER BY v.distance"

    try:
        cur = conn.execute(sql, params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vector search failed: %s", exc)
        return []

    col_names = [d[0] for d in cur.description]
    rows: list[dict[str, Any]] = []
    for raw in cur.fetchall():
        row = dict(zip(col_names, raw))
        if isinstance(row.get("tags"), str):
            try:
                import json

                row["tags"] = json.loads(row["tags"]) if row["tags"] else []
            except (ValueError, TypeError):
                pass
        # Express similarity as the inverse of distance for callers; the
        # ranking position is what matters for RRF.
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Backend-specific search primitives (PostgreSQL)
# ---------------------------------------------------------------------------


def _pg_format_vector_literal(vector: list[float]) -> str:
    """Format a Python list of floats as a pgvector/halfvec literal string.

    pgvector accepts the form ``[0.1,0.2,...]`` (no spaces, square brackets)
    when cast with ``::vector`` or ``::halfvec``. We do the formatting
    ourselves so we don't require ``pgvector[psycopg2]`` registration to
    be active on every connection. Duplicated (intentionally) in lore.server
    so the write path and read path stay decoupled.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def fts_search_postgres(
    db_client: Any,
    query: str,
    topic: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """PostgreSQL full-text search using ``websearch_to_tsquery`` and ``ts_rank_cd``.

    Uses the pre-existing ``idx_kb_search`` GIN index on
    ``to_tsvector('english', title || ' ' || content)``. Returns rows ordered
    by descending rank (higher = better).
    """
    try:
        conn = db_client._get_connection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to acquire PG connection for FTS: %s", exc)
        return []

    # Dual-config FTS (Issue #10): English-stemmed config handles natural language;
    # simple config + regexp_replace splits dotted/slashed identifiers so that
    # e.g. "asyncio gather" matches content containing "asyncio.gather".
    sql = (
        "SELECT kb_id, title, topic, tags, author, source_type, verified, "
        "       trust_score, "
        "       ts_rank_cd("
        "           to_tsvector('english', "
        "               coalesce(title,'') || ' ' || coalesce(content,'')), "
        "           websearch_to_tsquery('english', %s)"
        "       ) AS score "
        "FROM knowledge.kb_entries "
        "WHERE ("
        "    to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,'')) "
        "    @@ websearch_to_tsquery('english', %s) "
        "    OR to_tsvector('simple', regexp_replace("
        "               coalesce(title,'') || ' ' || coalesce(content,''),"
        "               '[.,/\\\\:_-]', ' ', 'g')) "
        "       @@ plainto_tsquery('simple', regexp_replace(%s,"
        "               '[.,/\\\\:_-]', ' ', 'g'))"
        ")"
    )
    params: list[Any] = [query, query, query]
    if topic:
        # AND applies to the entire OR expression because it is parenthesised above.
        sql += " AND topic = %s"
        params.append(topic)
    sql += " ORDER BY score DESC NULLS LAST LIMIT %s"
    params.append(int(limit))

    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
        col_names = [d[0] for d in cursor.description]
        rows = [dict(zip(col_names, raw)) for raw in cursor.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("PG FTS query failed for %r: %s", query, exc)
        return []
    finally:
        cursor.close()
    return rows


def vector_search_postgres(
    db_client: Any,
    query_vector: list[float],
    topic: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """K-NN cosine search via pgvector HNSW index on ``knowledge.kb_embeddings``.

    Picks ``halfvec`` vs ``vector`` based on ``db_client.vector_type`` so the
    same code path works on pgvector < 0.7 (fallback) and >= 0.7 (halfvec).
    Returns rows ordered by ascending cosine distance (best match first).
    """
    if not getattr(db_client, "vec_extension_loaded", False):
        return []

    try:
        conn = db_client._get_connection()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to acquire PG connection for vector search: %s", exc)
        return []

    vt = getattr(db_client, "vector_type", "vector")
    vec_literal = _pg_format_vector_literal(query_vector)

    # CTE casts the query vector once; ORDER BY references q.qvec (already cast).
    sql = (
        f"WITH q AS (SELECT %s::{vt} AS qvec) "
        "SELECT e.kb_id, e.title, e.topic, e.tags, e.author, e.source_type, "
        "       e.verified, e.trust_score, "
        "       (em.embedding <=> q.qvec) AS distance "
        "FROM knowledge.kb_embeddings em "
        "JOIN knowledge.kb_entries e ON e.kb_id = em.kb_id "
        "CROSS JOIN q "
        "WHERE TRUE"
    )
    params: list[Any] = [vec_literal]
    if topic:
        sql += " AND e.topic = %s"
        params.append(topic)
    sql += " ORDER BY em.embedding <=> q.qvec LIMIT %s"
    params.append(int(limit))

    cursor = conn.cursor()
    try:
        cursor.execute(sql, params)
        col_names = [d[0] for d in cursor.description]
        rows = [dict(zip(col_names, raw)) for raw in cursor.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("PG vector search failed: %s", exc)
        return []
    finally:
        cursor.close()
    return rows


def hybrid_search_postgres(
    db_client: Any,
    query: str,
    *,
    topic: str | None,
    top_k: int,
    search_mode: str,
    encode_query: Any,
) -> list[dict[str, Any]]:
    """Orchestrate FTS + pgvector + RRF on PostgreSQL.

    Mirrors :func:`hybrid_search_sqlite` so :func:`handle_kb_search` can stay
    backend-agnostic at the response layer.

    Args:
        db_client: LocalPostgresClient instance.
        query: User query string.
        topic: Optional topic filter.
        top_k: Number of final results to return.
        search_mode: ``"fts"``, ``"semantic"``, or ``"hybrid"``.
        encode_query: Callable returning the query embedding, or None when no
            embedder is available (forces ``fts`` even if ``hybrid`` requested).
    """
    try:
        conn = db_client._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM knowledge.kb_entries")
        corpus_size = cur.fetchone()[0]
        cur.close()
    except Exception:  # noqa: BLE001
        corpus_size = 0
    pool = candidate_pool_size(top_k, corpus_size)

    fts_rows: list[dict[str, Any]] = []
    vec_rows: list[dict[str, Any]] = []

    if search_mode in {"fts", "hybrid"}:
        fts_rows = fts_search_postgres(db_client, query, topic, pool)

    if search_mode in {"semantic", "hybrid"} and encode_query is not None:
        try:
            query_vec = encode_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to encode query for PG semantic search: %s", exc)
            query_vec = None
        if query_vec is not None and getattr(db_client, "vec_extension_loaded", False):
            vec_rows = vector_search_postgres(db_client, query_vec, topic, pool)

    if debug_search():
        logger.info(
            "pg_search[mode=%s] query=%r fts=%d vec=%d pool=%d corpus=%d",
            search_mode,
            query,
            len(fts_rows),
            len(vec_rows),
            pool,
            corpus_size,
        )

    if search_mode == "fts":
        return [_strip_content_for_response(r) for r in fts_rows[:top_k]]
    if search_mode == "semantic":
        return [_strip_content_for_response(r) for r in vec_rows[:top_k]]

    # Hybrid: RRF fuse.
    fts_ids = [r["kb_id"] for r in fts_rows]
    vec_ids = [r["kb_id"] for r in vec_rows]

    if not fts_ids and not vec_ids:
        return []
    if not fts_ids:
        return [_strip_content_for_response(r) for r in vec_rows[:top_k]]
    if not vec_ids:
        return [_strip_content_for_response(r) for r in fts_rows[:top_k]]

    fused = reciprocal_rank_fusion([fts_ids, vec_ids])
    fused_ids = [kb_id for kb_id, _score in fused[:top_k]]

    row_by_id: dict[str, dict[str, Any]] = {}
    for r in vec_rows:
        row_by_id[r["kb_id"]] = r
    for r in fts_rows:
        row_by_id[r["kb_id"]] = r

    rrf_score_by_id = dict(fused)
    results: list[dict[str, Any]] = []
    for kb_id in fused_ids:
        row = row_by_id.get(kb_id)
        if not row:
            continue
        out = _strip_content_for_response(row)
        out["rrf_score"] = rrf_score_by_id.get(kb_id)
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _strip_content_for_response(row: dict[str, Any]) -> dict[str, Any]:
    """Drop heavy ``content`` field from response payloads (search returns metadata)."""
    out = dict(row)
    out.pop("content", None)
    return out


def hybrid_search_sqlite(
    db_client: Any,
    query: str,
    *,
    topic: str | None,
    top_k: int,
    search_mode: str,
    encode_query: Any,
) -> list[dict[str, Any]]:
    """Orchestrate FTS5, vector, and RRF based on ``search_mode``.

    Args:
        db_client: SqliteClient instance.
        query: User query string.
        topic: Optional topic filter.
        top_k: Number of final results to return.
        search_mode: ``"fts"``, ``"semantic"``, or ``"hybrid"``.
        encode_query: Callable returning the query embedding, or None if the
            caller couldn't load the embedder. Permits in-mode fallback.

    Returns: list of result dicts (kb_id + metadata + ranking score).
    """
    # Estimate corpus size for adaptive pool sizing — cheap COUNT(*) is fine.
    try:
        conn = db_client._get_connection()
        corpus_size = conn.execute("SELECT COUNT(*) FROM knowledge_kb_entries").fetchone()[0]
    except Exception:  # noqa: BLE001
        corpus_size = 0
    pool = candidate_pool_size(top_k, corpus_size)

    fts_rows: list[dict[str, Any]] = []
    vec_rows: list[dict[str, Any]] = []

    if search_mode in {"fts", "hybrid"} and getattr(db_client, "fts5_available", False):
        fts_rows = fts5_search_sqlite(db_client, query, topic, pool)

    if search_mode in {"semantic", "hybrid"} and encode_query is not None:
        try:
            query_vec = encode_query(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to encode query for semantic search: %s", exc)
            query_vec = None
        if query_vec is not None and getattr(db_client, "vec_extension_loaded", False):
            vec_rows = vector_search_sqlite(db_client, query_vec, topic, pool)

    if debug_search():
        logger.info(
            "search[mode=%s] query=%r fts=%d vec=%d pool=%d corpus=%d",
            search_mode,
            query,
            len(fts_rows),
            len(vec_rows),
            pool,
            corpus_size,
        )

    # Pure-mode returns: rank already implicit in DB ordering.
    if search_mode == "fts":
        return [_strip_content_for_response(r) for r in fts_rows[:top_k]]
    if search_mode == "semantic":
        return [_strip_content_for_response(r) for r in vec_rows[:top_k]]

    # Hybrid: fuse rankings via RRF.
    fts_ids = [r["kb_id"] for r in fts_rows]
    vec_ids = [r["kb_id"] for r in vec_rows]

    if not fts_ids and not vec_ids:
        return []
    if not fts_ids:
        return [_strip_content_for_response(r) for r in vec_rows[:top_k]]
    if not vec_ids:
        return [_strip_content_for_response(r) for r in fts_rows[:top_k]]

    fused = reciprocal_rank_fusion([fts_ids, vec_ids])
    fused_ids = [kb_id for kb_id, _score in fused[:top_k]]

    # Build kb_id -> row map preferring FTS rows (they carry bm25 score) but
    # falling back to vector rows for ids only present semantically.
    row_by_id: dict[str, dict[str, Any]] = {}
    for r in vec_rows:
        row_by_id[r["kb_id"]] = r
    for r in fts_rows:
        row_by_id[r["kb_id"]] = r

    rrf_score_by_id = dict(fused)
    results: list[dict[str, Any]] = []
    for kb_id in fused_ids:
        row = row_by_id.get(kb_id)
        if not row:
            continue
        out = _strip_content_for_response(row)
        out["rrf_score"] = rrf_score_by_id.get(kb_id)
        results.append(out)
    return results
