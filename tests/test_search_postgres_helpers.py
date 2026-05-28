"""Unit tests for search.py PostgreSQL helper functions.

Targets the uncovered Postgres search paths:
- fts_search_postgres (connection failure, query execution with mock cursor)
- vector_search_postgres (no vec extension, connection failure, with mock cursor)
- hybrid_search_postgres (fts-only, semantic-only, hybrid with both)
- _pg_format_vector_literal
- hybrid_search_sqlite with no FTS/vec (pure orchestration branches)
- _strip_content_for_response

Uses lightweight mock objects that simulate psycopg2 cursors without
requiring a live PostgreSQL database. The mocks implement the minimal
cursor interface (execute, fetchall, description, close) and connection
interface (_get_connection) that the search helpers consume.
"""

from __future__ import annotations

import pytest

import lore.search as srch

# ---------------------------------------------------------------------------
# _pg_format_vector_literal
# ---------------------------------------------------------------------------


def test_pg_format_vector_literal_basic():
    result = srch._pg_format_vector_literal([0.1, 0.2, 0.3])
    assert result.startswith("[")
    assert result.endswith("]")
    # Contains the float values
    assert "0.1" in result or "0.10" in result


def test_pg_format_vector_literal_empty():
    result = srch._pg_format_vector_literal([])
    assert result == "[]"


def test_pg_format_vector_literal_single():
    result = srch._pg_format_vector_literal([1.0])
    assert result == "[1.0]"


# ---------------------------------------------------------------------------
# Mock cursor / connection helpers
# ---------------------------------------------------------------------------


class _MockCursor:
    """Minimal psycopg2-like cursor for testing search helpers."""

    def __init__(self, rows=None, col_names=None, count=0):
        self._rows = rows or []
        self._col_names = col_names or []
        self._count = count
        self.closed = False
        self.description = [(name,) for name in (col_names or [])]

    def execute(self, sql, params=None):
        pass  # no-op

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if self._rows:
            return self._rows[0]
        return (self._count,)

    def close(self):
        self.closed = True


class _MockConnection:
    """Minimal psycopg2-like connection."""

    def __init__(self, cursor: _MockCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


class _MockPgClient:
    """Fake LocalPostgresClient returning a controlled cursor."""

    def __init__(
        self,
        rows=None,
        col_names=None,
        count=0,
        fail_connect=False,
        vec_loaded=True,
        vector_type="vector",
    ):
        self._rows = rows or []
        self._col_names = col_names or []
        self._count = count
        self._fail_connect = fail_connect
        self.vec_extension_loaded = vec_loaded
        self.vector_type = vector_type

    def _get_connection(self):
        if self._fail_connect:
            raise RuntimeError("connection refused")
        return _MockConnection(_MockCursor(self._rows, self._col_names, self._count))


# ---------------------------------------------------------------------------
# fts_search_postgres
# ---------------------------------------------------------------------------


def test_fts_search_postgres_connection_failure():
    """Connection failure returns an empty list, no raise."""
    client = _MockPgClient(fail_connect=True)
    result = srch.fts_search_postgres(client, "python", None, 10)
    assert result == []


def test_fts_search_postgres_returns_rows():
    """With a working cursor, rows are returned as dicts."""
    rows = [("kb_1", "Python tips", "python", "[]", None, None, None, 1.0, 0.8)]
    col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "score",
    ]
    client = _MockPgClient(rows=rows, col_names=col_names)
    result = srch.fts_search_postgres(client, "python", None, 10)
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_1"
    assert result[0]["score"] == 0.8


def test_fts_search_postgres_with_topic_filter():
    """Topic filter is included in the query (no crash); rows still returned."""
    rows = [("kb_t", "Topic entry", "python", "[]", None, None, None, 1.0, 0.5)]
    col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "score",
    ]
    client = _MockPgClient(rows=rows, col_names=col_names)
    result = srch.fts_search_postgres(client, "async", "python", 10)
    assert len(result) == 1
    assert result[0]["topic"] == "python"


def test_fts_search_postgres_empty_results():
    """Empty result set returns empty list."""
    client = _MockPgClient(
        rows=[],
        col_names=[
            "kb_id",
            "title",
            "topic",
            "tags",
            "author",
            "source_type",
            "verified",
            "trust_score",
            "score",
        ],
    )
    result = srch.fts_search_postgres(client, "notfound", None, 10)
    assert result == []


def test_fts_search_postgres_cursor_execute_error():
    """If execute raises, returns empty list gracefully."""

    class _BrokenCursor(_MockCursor):
        def execute(self, sql, params=None):
            raise RuntimeError("syntax error in query")

    class _BrokenClient:
        def _get_connection(self):
            return _MockConnection(_BrokenCursor())

    result = srch.fts_search_postgres(_BrokenClient(), "test", None, 10)
    assert result == []


# ---------------------------------------------------------------------------
# vector_search_postgres
# ---------------------------------------------------------------------------


def test_vector_search_postgres_no_vec_extension():
    """Returns empty list when vec_extension_loaded=False."""
    client = _MockPgClient(vec_loaded=False)
    result = srch.vector_search_postgres(client, [0.1] * 5, None, 10)
    assert result == []


def test_vector_search_postgres_connection_failure():
    client = _MockPgClient(fail_connect=True, vec_loaded=True)
    result = srch.vector_search_postgres(client, [0.1] * 5, None, 10)
    assert result == []


def test_vector_search_postgres_returns_rows():
    rows = [("kb_v1", "Vector entry", "ml", "[]", None, None, None, 1.0, 0.05)]
    col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "distance",
    ]
    client = _MockPgClient(rows=rows, col_names=col_names, vec_loaded=True)
    result = srch.vector_search_postgres(client, [0.1, 0.2, 0.3], None, 10)
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_v1"


def test_vector_search_postgres_with_topic_filter():
    rows = [("kb_v2", "ML entry", "ml", "[]", None, None, None, 0.9, 0.1)]
    col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "distance",
    ]
    client = _MockPgClient(rows=rows, col_names=col_names, vec_loaded=True)
    result = srch.vector_search_postgres(client, [0.1, 0.2], "ml", 10)
    assert len(result) == 1


def test_vector_search_postgres_halfvec_type():
    """halfvec vector_type is used without crash."""
    rows = [("kb_hv", "Half vec entry", "ml", "[]", None, None, None, 1.0, 0.03)]
    col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "distance",
    ]
    client = _MockPgClient(rows=rows, col_names=col_names, vec_loaded=True, vector_type="halfvec")
    result = srch.vector_search_postgres(client, [0.1, 0.2], None, 10)
    assert len(result) == 1


def test_vector_search_postgres_cursor_error():
    """Cursor execute error returns empty list."""

    class _BrokenCursor(_MockCursor):
        def execute(self, sql, params=None):
            raise RuntimeError("pgvector error")

    class _BrokenClient:
        vec_extension_loaded = True
        vector_type = "vector"

        def _get_connection(self):
            return _MockConnection(_BrokenCursor())

    result = srch.vector_search_postgres(_BrokenClient(), [0.1, 0.2], None, 10)
    assert result == []


# ---------------------------------------------------------------------------
# hybrid_search_postgres
# ---------------------------------------------------------------------------


def _make_pg_client_for_hybrid(fts_rows, vec_rows, corpus_count=100):
    """Create a mock PG client that returns fts_rows or vec_rows based on SQL."""

    fts_col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "score",
    ]
    vec_col_names = [
        "kb_id",
        "title",
        "topic",
        "tags",
        "author",
        "source_type",
        "verified",
        "trust_score",
        "distance",
    ]

    class _MultiCursor:
        """Returns different data depending on the SQL executed."""

        def __init__(self):
            self._result_rows = []
            self._col_names = []
            self._fetchone_val = (corpus_count,)

        def execute(self, sql, params=None):
            if "COUNT(*)" in sql:
                self._result_rows = [(corpus_count,)]
                self._col_names = ["count"]
            elif "kb_embeddings" in sql or "halfvec" in sql or "qvec" in sql:
                # vector search
                self._result_rows = vec_rows
                self._col_names = vec_col_names
            else:
                # FTS search
                self._result_rows = fts_rows
                self._col_names = fts_col_names

        @property
        def description(self):
            return [(name,) for name in self._col_names]

        def fetchall(self):
            return self._result_rows

        def fetchone(self):
            return (corpus_count,)

        def close(self):
            pass

    class _MultiConn:
        def __init__(self):
            self._cursor = _MultiCursor()

        def cursor(self):
            return self._cursor

    class _MultiClient:
        vec_extension_loaded = True
        vector_type = "vector"

        def _get_connection(self):
            return _MultiConn()

    return _MultiClient()


def test_hybrid_search_postgres_fts_mode():
    """search_mode='fts' returns only FTS rows."""
    fts_rows = [("kb_f1", "FTS entry", "python", "[]", None, None, None, 1.0, 0.8)]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=[])
    result = srch.hybrid_search_postgres(
        client, "python", topic=None, top_k=10, search_mode="fts", encode_query=lambda q: [0.1, 0.2]
    )
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_f1"
    # content should be stripped from response
    assert "content" not in result[0]


def test_hybrid_search_postgres_semantic_mode():
    """search_mode='semantic' returns only vector rows."""
    vec_rows = [("kb_v1", "Vec entry", "ml", "[]", None, None, None, 1.0, 0.05)]
    client = _make_pg_client_for_hybrid(fts_rows=[], vec_rows=vec_rows)
    result = srch.hybrid_search_postgres(
        client,
        "vector search",
        topic=None,
        top_k=10,
        search_mode="semantic",
        encode_query=lambda q: [0.1, 0.2],
    )
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_v1"


def test_hybrid_search_postgres_hybrid_mode_fuses():
    """Hybrid mode RRF-fuses FTS and vector results."""
    fts_rows = [
        ("kb_both", "Both entry", "ml", "[]", None, None, None, 1.0, 0.9),
        ("kb_fts_only", "FTS only", "ml", "[]", None, None, None, 1.0, 0.5),
    ]
    vec_rows = [
        ("kb_both", "Both entry", "ml", "[]", None, None, None, 1.0, 0.05),
        ("kb_vec_only", "Vec only", "ml", "[]", None, None, None, 1.0, 0.1),
    ]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=vec_rows)
    result = srch.hybrid_search_postgres(
        client,
        "test",
        topic=None,
        top_k=10,
        search_mode="hybrid",
        encode_query=lambda q: [0.1, 0.2],
    )
    # kb_both should be first (appears in both lists → higher RRF score)
    assert len(result) >= 1
    assert result[0]["kb_id"] == "kb_both"


def test_hybrid_search_postgres_hybrid_no_vec_rows():
    """When only FTS rows exist, returns them (not empty)."""
    fts_rows = [("kb_f", "FTS only", "ml", "[]", None, None, None, 1.0, 0.7)]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=[])
    result = srch.hybrid_search_postgres(
        client,
        "test",
        topic=None,
        top_k=10,
        search_mode="hybrid",
        encode_query=lambda q: [0.1, 0.2],
    )
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_f"


def test_hybrid_search_postgres_hybrid_no_fts_rows():
    """When only vector rows exist, returns them (not empty)."""
    vec_rows = [("kb_v", "Vec only", "ml", "[]", None, None, None, 1.0, 0.1)]
    client = _make_pg_client_for_hybrid(fts_rows=[], vec_rows=vec_rows)
    result = srch.hybrid_search_postgres(
        client,
        "test",
        topic=None,
        top_k=10,
        search_mode="hybrid",
        encode_query=lambda q: [0.1, 0.2],
    )
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_v"


def test_hybrid_search_postgres_hybrid_no_rows():
    """When both FTS and vector return empty, returns empty list."""
    client = _make_pg_client_for_hybrid(fts_rows=[], vec_rows=[])
    result = srch.hybrid_search_postgres(
        client,
        "test",
        topic=None,
        top_k=10,
        search_mode="hybrid",
        encode_query=lambda q: [0.1, 0.2],
    )
    assert result == []


def test_hybrid_search_postgres_encode_query_none():
    """encode_query=None skips vector search; returns FTS only."""
    fts_rows = [("kb_fts", "FTS entry", "ml", "[]", None, None, None, 1.0, 0.6)]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=[])
    result = srch.hybrid_search_postgres(
        client, "test", topic=None, top_k=10, search_mode="hybrid", encode_query=None
    )
    # FTS rows returned since encode_query is None (no vector search)
    assert len(result) == 1


def test_hybrid_search_postgres_connection_failure_corpus():
    """Connection failure for corpus count defaults to 0, doesn't crash."""

    class _FailClient:
        vec_extension_loaded = True
        vector_type = "vector"

        def _get_connection(self):
            raise RuntimeError("can't connect")

    result = srch.hybrid_search_postgres(
        _FailClient(), "test", topic=None, top_k=10, search_mode="fts", encode_query=None
    )
    # Should return empty (can't get connection for FTS either)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _strip_content_for_response
# ---------------------------------------------------------------------------


def test_strip_content_removes_content_field():
    row = {"kb_id": "x", "title": "T", "content": "heavy content here"}
    out = srch._strip_content_for_response(row)
    assert "content" not in out
    assert out["kb_id"] == "x"
    assert out["title"] == "T"


def test_strip_content_passthrough_when_no_content():
    row = {"kb_id": "x", "title": "T", "score": 0.5}
    out = srch._strip_content_for_response(row)
    assert out == {"kb_id": "x", "title": "T", "score": 0.5}


def test_strip_content_does_not_mutate_original():
    row = {"kb_id": "x", "content": "should stay"}
    out = srch._strip_content_for_response(row)
    assert "content" in row  # original unchanged
    assert "content" not in out


# ---------------------------------------------------------------------------
# hybrid_search_postgres — encode_query exception path
# ---------------------------------------------------------------------------


def test_hybrid_search_postgres_encode_query_raises():
    """If encode_query raises, query_vec becomes None and vector search is skipped."""
    fts_rows = [("kb_fts_ok", "FTS entry", "ml", "[]", None, None, None, 1.0, 0.8)]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=[])

    def _fail_encode(q):
        raise RuntimeError("embedding model unavailable")

    result = srch.hybrid_search_postgres(
        client, "test", topic=None, top_k=10, search_mode="hybrid", encode_query=_fail_encode
    )
    # FTS rows are returned (vector search skipped due to exception)
    assert len(result) == 1
    assert result[0]["kb_id"] == "kb_fts_ok"


def test_hybrid_search_postgres_debug_mode(monkeypatch):
    """debug_search()=True doesn't crash hybrid_search_postgres."""
    monkeypatch.setenv("LORE_DEBUG_SEARCH", "true")
    fts_rows = [("kb_d1", "Debug entry", "ml", "[]", None, None, None, 1.0, 0.5)]
    client = _make_pg_client_for_hybrid(fts_rows=fts_rows, vec_rows=[])
    result = srch.hybrid_search_postgres(
        client, "debug test", topic=None, top_k=10, search_mode="fts", encode_query=None
    )
    assert isinstance(result, list)
    monkeypatch.delenv("LORE_DEBUG_SEARCH", raising=False)


# ---------------------------------------------------------------------------
# hybrid_search_sqlite — pure orchestration branches (no real SQLite needed)
# ---------------------------------------------------------------------------


class _SqliteFakeClient:
    """Minimal fake for hybrid_search_sqlite that has no FTS5 or vec."""

    def __init__(self, fts5=False, vec=False, fail_count=False):
        self.fts5_available = fts5
        self.vec_extension_loaded = vec
        self._fail_count = fail_count

    def _get_connection(self):
        if self._fail_count:
            raise RuntimeError("no connection")

        class _Conn:
            def execute(self, sql, *args):
                class _Cur:
                    def fetchone(self):
                        return (100,)

                return _Cur()

        return _Conn()


def test_hybrid_search_sqlite_no_fts5_no_vec_returns_empty():
    """With no FTS5 and no vec extension, both row sets are empty → empty result."""
    client = _SqliteFakeClient(fts5=False, vec=False)
    result = srch.hybrid_search_sqlite(
        client,
        "test",
        topic=None,
        top_k=10,
        search_mode="hybrid",
        encode_query=lambda q: [0.1, 0.2],
    )
    assert result == []


def test_hybrid_search_sqlite_fts_mode_no_fts5():
    """search_mode='fts' but no FTS5 → empty rows returned."""
    client = _SqliteFakeClient(fts5=False, vec=False)
    result = srch.hybrid_search_sqlite(
        client, "test", topic=None, top_k=10, search_mode="fts", encode_query=None
    )
    assert result == []


def test_hybrid_search_sqlite_semantic_mode_no_vec():
    """search_mode='semantic' but no vec → encode called but no results."""
    client = _SqliteFakeClient(fts5=False, vec=False)
    encode_called = []

    def _enc(q):
        encode_called.append(q)
        return [0.1, 0.2]

    result = srch.hybrid_search_sqlite(
        client, "test", topic=None, top_k=10, search_mode="semantic", encode_query=_enc
    )
    # encode is called but vector search sees vec_extension_loaded=False → empty
    assert result == []
    assert encode_called  # encode WAS called


def test_hybrid_search_sqlite_encode_query_raises():
    """If encode_query raises, semantic results are skipped (query_vec=None)."""
    client = _SqliteFakeClient(fts5=False, vec=False)

    def _fail(q):
        raise RuntimeError("embedding unavailable")

    result = srch.hybrid_search_sqlite(
        client, "test", topic=None, top_k=10, search_mode="hybrid", encode_query=_fail
    )
    # No crash — vec_rows stays empty
    assert result == []


def test_hybrid_search_sqlite_connection_failure_for_count():
    """Corpus count failure defaults to 0 (doesn't crash)."""
    client = _SqliteFakeClient(fts5=False, vec=False, fail_count=True)
    result = srch.hybrid_search_sqlite(
        client, "test", topic=None, top_k=10, search_mode="fts", encode_query=None
    )
    assert result == []  # no FTS5 anyway


# ---------------------------------------------------------------------------
# candidate_pool_size (already covered but ensure the rounding branches)
# ---------------------------------------------------------------------------


def test_candidate_pool_size_small_corpus():
    result = srch.candidate_pool_size(10, 50)
    assert isinstance(result, int)
    assert result >= 10


def test_candidate_pool_size_large_corpus():
    result = srch.candidate_pool_size(20, 10000)
    assert result >= 20
