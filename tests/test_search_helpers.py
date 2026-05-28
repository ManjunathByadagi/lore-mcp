"""Unit tests for lore.search helper functions.

Covers the pure utility functions (config helpers, fts5_search_sqlite,
vector_search_sqlite, hybrid_search_sqlite) that are currently at ~16%
coverage. All tests use real sqlite via a minimal in-memory fake, or plain
mocks, rather than a live Postgres backend.

Chosen because search.py has 258 statements at 16.1% — the largest
coverage gap relative to impact for a user-facing module.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import MagicMock

import pytest

from lore.search import (
    _pg_format_vector_literal,
    _strip_content_for_response,
    candidate_pool_size,
    debug_search,
    default_search_mode,
    fts5_search_sqlite,
    hybrid_search_sqlite,
    reciprocal_rank_fusion,
    rrf_k,
    semantic_enabled,
    vector_search_sqlite,
)

# ---------------------------------------------------------------------------
# Config helper tests
# ---------------------------------------------------------------------------


def test_semantic_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    assert semantic_enabled() is False


def test_semantic_enabled_true_when_set(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    assert semantic_enabled() is True


def test_semantic_enabled_case_insensitive(monkeypatch):
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "TRUE")
    assert semantic_enabled() is True
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "True")
    assert semantic_enabled() is True


def test_semantic_enabled_false_for_other_values(monkeypatch):
    for v in ("1", "yes", "on", ""):
        monkeypatch.setenv("LORE_SEMANTIC_SEARCH", v)
        assert semantic_enabled() is False


def test_default_search_mode_returns_hybrid_by_default(monkeypatch):
    monkeypatch.delenv("LORE_SEARCH_MODE_DEFAULT", raising=False)
    assert default_search_mode() == "hybrid"


def test_default_search_mode_valid_values(monkeypatch):
    for v in ("fts", "semantic", "hybrid"):
        monkeypatch.setenv("LORE_SEARCH_MODE_DEFAULT", v)
        assert default_search_mode() == v


def test_default_search_mode_invalid_falls_back_to_hybrid(monkeypatch):
    monkeypatch.setenv("LORE_SEARCH_MODE_DEFAULT", "bogus")
    assert default_search_mode() == "hybrid"


def test_rrf_k_positive_clamp(monkeypatch):
    monkeypatch.setenv("LORE_RRF_K", "0")
    assert rrf_k() == 1  # max(1, 0) = 1


def test_rrf_k_negative_clamp(monkeypatch):
    monkeypatch.setenv("LORE_RRF_K", "-5")
    assert rrf_k() == 1


def test_debug_search_false_by_default(monkeypatch):
    monkeypatch.delenv("LORE_DEBUG_SEARCH", raising=False)
    assert debug_search() is False


def test_debug_search_true_when_set(monkeypatch):
    monkeypatch.setenv("LORE_DEBUG_SEARCH", "true")
    assert debug_search() is True


# ---------------------------------------------------------------------------
# _strip_content_for_response
# ---------------------------------------------------------------------------


def test_strip_content_removes_content_field():
    row = {"kb_id": "x", "title": "T", "content": "big body", "score": 0.9}
    out = _strip_content_for_response(row)
    assert "content" not in out
    assert out["kb_id"] == "x"
    assert out["title"] == "T"


def test_strip_content_ok_when_content_absent():
    row = {"kb_id": "y", "title": "T2"}
    out = _strip_content_for_response(row)
    assert out == {"kb_id": "y", "title": "T2"}


def test_strip_content_does_not_mutate_original():
    row = {"kb_id": "z", "content": "keep me"}
    _strip_content_for_response(row)
    assert "content" in row  # original intact


# ---------------------------------------------------------------------------
# _pg_format_vector_literal
# ---------------------------------------------------------------------------


def test_pg_format_vector_literal_basic():
    result = _pg_format_vector_literal([0.1, 0.2, 0.3])
    assert result.startswith("[")
    assert result.endswith("]")
    # Must contain three float reprs separated by commas, no spaces.
    inner = result[1:-1]
    parts = inner.split(",")
    assert len(parts) == 3
    assert all(float(p) == pytest.approx(expected) for p, expected in zip(parts, [0.1, 0.2, 0.3]))


def test_pg_format_vector_literal_empty():
    result = _pg_format_vector_literal([])
    assert result == "[]"


def test_pg_format_vector_literal_large_dim():
    vec = [float(i) / 100.0 for i in range(1536)]
    result = _pg_format_vector_literal(vec)
    assert result.count(",") == 1535


# ---------------------------------------------------------------------------
# fts5_search_sqlite — uses a real in-memory sqlite with a FTS5 table
# ---------------------------------------------------------------------------


def _make_fts_db_client() -> Any:
    """Return a minimal fake db_client backed by an in-memory SQLite FTS5 table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE VIRTUAL TABLE knowledge_kb_entries_fts USING fts5(title, content, kb_id UNINDEXED)"
    )
    # Also need the base table for the JOIN
    conn.execute(
        "CREATE TABLE knowledge_kb_entries ("
        "  kb_id TEXT PRIMARY KEY, title TEXT, content TEXT, topic TEXT, "
        "  tags TEXT, author TEXT, source_type TEXT, verified INTEGER, trust_score REAL"
        ")"
    )
    # Insert test rows in both tables (rowid link)
    entries = [
        (
            "kb-001",
            "Python asyncio",
            "Async IO guide for Python",
            "python",
            '["async"]',
            "sys",
            "manual",
            1,
            0.9,
        ),
        (
            "kb-002",
            "Redis caching",
            "Caching patterns with Redis",
            "infra",
            '["cache"]',
            "sys",
            "doc",
            1,
            0.8,
        ),
        (
            "kb-003",
            "Database indexing",
            "How to write fast SQL queries",
            "db",
            "[]",
            "sys",
            "blog",
            0,
            0.7,
        ),
    ]
    for e in entries:
        conn.execute("INSERT INTO knowledge_kb_entries VALUES (?,?,?,?,?,?,?,?,?)", e)
        conn.execute(
            "INSERT INTO knowledge_kb_entries_fts(title, content, kb_id) VALUES (?,?,?)",
            (e[1], e[2], e[0]),
        )
    conn.commit()

    mock = MagicMock()
    mock.fts5_available = True
    mock._get_connection.return_value = conn
    return mock


def test_fts5_search_returns_results():
    client = _make_fts_db_client()
    rows = fts5_search_sqlite(client, "Python", None, 10)
    assert len(rows) >= 1
    kb_ids = [r["kb_id"] for r in rows]
    assert "kb-001" in kb_ids


def test_fts5_search_with_topic_filter():
    client = _make_fts_db_client()
    rows = fts5_search_sqlite(client, "Python", "python", 10)
    assert all(r.get("topic") == "python" for r in rows)


def test_fts5_search_returns_empty_when_fts5_unavailable():
    client = MagicMock()
    client.fts5_available = False
    rows = fts5_search_sqlite(client, "anything", None, 10)
    assert rows == []


def test_fts5_search_handles_syntax_error_gracefully():
    """FTS5 syntax errors (unbalanced quote) must return [] not raise."""
    client = _make_fts_db_client()
    rows = fts5_search_sqlite(client, '"unclosed', None, 10)
    # Should return empty list without raising
    assert isinstance(rows, list)


def test_fts5_search_parses_json_tags():
    """Tags stored as JSON strings should be parsed back to lists."""
    client = _make_fts_db_client()
    rows = fts5_search_sqlite(client, "asyncio", None, 10)
    if rows:
        tags = rows[0].get("tags")
        assert isinstance(tags, list), f"Expected list, got {type(tags)}: {tags}"


def test_fts5_search_respects_limit():
    client = _make_fts_db_client()
    rows = fts5_search_sqlite(client, "guide", None, 1)
    assert len(rows) <= 1


# ---------------------------------------------------------------------------
# vector_search_sqlite — without real sqlite-vec (just tests early-exit paths)
# ---------------------------------------------------------------------------


def test_vector_search_sqlite_returns_empty_when_vec_unavailable():
    client = MagicMock()
    client.vec_extension_loaded = False
    rows = vector_search_sqlite(client, [0.1, 0.2], None, 5)
    assert rows == []


def test_vector_search_sqlite_returns_empty_when_import_fails(monkeypatch):
    """When sqlite_vec is not importable, return [] gracefully."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sqlite_vec":
            raise ImportError("sqlite_vec not installed")
        return original_import(name, *args, **kwargs)

    client = MagicMock()
    client.vec_extension_loaded = True

    monkeypatch.setattr(builtins, "__import__", mock_import)
    rows = vector_search_sqlite(client, [0.1, 0.2], None, 5)
    assert rows == []


# ---------------------------------------------------------------------------
# hybrid_search_sqlite — pure-mode paths using fts5_available=True but with
# no actual FTS5 table (so fts5_search_sqlite returns []); we inject controlled
# results via monkeypatching to test the RRF and fallback branches.
# ---------------------------------------------------------------------------


def _make_hybrid_client(corpus_count: int = 0) -> Any:
    """Client that reports fts5_available but yields no actual rows (empty corpus)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE knowledge_kb_entries (kb_id TEXT PRIMARY KEY)")
    conn.commit()
    mock = MagicMock()
    mock.fts5_available = True
    mock.vec_extension_loaded = False
    mock._get_connection.return_value = conn
    return mock


def test_hybrid_search_fts_mode_returns_fts_results(monkeypatch):
    """In fts mode, hybrid_search_sqlite must return FTS rows stripped of content."""
    import lore.search as search_mod

    fake_fts = [
        {"kb_id": "k1", "title": "A", "content": "body", "score": -1.0},
        {"kb_id": "k2", "title": "B", "content": "body2", "score": -2.0},
    ]
    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: fake_fts)

    client = _make_hybrid_client()
    results = hybrid_search_sqlite(
        client, "query", topic=None, top_k=5, search_mode="fts", encode_query=None
    )
    assert len(results) == 2
    # Content must be stripped
    assert all("content" not in r for r in results)
    assert [r["kb_id"] for r in results] == ["k1", "k2"]


def test_hybrid_search_both_empty_returns_empty(monkeypatch):
    """Empty FTS and vector results → empty output in hybrid mode."""
    import lore.search as search_mod

    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: [])
    monkeypatch.setattr(search_mod, "vector_search_sqlite", lambda *a, **k: [])

    client = _make_hybrid_client()
    results = hybrid_search_sqlite(
        client, "query", topic=None, top_k=5, search_mode="hybrid", encode_query=None
    )
    assert results == []


def test_hybrid_search_fts_only_when_no_vec(monkeypatch):
    """When vec is empty but FTS has results, hybrid falls back to FTS ordering."""
    import lore.search as search_mod

    fake_fts = [{"kb_id": "k1", "title": "X", "content": "c", "score": -1.0}]
    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: fake_fts)

    client = _make_hybrid_client()
    # vec_extension_loaded=False means vector_search_sqlite bails early
    results = hybrid_search_sqlite(
        client, "query", topic=None, top_k=5, search_mode="hybrid", encode_query=None
    )
    # Only FTS results, content stripped
    assert len(results) == 1
    assert results[0]["kb_id"] == "k1"


def test_hybrid_search_rrf_fusion_both_sources(monkeypatch):
    """When both FTS and vector rows exist, RRF fuses them and returns top_k."""
    import lore.search as search_mod

    fts_rows = [
        {"kb_id": "shared", "title": "S", "content": "c", "score": -1.0},
        {"kb_id": "fts-only", "title": "F", "content": "c", "score": -2.0},
    ]
    vec_rows = [
        {"kb_id": "shared", "title": "S", "distance": 0.1},
        {"kb_id": "vec-only", "title": "V", "distance": 0.2},
    ]

    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: fts_rows)
    monkeypatch.setattr(search_mod, "vector_search_sqlite", lambda *a, **k: vec_rows)

    client = _make_hybrid_client()
    client.vec_extension_loaded = True  # so vector path runs

    encode_mock = MagicMock(return_value=[0.1, 0.2])

    results = hybrid_search_sqlite(
        client,
        "query",
        topic=None,
        top_k=5,
        search_mode="hybrid",
        encode_query=encode_mock,
    )
    # "shared" should rank highest (appears in both)
    ids = [r["kb_id"] for r in results]
    assert "shared" in ids
    assert ids[0] == "shared"
    # rrf_score should be annotated on results
    assert "rrf_score" in results[0]


def test_hybrid_search_semantic_mode_uses_vec(monkeypatch):
    """In semantic mode, only vector rows are returned."""
    import lore.search as search_mod

    vec_rows = [{"kb_id": "v1", "title": "V", "distance": 0.05}]
    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: [])
    monkeypatch.setattr(search_mod, "vector_search_sqlite", lambda *a, **k: vec_rows)

    client = _make_hybrid_client()
    client.vec_extension_loaded = True

    encode_mock = MagicMock(return_value=[0.1, 0.2])
    results = hybrid_search_sqlite(
        client,
        "query",
        topic=None,
        top_k=5,
        search_mode="semantic",
        encode_query=encode_mock,
    )
    assert len(results) == 1
    assert results[0]["kb_id"] == "v1"


def test_hybrid_search_encode_exception_falls_back(monkeypatch):
    """If encode_query raises, the search continues without vector rows."""
    import lore.search as search_mod

    fts_rows = [{"kb_id": "k1", "title": "A", "content": "c", "score": -1.0}]
    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: fts_rows)

    client = _make_hybrid_client()
    client.vec_extension_loaded = True

    def bad_encoder(q):
        raise RuntimeError("model not loaded")

    results = hybrid_search_sqlite(
        client,
        "query",
        topic=None,
        top_k=5,
        search_mode="hybrid",
        encode_query=bad_encoder,
    )
    # Falls back to FTS only
    assert len(results) == 1
    assert results[0]["kb_id"] == "k1"


def test_hybrid_search_debug_mode_does_not_crash(monkeypatch):
    """LORE_DEBUG_SEARCH=true must not cause any exception."""
    import lore.search as search_mod

    monkeypatch.setenv("LORE_DEBUG_SEARCH", "true")
    monkeypatch.setattr(search_mod, "fts5_search_sqlite", lambda *a, **k: [])

    client = _make_hybrid_client()
    results = hybrid_search_sqlite(
        client, "query", topic=None, top_k=5, search_mode="fts", encode_query=None
    )
    assert isinstance(results, list)
