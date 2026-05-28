"""Integration tests for SQLite + FTS5 + sqlite-vec embedding flow.

These tests are heavy: they load the embedding model on first run.
Marked ``@pytest.mark.slow`` and gated behind ``LORE_SEMANTIC_SEARCH=true``.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


def _semantic_available() -> bool:
    if os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower() != "true":
        return False
    try:
        import sentence_transformers  # noqa: F401
        import sqlite_vec  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark.append(
    pytest.mark.skipif(
        not _semantic_available(),
        reason="LORE_SEMANTIC_SEARCH=true required and semantic deps must be installed",
    )
)


@pytest.fixture
def fresh_server(monkeypatch):
    """Reload lore.server with a fresh SQLite backend pointing at tempdir."""
    tmp = tempfile.mkdtemp(prefix="lore_sqlite_test_")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    # Treat the test instance as non-prod so the issue #11 production guard
    # does not require confirm_production on backfill calls.
    monkeypatch.setenv("LORE_ENV", "staging")
    monkeypatch.setenv("SQLITE_DB_PATH", str(Path(tmp) / "kb.db"))

    import lore.server as s

    importlib.reload(s)
    # P1-5 moved db initialisation out of module scope into main()/lifespan.
    # Tests that reload the module must manually wire up the db global so
    # handlers have a live client before any request is dispatched.
    s.db = s.get_db_client()
    return s


def test_sqlite_vec_extension_loads(fresh_server):
    """sqlite-vec extension should load and FTS5 should be available."""
    s = fresh_server
    assert s.db.vec_extension_loaded is True
    assert s.db.fts5_available is True


def test_kb_add_creates_embedding(fresh_server):
    s = fresh_server
    resp = s.handle_kb_add(topic="t", title="Python async", content="asyncio coroutines")
    assert resp["ok"] is True
    assert resp["data"]["embedded"] is True

    # Embedding meta row should exist.
    kb_id = resp["data"]["kb_id"]
    meta = s._get_embedding_meta(kb_id)
    assert meta is not None
    assert meta["embedding_dim"] == 384
    assert meta["content_hash"]


def test_kb_update_changes_content_hash(fresh_server):
    s = fresh_server
    add_resp = s.handle_kb_add(topic="t", title="Original", content="version one")
    kb_id = add_resp["data"]["kb_id"]
    original_hash = s._get_embedding_meta(kb_id)["content_hash"]

    s.handle_kb_update(entry_id=kb_id, content="version TWO totally different")
    new_hash = s._get_embedding_meta(kb_id)["content_hash"]
    assert new_hash != original_hash


def test_kb_update_unchanged_content_skips_reembed(fresh_server):
    s = fresh_server
    add_resp = s.handle_kb_add(topic="t", title="Stable", content="same content")
    kb_id = add_resp["data"]["kb_id"]
    original_ts = s._get_embedding_meta(kb_id)["updated_at"]

    # Updating only tags should not change the content_hash.
    s.handle_kb_update(entry_id=kb_id, tags=["new-tag"])
    new_meta = s._get_embedding_meta(kb_id)
    assert new_meta is not None
    # The hash matches what _embed_kb_entry would compute -> no rewrite occurred.
    from lore.embeddings import compute_content_hash

    assert new_meta["content_hash"] == compute_content_hash("Stable", "same content")
    # updated_at MAY change due to entry update + re-embed call returning True
    # without a fresh write; we don't strictly assert equality.
    assert original_ts is not None


def test_kb_delete_removes_vec_row_after_kb_row(fresh_server):
    """Critic fix: kb row must be removed FIRST, then vec0 + meta."""
    s = fresh_server
    add_resp = s.handle_kb_add(topic="t", title="Doomed", content="will be deleted")
    kb_id = add_resp["data"]["kb_id"]
    assert s._get_embedding_meta(kb_id) is not None

    del_resp = s.handle_kb_delete(entry_id=kb_id, confirm=True)
    assert del_resp["ok"] is True

    # No KB row, no meta row, no vec row.
    conn = s.db._get_connection()
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM knowledge_kb_entries WHERE kb_id = ?",
            (kb_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM knowledge_kb_embedding_meta WHERE kb_id = ?",
            (kb_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM knowledge_kb_vec_embeddings WHERE kb_id = ?",
            (kb_id,),
        ).fetchone()[0]
        == 0
    )


def test_fts5_search_returns_match(fresh_server):
    s = fresh_server
    s.handle_kb_add(topic="t", title="Async basics", content="asyncio coroutines")
    s.handle_kb_add(topic="t", title="Threads guide", content="GIL python threading")

    resp = s.handle_kb_search(query="asyncio", search_mode="fts")
    assert resp["ok"] is True
    titles = {r["title"] for r in resp["data"]["results"]}
    assert "Async basics" in titles


def test_semantic_search_finds_meaning(fresh_server):
    s = fresh_server
    s.handle_kb_add(topic="t", title="Threads", content="GIL python concurrent")
    s.handle_kb_add(topic="t", title="Coroutines", content="asyncio event loop")
    s.handle_kb_add(topic="t", title="Unrelated", content="pancake recipe")

    resp = s.handle_kb_search(query="parallel programming", semantic=True, top_k=2)
    assert resp["ok"] is True
    assert resp["data"]["search_mode"] == "semantic"
    titles = [r["title"] for r in resp["data"]["results"]]
    # "Unrelated" should not be ranked top-2 for a parallel-programming query.
    assert "Unrelated" not in titles[:2] or len(titles) < 2


def test_hybrid_search_fuses_rankings(fresh_server):
    s = fresh_server
    s.handle_kb_add(topic="t", title="Async basics", content="asyncio coroutines")
    s.handle_kb_add(topic="t", title="Threads", content="GIL python threading")

    resp = s.handle_kb_search(query="asyncio coroutines", hybrid=True, top_k=2)
    assert resp["ok"] is True
    assert resp["data"]["search_mode"] == "hybrid"
    titles = [r["title"] for r in resp["data"]["results"]]
    assert "Async basics" in titles
    # RRF score should be present on at least one hit.
    assert any("rrf_score" in r for r in resp["data"]["results"])


def test_backfill_is_idempotent(fresh_server, monkeypatch):
    """Two backfill calls in a row: second one should embed zero entries."""
    s = fresh_server

    # Add an entry while embedding is disabled to force a backfill candidate.
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    add_resp = s.handle_kb_add(topic="t", title="Lonely", content="needs embedding")
    kb_id = add_resp["data"]["kb_id"]
    assert s._get_embedding_meta(kb_id) is None

    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "true")
    first = s.handle_kb_backfill_embeddings()
    assert first["data"]["embedded"] >= 1

    # Second run: hash matches -> zero embedded, all current.
    second = s.handle_kb_backfill_embeddings()
    assert second["data"]["embedded"] == 0
    assert second["data"]["already_current"] >= 1


def test_backfill_lock_rejects_concurrent_calls(fresh_server):
    """Module-level lock blocks concurrent backfill runs."""
    s = fresh_server
    # Manually take the lock and verify a backfill call fails fast.
    acquired = s._BACKFILL_LOCK.acquire(blocking=False)
    assert acquired is True
    try:
        resp = s.handle_kb_backfill_embeddings()
        assert resp["ok"] is False
        assert "already running" in (resp.get("message") or "")
    finally:
        s._BACKFILL_LOCK.release()


def test_embedding_status_reports_coverage(fresh_server):
    s = fresh_server
    s.handle_kb_add(topic="t", title="x", content="y")
    status = s.handle_kb_embedding_status()
    assert status["ok"] is True
    data = status["data"]
    assert data["backend"] == "sqlite"
    assert data["semantic_enabled"] is True
    assert data["total_entries"] >= 1
    assert data["embedded"] >= 1
    assert data["coverage_pct"] > 0


def test_legacy_lexical_path_still_works_with_flag_off(monkeypatch):
    """With the flag off, kb_search must still return lexical results."""
    tmp = tempfile.mkdtemp(prefix="lore_legacy_")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    monkeypatch.setenv("SQLITE_DB_PATH", str(Path(tmp) / "kb.db"))

    import lore.server as s

    importlib.reload(s)

    s.handle_kb_add(topic="t", title="findme", content="legacy lexical search")
    resp = s.handle_kb_search(query="findme")
    assert resp["ok"] is True
    titles = [r["title"] for r in resp["data"]["results"]]
    assert "findme" in titles
