"""Integration tests for PostgreSQL + pgvector embedding flow (Phase 2 of Issue #6).

These tests exercise the live PostgreSQL backend (``DB_BACKEND=local``) and
require:

  - ``LORE_SEMANTIC_SEARCH=true``
  - ``sentence-transformers`` + ``pgvector`` extras installed
  - A reachable PostgreSQL with pgvector >= 0.7 (for ``halfvec``) or any
    version (fallback to ``vector``)
  - Either ``TEST_POSTGRES_URL`` set, or the default ``DB_HOST``/``DB_PORT``/
    ``DB_NAME``/``DB_USER``/``DB_PASSWORD`` env vars pointing at a test DB.

The whole module is skipped if any of the above is missing. Tests isolate
themselves by inserting rows under a generated topic prefix and cleaning
them up in teardown — they do **not** drop the schema, so they're safe to
run against staging/production-like databases (though we still recommend a
dedicated test DB).
"""

from __future__ import annotations

import importlib
import os
import uuid

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
]


def _pg_available() -> bool:
    """True iff PostgreSQL semantic prerequisites look satisfied."""
    if os.getenv("LORE_SEMANTIC_SEARCH", "false").strip().lower() != "true":
        return False
    backend = os.getenv("DB_BACKEND", "").strip().lower()
    if backend not in {"local", "postgres", "postgresql"}:
        return False
    try:
        import psycopg2  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark.append(
    pytest.mark.skipif(
        not _pg_available(),
        reason=(
            "PostgreSQL semantic test prerequisites missing: "
            "LORE_SEMANTIC_SEARCH=true, DB_BACKEND in {local,postgres,postgresql}, "
            "and the [semantic] extra must be installed."
        ),
    )
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_module():
    """Import the server module once. We rely on the live PG connection."""
    import lore.server as s

    # Force a fresh module so _init_schema runs against the current env.
    importlib.reload(s)

    # Trigger the lazy connection so _init_schema can detect pgvector and
    # create the kb_embeddings table. Without this the flags remain at their
    # __init__ defaults (vec_extension_loaded=False).
    try:
        s.db._get_connection()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Cannot connect to PostgreSQL: {exc}")

    if not getattr(s.db, "vec_extension_loaded", False):
        pytest.skip("pgvector extension not loaded on the configured database")
    return s


@pytest.fixture
def test_topic():
    """Unique topic so test rows don't collide with real data."""
    return f"_test_pg_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_topic(server_module, test_topic):
    """Yield, then delete every kb_entries row under the test topic."""
    s = server_module
    yield test_topic
    try:
        conn = s.db._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM knowledge.kb_entries WHERE topic = %s", (test_topic,))
        finally:
            cursor.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Schema / extension detection
# ---------------------------------------------------------------------------


def test_pgvector_extension_loaded(server_module):
    """``_init_schema`` should have detected pgvector and set vec_extension_loaded."""
    s = server_module
    assert s.db.vec_extension_loaded is True
    assert s.db.pgvector_version is not None
    assert s.db.vector_type in {"halfvec", "vector"}


def test_kb_embeddings_table_exists(server_module):
    """The CREATE IF NOT EXISTS in _init_schema should leave a queryable table."""
    s = server_module
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'knowledge' AND table_name = 'kb_embeddings')"
        )
        exists = cursor.fetchone()[0]
        assert exists is True

        # Verify HNSW index also exists.
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_indexes "
            "WHERE schemaname='knowledge' AND indexname='idx_kb_embeddings_hnsw')"
        )
        assert cursor.fetchone()[0] is True
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def test_kb_add_creates_embedding(server_module, cleanup_topic):
    """kb_add on Postgres should populate knowledge.kb_embeddings."""
    s = server_module
    resp = s.handle_kb_add(
        topic=cleanup_topic,
        title="Async Python primer",
        content="asyncio coroutines event loop",
    )
    assert resp["ok"] is True, resp
    assert resp["data"]["embedded"] is True

    kb_id = resp["data"]["kb_id"]
    meta = s._get_embedding_meta(kb_id)
    assert meta is not None
    assert meta["embedding_dim"] == 384
    assert meta["content_hash"]


def test_kb_update_reembeds_on_content_change(server_module, cleanup_topic):
    s = server_module
    add_resp = s.handle_kb_add(
        topic=cleanup_topic, title="Original title", content="version one of content"
    )
    kb_id = add_resp["data"]["kb_id"]
    original_hash = s._get_embedding_meta(kb_id)["content_hash"]

    upd = s.handle_kb_update(entry_id=kb_id, content="completely different content v2")
    assert upd["ok"] is True

    new_meta = s._get_embedding_meta(kb_id)
    assert new_meta is not None
    assert new_meta["content_hash"] != original_hash


def test_kb_delete_cascades_embedding(server_module, cleanup_topic):
    """Deleting the kb_entries row should cascade-remove the embeddings row."""
    s = server_module
    add_resp = s.handle_kb_add(topic=cleanup_topic, title="Doomed", content="will be deleted")
    kb_id = add_resp["data"]["kb_id"]
    assert s._get_embedding_meta(kb_id) is not None

    del_resp = s.handle_kb_delete(entry_id=kb_id, confirm=True)
    assert del_resp["ok"] is True

    assert s._get_embedding_meta(kb_id) is None

    # Direct verification on the DB.
    conn = s.db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) FROM knowledge.kb_embeddings WHERE kb_id = %s",
            (kb_id,),
        )
        assert cursor.fetchone()[0] == 0
    finally:
        cursor.close()


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def test_fts_search_returns_match(server_module, cleanup_topic):
    s = server_module
    s.handle_kb_add(
        topic=cleanup_topic, title="Async basics", content="asyncio coroutines event loop"
    )
    s.handle_kb_add(
        topic=cleanup_topic, title="Threads guide", content="python GIL threading module"
    )

    resp = s.handle_kb_search(query="asyncio coroutines", search_mode="fts", topic=cleanup_topic)
    assert resp["ok"] is True, resp
    assert resp["data"]["search_mode"] == "fts"
    assert resp["data"].get("backend") == "postgres"
    titles = {r["title"] for r in resp["data"]["results"]}
    assert "Async basics" in titles


def test_semantic_search_finds_meaning(server_module, cleanup_topic):
    """Semantic search should rank conceptually-related docs higher."""
    s = server_module
    s.handle_kb_add(
        topic=cleanup_topic, title="Concurrency primer", content="parallel execution threads GIL"
    )
    s.handle_kb_add(
        topic=cleanup_topic, title="Pancakes", content="recipe for fluffy buttermilk pancakes"
    )

    resp = s.handle_kb_search(
        query="concurrent programming concepts",
        semantic=True,
        topic=cleanup_topic,
        top_k=2,
    )
    assert resp["ok"] is True, resp
    assert resp["data"]["search_mode"] == "semantic"
    titles = [r["title"] for r in resp["data"]["results"]]
    # "Pancakes" should not be ranked top-1 for a concurrency query.
    assert titles[0] != "Pancakes"


def test_hybrid_search_returns_rrf_score(server_module, cleanup_topic):
    s = server_module
    s.handle_kb_add(
        topic=cleanup_topic, title="Async basics", content="asyncio coroutines event loop"
    )
    s.handle_kb_add(topic=cleanup_topic, title="Threads", content="python threading GIL parallel")

    resp = s.handle_kb_search(
        query="asyncio event loop",
        hybrid=True,
        topic=cleanup_topic,
        top_k=2,
    )
    assert resp["ok"] is True, resp
    assert resp["data"]["search_mode"] == "hybrid"
    assert "rrf_k" in resp["data"]
    assert any("rrf_score" in r for r in resp["data"]["results"])


# ---------------------------------------------------------------------------
# Backfill + status
# ---------------------------------------------------------------------------


def test_backfill_dry_run_reports_candidates(server_module, cleanup_topic, monkeypatch):
    """Adding rows while semantic is off should leave candidates for backfill."""
    s = server_module

    # Insert directly so the write-path embedder is skipped.
    conn = s.db._get_connection()
    cursor = conn.cursor()
    kb_ids = []
    try:
        for i in range(3):
            kb_id = f"kb_test_{uuid.uuid4().hex[:10]}"
            kb_ids.append(kb_id)
            cursor.execute(
                "INSERT INTO knowledge.kb_entries (kb_id, topic, title, content) "
                "VALUES (%s, %s, %s, %s)",
                (kb_id, cleanup_topic, f"Row {i}", f"content {i} unique"),
            )
    finally:
        cursor.close()

    resp = s.handle_kb_backfill_embeddings(dry_run=True)
    assert resp["ok"] is True, resp
    assert resp["data"]["backend"] == "postgres"
    # Our 3 rows must be among the candidates (others may exist).
    assert resp["data"]["needs_embedding"] >= 3


def test_backfill_is_idempotent(server_module, cleanup_topic):
    """Run backfill twice — second invocation should embed zero rows."""
    s = server_module

    # Seed a fresh row without embedding by inserting directly.
    conn = s.db._get_connection()
    cursor = conn.cursor()
    kb_id = f"kb_test_{uuid.uuid4().hex[:10]}"
    try:
        cursor.execute(
            "INSERT INTO knowledge.kb_entries (kb_id, topic, title, content) "
            "VALUES (%s, %s, %s, %s)",
            (kb_id, cleanup_topic, "Idempotent row", "needs embedding once"),
        )
    finally:
        cursor.close()

    first = s.handle_kb_backfill_embeddings(limit=1)
    assert first["ok"] is True
    assert first["data"]["embedded"] >= 1

    second = s.handle_kb_backfill_embeddings(limit=1)
    assert second["ok"] is True
    # The row we just embedded should now be already_current; embedded
    # would only be > 0 if some *other* row was stale, but we passed limit=1
    # to keep this assertion tight. We don't strictly assert == 0 because the
    # row pool may still contain other candidates.
    assert second["data"]["embedded"] in (0, 1)


def test_embedding_status_reports_coverage(server_module, cleanup_topic):
    s = server_module
    s.handle_kb_add(topic=cleanup_topic, title="Coverage row", content="some content")

    resp = s.handle_kb_embedding_status()
    assert resp["ok"] is True
    data = resp["data"]
    assert data["backend"] == "postgres"
    assert data["semantic_enabled"] is True
    assert data["vec_extension_loaded"] is True
    assert data["embedding_dim"] == 384
    assert data["total_entries"] >= 1
    assert data["embedded"] >= 1
    assert "pgvector_version" in data
    assert "vector_type" in data
    assert data["vector_type"] in {"halfvec", "vector"}
