"""Verify graceful degradation when semantic features are unavailable.

These tests intentionally avoid loading the embedding model — they only
check that the code paths exist, gates work, and the search module
short-circuits correctly.
"""

from __future__ import annotations

import os
import tempfile

import pytest


def _fresh_sqlite_client(semantic: bool, tmp_path):
    """Create a clean SqliteClient for the requested semantic flag state."""
    db_path = str(tmp_path / "kb.db")
    if semantic:
        os.environ["LORE_SEMANTIC_SEARCH"] = "true"
    else:
        os.environ["LORE_SEMANTIC_SEARCH"] = "false"
    from lore.db_client import SqliteClient

    return SqliteClient(db_path=db_path)


@pytest.fixture
def tmp_path_fixture():
    with tempfile.TemporaryDirectory(prefix="lore_degrad_") as d:
        from pathlib import Path

        yield Path(d)


def test_semantic_flag_default_off(monkeypatch):
    """Default state: LORE_SEMANTIC_SEARCH is False."""
    monkeypatch.delenv("LORE_SEMANTIC_SEARCH", raising=False)
    from lore.search import semantic_enabled

    assert semantic_enabled() is False


def test_semantic_flag_recognises_true(monkeypatch):
    for raw in ("true", "TRUE", "True"):
        monkeypatch.setenv("LORE_SEMANTIC_SEARCH", raw)
        from lore.search import semantic_enabled

        assert semantic_enabled() is True, f"Failed for value: {raw}"


def test_semantic_flag_false_for_anything_else(monkeypatch):
    for raw in ("false", "0", "", "yes", "1", "no"):
        monkeypatch.setenv("LORE_SEMANTIC_SEARCH", raw)
        from lore.search import semantic_enabled

        assert semantic_enabled() is False, f"Should be off for: {raw!r}"


def test_embedder_raises_when_disabled(monkeypatch):
    """get_embedder() must raise EmbeddingUnavailableError when flag is off."""
    monkeypatch.setenv("LORE_SEMANTIC_SEARCH", "false")
    from lore.embeddings import EmbeddingUnavailableError, get_embedder, reset_for_tests

    reset_for_tests()
    with pytest.raises(EmbeddingUnavailableError):
        get_embedder()


def test_fts5_search_returns_empty_when_not_available():
    """fts5_search_sqlite short-circuits when client lacks FTS5."""
    from lore.search import fts5_search_sqlite

    class FakeClient:
        fts5_available = False
        vec_extension_loaded = False

    assert fts5_search_sqlite(FakeClient(), "q", None, 10) == []


def test_vector_search_returns_empty_when_not_available():
    """vector_search_sqlite short-circuits when vec extension is missing."""
    from lore.search import vector_search_sqlite

    class FakeClient:
        fts5_available = True
        vec_extension_loaded = False

    assert vector_search_sqlite(FakeClient(), [0.0] * 384, None, 10) == []


def test_compute_content_hash_stable():
    """Hash is stable across calls for the same input."""
    from lore.embeddings import compute_content_hash

    h1 = compute_content_hash("title", "content")
    h2 = compute_content_hash("title", "content")
    assert h1 == h2
    # Different inputs hash differently.
    assert compute_content_hash("a", "b") != compute_content_hash("b", "a")


def test_compute_content_hash_tolerates_none():
    from lore.embeddings import compute_content_hash

    # Should not crash on None.
    h = compute_content_hash("title", None)
    assert isinstance(h, str) and len(h) == 64
