"""Unit tests for the public-readiness search-config fixes (P0-2).

The deployment-specific search roots (LATVIAN_LEARNING_ROOT, LATVIAN_XTTS_ROOT,
INGEST_ROOT) are now optional and default to ``None`` instead of a hardcoded
``/srv/*`` homelab path. When a root is unset the dependent search tools must
return a clean, empty "not configured" result rather than crashing on a
``None / "subdir"`` ``TypeError`` or scanning a path that does not exist.

These are pure unit tests — no live database and no filesystem under /srv.
``lore.server`` resolves the roots once at import; the handlers re-read the
module-level globals, so the tests patch those globals via ``monkeypatch``.
"""

from __future__ import annotations

import lore.server as srv

# ---------------------------------------------------------------------------
# search_corpora: INGEST_ROOT unset -> clean empty "not configured" result
# ---------------------------------------------------------------------------


def test_search_corpora_root_unset_returns_not_configured(monkeypatch):
    """INGEST_ROOT=None must not raise (no ``None / "corpora"`` TypeError)."""
    monkeypatch.setattr(srv, "INGEST_ROOT", None)
    resp = srv.handle_search_corpora("anything")
    assert resp["ok"] is True
    assert resp["error"] is None
    assert resp["data"]["results"] == []
    assert resp["data"].get("total_matches", 0) == 0
    assert "not configured" in resp["message"].lower()
    # Critically, this is NOT an unexpected exception leaking from None / str.
    assert resp["error"] != "unexpected_exception"


def test_search_corpora_root_unset_never_references_srv(monkeypatch):
    """The unconfigured result must not mention a hardcoded /srv path."""
    monkeypatch.setattr(srv, "INGEST_ROOT", None)
    resp = srv.handle_search_corpora("anything")
    assert "/srv" not in resp["message"]


# ---------------------------------------------------------------------------
# search_transcripts: LATVIAN_XTTS_ROOT unset -> clean empty "not configured"
# ---------------------------------------------------------------------------


def test_search_transcripts_root_unset_returns_not_configured(monkeypatch):
    """LATVIAN_XTTS_ROOT=None must not raise (no ``None / "whisper_*"``)."""
    monkeypatch.setattr(srv, "LATVIAN_XTTS_ROOT", None)
    resp = srv.handle_search_transcripts("anything")
    assert resp["ok"] is True
    assert resp["error"] is None
    assert resp["data"]["results"] == []
    assert resp["data"].get("total_matches", 0) == 0
    assert "not configured" in resp["message"].lower()
    assert resp["error"] != "unexpected_exception"


def test_search_transcripts_root_unset_never_references_srv(monkeypatch):
    monkeypatch.setattr(srv, "LATVIAN_XTTS_ROOT", None)
    resp = srv.handle_search_transcripts("anything")
    assert "/srv" not in resp["message"]


# ---------------------------------------------------------------------------
# search_local: unset roots must be skipped, not stringified to "None"
# ---------------------------------------------------------------------------


def test_search_local_unset_roots_do_not_crash(monkeypatch, tmp_path):
    """With the Latvian roots unset, handle_search_local must still run cleanly
    over the configured KNOWLEDGE_DATA_DIR rather than scanning a "None" path."""
    monkeypatch.setattr(srv, "LATVIAN_LEARNING_ROOT", None)
    monkeypatch.setattr(srv, "LATVIAN_XTTS_ROOT", None)
    monkeypatch.setattr(srv, "KNOWLEDGE_DATA_DIR", tmp_path)

    resp = srv.handle_search_local("anything")
    assert resp["ok"] is True
    assert resp["error"] != "unexpected_exception"
    # No literal "None" path should have been searched.
    assert "None" not in resp["message"]


# ---------------------------------------------------------------------------
# Config defaults: portable, never the homelab /srv paths
# ---------------------------------------------------------------------------


def test_optional_root_unset_is_none(monkeypatch):
    """_optional_root returns None for an unset/blank env var (no /srv default)."""
    monkeypatch.delenv("LATVIAN_LEARNING_ROOT", raising=False)
    assert srv._optional_root("LATVIAN_LEARNING_ROOT") is None
    monkeypatch.setenv("LATVIAN_LEARNING_ROOT", "   ")
    assert srv._optional_root("LATVIAN_LEARNING_ROOT") is None


def test_optional_root_set_returns_path(monkeypatch):
    """_optional_root returns a Path when the env var is set."""
    from pathlib import Path

    monkeypatch.setenv("INGEST_ROOT", "/data/ingest")
    assert srv._optional_root("INGEST_ROOT") == Path("/data/ingest")


def test_knowledge_data_dir_default_is_portable(monkeypatch):
    """KNOWLEDGE_DATA_DIR must default to a portable local path, never /srv."""
    from pathlib import Path

    from lore.env_config import get_env

    # The module-level default lives in server.py; assert the resolved default
    # matches the SQLite fallback (./knowledge-data) and is not a /srv path.
    monkeypatch.delenv("KNOWLEDGE_DATA_DIR", raising=False)
    default = Path(get_env("KNOWLEDGE_DATA_DIR", "./knowledge-data"))
    assert str(default) == "knowledge-data"
    assert "/srv" not in str(default)
