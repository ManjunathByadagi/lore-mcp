"""Unit tests for LORE_ENV response labeling and production guardrails (issue #11).

These tests are intentionally lightweight: the production guard fires BEFORE any
database or embedder work, so no semantic deps / live DB are required. They
verify:

  * every response envelope carries an ``env`` field sourced from LORE_ENV
  * ``env`` defaults to "production" when LORE_ENV is unset (fail-safe)
  * kb_backfill_embeddings blocks in production without confirm_production
  * dry runs and non-prod environments are never blocked by the guard
"""

from __future__ import annotations

import pytest

from lore.response import ErrorCodes, ResponseEnvelope, current_env

# ---------------------------------------------------------------------------
# Response envelope env labeling
# ---------------------------------------------------------------------------


def test_success_envelope_includes_env(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "staging")
    resp = ResponseEnvelope.success("ok", {"a": 1})
    assert resp["env"] == "staging"
    assert set(resp) == {"ok", "error", "message", "env", "data"}


def test_error_envelope_includes_env(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "development")
    resp = ResponseEnvelope.error(ErrorCodes.INVALID_INPUT, "nope")
    assert resp["env"] == "development"
    assert resp["ok"] is False


def test_env_defaults_to_production_when_unset(monkeypatch):
    """Fail-safe: an unconfigured deployment is treated as production."""
    monkeypatch.delenv("LORE_ENV", raising=False)
    assert current_env() == "production"
    assert ResponseEnvelope.success("x")["env"] == "production"


# ---------------------------------------------------------------------------
# _production_guard helper
# ---------------------------------------------------------------------------


@pytest.fixture
def srv():
    import lore.server as s

    return s


def test_guard_blocks_in_production_without_confirm(srv, monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)  # unset -> production
    guard = srv._production_guard("kb_backfill_embeddings", confirm_production=False)
    assert guard is not None
    assert guard["error"] == ErrorCodes.PRODUCTION_GUARD
    assert "confirm_production=true" in guard["message"]


def test_guard_allows_in_production_with_confirm(srv, monkeypatch):
    monkeypatch.setenv("LORE_ENV", "production")
    assert srv._production_guard("kb_backfill_embeddings", confirm_production=True) is None


def test_guard_allows_dry_run_in_production(srv, monkeypatch):
    monkeypatch.setenv("LORE_ENV", "production")
    assert (
        srv._production_guard("kb_backfill_embeddings", confirm_production=False, dry_run=True)
        is None
    )


@pytest.mark.parametrize("env", ["staging", "development", "STAGING"])
def test_guard_allows_non_production(srv, monkeypatch, env):
    monkeypatch.setenv("LORE_ENV", env)
    assert srv._production_guard("kb_backfill_embeddings", confirm_production=False) is None


# ---------------------------------------------------------------------------
# kb_backfill_embeddings handler behaviour (guard fires before DB/embedder)
# ---------------------------------------------------------------------------


def test_backfill_blocks_in_production_without_confirm(srv, monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)  # production by default
    resp = srv.handle_kb_backfill_embeddings(confirm_production=False, dry_run=False)
    assert resp["ok"] is False
    assert resp["error"] == ErrorCodes.PRODUCTION_GUARD
    assert resp["env"] == "production"


def test_backfill_dry_run_skips_guard_in_production(srv, monkeypatch):
    """A dry run must pass the guard (it may still fail later for other reasons,
    but it must NOT be a production_guard rejection)."""
    monkeypatch.setenv("LORE_ENV", "production")
    resp = srv.handle_kb_backfill_embeddings(dry_run=True)
    assert resp.get("error") != ErrorCodes.PRODUCTION_GUARD


def test_backfill_staging_skips_guard(srv, monkeypatch):
    monkeypatch.setenv("LORE_ENV", "staging")
    resp = srv.handle_kb_backfill_embeddings(confirm_production=False)
    assert resp.get("error") != ErrorCodes.PRODUCTION_GUARD
    assert resp["env"] == "staging"


def test_backfill_with_confirm_skips_guard_in_production(srv, monkeypatch):
    monkeypatch.setenv("LORE_ENV", "production")
    resp = srv.handle_kb_backfill_embeddings(confirm_production=True)
    assert resp.get("error") != ErrorCodes.PRODUCTION_GUARD
