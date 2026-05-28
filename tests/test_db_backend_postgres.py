"""Unit tests for the DB_BACKEND=postgres public-readiness fix (P0-1).

The README documents ``export DB_BACKEND=postgres``. Before this fix the
``DatabaseBackend`` enum had no ``postgres`` value, so ``get_db_client()``
silently fell through to the Supabase branch and crashed demanding
SUPABASE_URL. ``postgres`` and ``postgresql`` are now first-class aliases that
route to the same local PostgreSQL client as ``DB_BACKEND=local``.

These tests never open a real connection: ``LocalPostgresClient.__init__``
stores config without connecting (connections are made lazily), so asserting
the returned *type* and that no Supabase error is raised is sufficient and
hermetic.
"""

from __future__ import annotations

import pytest

from lore.db_client import (
    DatabaseBackend,
    LocalPostgresClient,
    SupabaseWrapper,
    get_db_client,
)

# ---------------------------------------------------------------------------
# Enum: postgres / postgresql are recognised values
# ---------------------------------------------------------------------------


def test_postgres_backend_enum_values_available():
    """postgres and postgresql are valid DatabaseBackend members."""
    assert DatabaseBackend.POSTGRES.value == "postgres"
    assert DatabaseBackend.POSTGRESQL.value == "postgresql"
    # Parsing the documented env-var string must succeed (no ValueError).
    assert DatabaseBackend("postgres") is DatabaseBackend.POSTGRES
    assert DatabaseBackend("postgresql") is DatabaseBackend.POSTGRESQL


# ---------------------------------------------------------------------------
# get_db_client(): DB_BACKEND=postgres selects the local Postgres client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_value", ["postgres", "postgresql", "local"])
def test_db_backend_postgres_selects_local_client(monkeypatch, backend_value):
    """postgres/postgresql/local all return a LocalPostgresClient.

    The crucial regression: ``postgres`` must NOT fall through to Supabase and
    must NOT raise the "SUPABASE_URL ... required" ValueError.
    """
    monkeypatch.setenv("DB_BACKEND", backend_value)
    # Ensure no Supabase creds are present so a wrong fall-through would crash.
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    client = get_db_client()
    assert isinstance(client, LocalPostgresClient)
    assert not isinstance(client, SupabaseWrapper)


def test_db_backend_postgres_does_not_raise_supabase_error(monkeypatch):
    """DB_BACKEND=postgres with no Supabase creds must not raise ValueError."""
    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

    # The pre-fix behaviour raised: "SUPABASE_URL and SUPABASE_KEY ... required".
    try:
        client = get_db_client()
    except ValueError as exc:  # pragma: no cover - this is the failure we guard
        pytest.fail(f"DB_BACKEND=postgres unexpectedly raised: {exc}")
    assert isinstance(client, LocalPostgresClient)


def test_db_backend_postgres_uses_generic_defaults(monkeypatch):
    """With DB_NAME/DB_USER unset, the generic lore/lore_user defaults apply.

    Confirms the homelab-specific mpm_system / latvian_user defaults are gone.
    """
    monkeypatch.setenv("DB_BACKEND", "postgres")
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)

    client = get_db_client()
    assert isinstance(client, LocalPostgresClient)
    # LocalPostgresClient stores connection config as plain attributes; the
    # connection itself is created lazily so no live DB is needed here.
    assert client.database == "lore"
    assert client.user == "lore_user"
    assert client.database != "mpm_system"
    assert client.user != "latvian_user"
