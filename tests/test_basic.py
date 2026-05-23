"""Basic tests for the Lore MCP server.

Verifies importability of the package, presence of the module-level
tool handlers in lore.server, and the public CLI entry point.
"""

import os

import pytest


def test_package_import():
    """The lore package imports cleanly."""
    import lore  # noqa: F401


def test_server_module_import():
    """The server module imports cleanly."""
    from lore import server

    # The MCP Server instance is exposed as 'app' (used by the HTTP wrapper)
    assert hasattr(server, "app"), "lore.server.app (MCP Server instance) missing"

    # main() is the entry point referenced by pyproject.toml
    assert callable(getattr(server, "main", None)), "lore.server.main missing"


def test_response_envelope_import():
    """ResponseEnvelope helper is importable (used by every handler)."""
    from lore.response import ErrorCodes, ResponseEnvelope

    assert hasattr(ResponseEnvelope, "ok")
    assert hasattr(ResponseEnvelope, "error")
    assert hasattr(ErrorCodes, "UNEXPECTED_EXCEPTION")


@pytest.mark.parametrize(
    "handler_name",
    [
        "handle_kb_add",
        "handle_kb_search",
        "handle_kb_get",
        "handle_kb_list",
        "handle_kb_update",
        "handle_kb_delete",
        "handle_kb_ingest_doc",
        "handle_kb_ingest_dir",
        "handle_kb_sync_status",
    ],
)
def test_kb_handlers_exist(handler_name):
    """Each knowledge-base handler is exported as a module-level callable."""
    from lore import server

    handler = getattr(server, handler_name, None)
    assert callable(handler), f"{handler_name} not found on lore.server"


@pytest.mark.parametrize(
    "handler_name",
    [
        "handle_investigation_add",
        "handle_investigation_list",
        "handle_investigation_get",
        "handle_investigation_log_experiment",
        "handle_investigation_list_experiments",
    ],
)
def test_investigation_handlers_exist(handler_name):
    """Investigation workflow handlers are present (renamed from research_* in v0.4.0)."""
    from lore import server

    handler = getattr(server, handler_name, None)
    assert callable(handler), f"{handler_name} not found on lore.server"


@pytest.mark.parametrize(
    "handler_name",
    [
        "handle_journal_append",
        "handle_journal_get",
        "handle_journal_list",
    ],
)
def test_journal_handlers_exist(handler_name):
    """Journal handlers are present."""
    from lore import server

    handler = getattr(server, handler_name, None)
    assert callable(handler), f"{handler_name} not found on lore.server"


@pytest.mark.parametrize(
    "handler_name",
    [
        "handle_mcp_index_scan",
        "handle_mcp_index_search",
        "handle_mcp_index_get_server",
        "handle_mcp_index_get_tool",
        "handle_mcp_index_rebuild",
    ],
)
def test_mcp_index_handlers_exist(handler_name):
    """MCP-index handlers are present."""
    from lore import server

    handler = getattr(server, handler_name, None)
    assert callable(handler), f"{handler_name} not found on lore.server"


def test_removed_legacy_handlers_are_gone():
    """Knowledge-graph and source-tracking handlers were removed in v0.4.0."""
    from lore import server

    for legacy in (
        "handle_kg_add",
        "handle_kg_search",
        "handle_research_add_source",
        "handle_research_list_sources",
        "handle_kb_link_to_source",
    ):
        assert not hasattr(server, legacy), f"Legacy handler {legacy} still present"


def test_sqlite_backend_enum_available():
    """SQLite is a first-class backend option."""
    from lore.db_client import DatabaseBackend

    assert DatabaseBackend.SQLITE.value == "sqlite"
    assert DatabaseBackend.LOCAL.value == "local"
