"""End-to-end tests for the Lore MCP server.

These tests target a deployed Lore instance over HTTP.  They are automatically
skipped when the ``LORE_E2E_URL`` environment variable is not set, so they are
safe to collect in CI without requiring a live server.

Set ``LORE_E2E_URL`` to the base URL of the target instance before running:

    LORE_E2E_URL=http://lore-staging:5555 pytest tests/e2e/ -v

See ``docs/development.md`` for the full testing guide.
"""
