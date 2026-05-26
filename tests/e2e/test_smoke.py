"""Smoke tests: verify the Lore service is reachable and sane.

These are the first tests to run.  They make no assumptions about stored data
and do not modify the knowledge base.  A failure here usually means the server
is misconfigured or unreachable rather than a feature regression.
"""

from __future__ import annotations

import pytest

from .client import LoreClient

# Tag every test in this module as e2e so the suite can be filtered with
# `-m "not e2e"` in addition to the LORE_E2E_URL env gate in conftest.py.
pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Expected tools (subset — the server may advertise extras in future)
# ---------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "kb_add",
    "kb_search",
    "kb_get",
    "kb_update",
    "kb_delete",
    "kb_list",
    "kb_backfill_embeddings",
    "kb_embedding_status",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServiceReachability:
    """Verify the service is up and responds to protocol introspection."""

    def test_ping(self, client: LoreClient) -> None:
        """Service must answer tools/list without error."""
        assert client.ping(), "Lore service did not respond to tools/list"

    def test_tools_list_returns_list(self, client: LoreClient) -> None:
        """tools/list must return a non-empty list."""
        tools = client.list_tools()
        assert isinstance(tools, list), "tools/list result is not a list"
        assert len(tools) > 0, "tools/list returned an empty tool list"

    def test_expected_tools_present(self, client: LoreClient) -> None:
        """All required Lore tools must be advertised by the server."""
        advertised = {t["name"] for t in client.list_tools()}
        missing = EXPECTED_TOOLS - advertised
        assert not missing, (
            f"The following expected tools are missing from tools/list: {sorted(missing)}"
        )

    def test_tool_descriptors_have_name_and_description(self, client: LoreClient) -> None:
        """Each tool descriptor must contain at least 'name' and 'description'."""
        for tool in client.list_tools():
            assert "name" in tool, f"Tool missing 'name' key: {tool}"
            assert "description" in tool, f"Tool {tool.get('name')!r} missing 'description'"


class TestEmbeddingStatus:
    """Verify the kb_embedding_status tool returns a sensible shape."""

    def test_embedding_status_responds(self, client: LoreClient) -> None:
        """kb_embedding_status must return a dict without error."""
        status = client.kb_embedding_status()
        assert isinstance(status, dict), f"Expected dict, got {type(status).__name__}"

    def test_embedding_status_has_coverage_fields(self, client: LoreClient) -> None:
        """Status dict must contain total and embedded counts."""
        status = client.kb_embedding_status()
        # Accept either naming convention observed in the codebase
        has_total = "total" in status or "total_entries" in status
        has_embedded = "embedded" in status or "embedded_entries" in status
        assert has_total, f"embedding_status missing total count key. Got: {list(status)}"
        assert has_embedded, f"embedding_status missing embedded count key. Got: {list(status)}"

    def test_embedding_status_numeric_counts(self, client: LoreClient) -> None:
        """Total and embedded counts must be non-negative integers."""
        status = client.kb_embedding_status()
        total = status.get("total") or status.get("total_entries", 0)
        embedded = status.get("embedded") or status.get("embedded_entries", 0)
        assert isinstance(total, int), f"total is not int: {total!r}"
        assert isinstance(embedded, int), f"embedded is not int: {embedded!r}"
        assert total >= 0, f"total is negative: {total}"
        assert embedded >= 0, f"embedded is negative: {embedded}"
        assert embedded <= total, f"embedded ({embedded}) > total ({total}) — impossible coverage"

    @pytest.mark.slow
    def test_embedding_status_coverage_pct_range(self, client: LoreClient) -> None:
        """Coverage percentage must be in [0, 100]."""
        status = client.kb_embedding_status()
        pct = status.get("coverage_pct") or status.get("coverage_percent")
        if pct is None:
            pytest.skip("Server does not return a coverage_pct field")
        assert 0.0 <= float(pct) <= 100.0, f"coverage_pct out of range: {pct}"
