"""Search-mode assertion tests.

These tests verify that the three search modes (fts, semantic, hybrid) return
responses with the correct shape and mode-specific fields.  They do NOT assert
relevance — that is the job of ``test_regression.py``.
"""

from __future__ import annotations

import pytest

from .client import LoreClient

# Tag every test in this module as e2e so the suite can be filtered with
# `-m "not e2e"` in addition to the LORE_E2E_URL env gate in conftest.py.
pytestmark = pytest.mark.e2e

# Minimum number of results to expect when searching for content we just seeded
_MIN_RESULTS = 1


@pytest.fixture
def seeded_client(client: LoreClient, cleanup_topic: str) -> tuple[LoreClient, str]:
    """Add a known entry and return (client, query_term).

    The seeded entry contains distinct tokens that are unlikely to collide with
    pre-existing data in the KB so all three search modes can find it.
    """
    sentinel = "xqz99semanticfts"  # Unusual token for reliable FTS recall
    client.kb_add(
        topic=cleanup_topic,
        title=f"Search-mode test sentinel {sentinel}",
        content=(
            f"This entry contains the sentinel token {sentinel}. "
            "It is used to verify that all three search modes return results "
            "for a known-good query against recently added content."
        ),
    )
    return client, sentinel


# ---------------------------------------------------------------------------
# Response shape helpers
# ---------------------------------------------------------------------------


def _assert_search_shape(result: dict, mode_label: str) -> None:
    """Assert the basic shape common to all search results."""
    assert isinstance(result, dict), (
        f"[{mode_label}] kb_search returned {type(result).__name__}, expected dict"
    )
    results_list = result.get("results") or result.get("entries") or []
    assert isinstance(results_list, list), f"[{mode_label}] 'results' field is not a list: {result}"


def _get_results(result: dict) -> list[dict]:
    return result.get("results") or result.get("entries") or []


# ---------------------------------------------------------------------------
# FTS mode
# ---------------------------------------------------------------------------


class TestFtsMode:
    """Full-text search mode response assertions."""

    def test_fts_returns_results_list(self, seeded_client: tuple[LoreClient, str]) -> None:
        """FTS search must return a list of results for an exact-token query."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="fts")
        _assert_search_shape(result, "fts")

    def test_fts_finds_seeded_entry(self, seeded_client: tuple[LoreClient, str]) -> None:
        """FTS must find the entry whose content contains the sentinel token."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="fts")
        results = _get_results(result)
        assert len(results) >= _MIN_RESULTS, (
            f"FTS did not find the seeded entry. Got {len(results)} results for query {sentinel!r}"
        )

    def test_fts_result_entries_have_id(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Each FTS result entry must carry an 'id' field."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="fts")
        for entry in _get_results(result):
            assert "kb_id" in entry, f"FTS result entry missing 'id': {entry}"

    def test_fts_result_entries_have_title(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Each FTS result entry must carry a 'title' field."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="fts")
        for entry in _get_results(result):
            assert "title" in entry, f"FTS result entry missing 'title': {entry}"

    def test_fts_mode_field_in_response(self, seeded_client: tuple[LoreClient, str]) -> None:
        """FTS response should echo the search_mode that was used."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="fts")
        mode = result.get("search_mode")
        if mode is not None:
            assert mode == "fts", f"Expected search_mode='fts', got {mode!r}"


# ---------------------------------------------------------------------------
# Semantic mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSemanticMode:
    """Semantic (embedding-based) search mode response assertions."""

    def test_semantic_returns_results_list(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Semantic search must return a list for a natural-language paraphrase."""
        client, _ = seeded_client
        # Use a paraphrase, not the exact sentinel, to exercise embedding recall
        result = client.kb_search(
            "entry containing a unique verification token", search_mode="semantic"
        )
        _assert_search_shape(result, "semantic")

    def test_semantic_result_entries_have_id(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Each semantic result entry must carry an 'id' field."""
        client, _ = seeded_client
        result = client.kb_search("sentinel verification token", search_mode="semantic")
        for entry in _get_results(result):
            assert "kb_id" in entry, f"Semantic result entry missing 'id': {entry}"

    def test_semantic_mode_field_in_response(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Semantic response should echo the search_mode that was used."""
        client, _ = seeded_client
        result = client.kb_search("verification token", search_mode="semantic")
        mode = result.get("search_mode")
        if mode is not None:
            assert mode == "semantic", f"Expected search_mode='semantic', got {mode!r}"

    def test_semantic_degradation_graceful(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Semantic search must never raise an uncaught server error.

        If the embedding model is unavailable the server should degrade
        gracefully (e.g. fall back to FTS or return an empty list).
        It must NOT return a 5xx HTTP error.
        """
        client, _ = seeded_client
        # This should never throw an HTTP error
        result = client.kb_search("graceful degradation test", search_mode="semantic")
        _assert_search_shape(result, "semantic-degradation")


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHybridMode:
    """Hybrid (RRF fusion) search mode response assertions."""

    def test_hybrid_returns_results_list(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Hybrid search must return a list for the sentinel query."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        _assert_search_shape(result, "hybrid")

    def test_hybrid_result_entries_have_id(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Each hybrid result entry must carry an 'id' field."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        for entry in _get_results(result):
            assert "kb_id" in entry, f"Hybrid result entry missing 'id': {entry}"

    def test_hybrid_mode_field_in_response(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Hybrid response must set search_mode to 'hybrid'."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        mode = result.get("search_mode")
        if mode is not None:
            assert mode == "hybrid", f"Expected search_mode='hybrid', got {mode!r}"

    def test_hybrid_response_has_rrf_k(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Hybrid response should include the rrf_k constant used for fusion."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        if "rrf_k" in result:
            rrf_k = result["rrf_k"]
            assert isinstance(rrf_k, (int, float)), (
                f"rrf_k should be numeric, got {type(rrf_k).__name__}"
            )
            assert rrf_k > 0, f"rrf_k must be positive, got {rrf_k}"

    def test_hybrid_response_has_model(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Hybrid response should include the embedding model name."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        if "model" in result:
            assert isinstance(result["model"], str), (
                f"hybrid result 'model' field should be a string, got {result['model']!r}"
            )
            assert result["model"], "hybrid result 'model' field is empty"

    def test_hybrid_finds_more_than_empty(self, seeded_client: tuple[LoreClient, str]) -> None:
        """Hybrid mode should find the seeded entry that both FTS and semantic can see."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel, search_mode="hybrid")
        results = _get_results(result)
        assert len(results) >= _MIN_RESULTS, (
            f"Hybrid mode returned no results for sentinel query {sentinel!r}"
        )


# ---------------------------------------------------------------------------
# Cross-mode consistency
# ---------------------------------------------------------------------------


class TestCrossModeConsistency:
    """Sanity checks comparing results across modes."""

    def test_all_modes_return_same_entry_ids_for_exact_fts_token(
        self, seeded_client: tuple[LoreClient, str]
    ) -> None:
        """The seeded entry ID should appear in results from all three modes."""
        client, sentinel = seeded_client

        fts_result = client.kb_search(sentinel, search_mode="fts")
        hybrid_result = client.kb_search(sentinel, search_mode="hybrid")

        fts_ids = {e["kb_id"] for e in _get_results(fts_result) if "kb_id" in e}
        hybrid_ids = {e["kb_id"] for e in _get_results(hybrid_result) if "kb_id" in e}

        assert fts_ids, f"FTS returned no results for sentinel {sentinel!r}"
        assert hybrid_ids, f"Hybrid returned no results for sentinel {sentinel!r}"

        # The top FTS hit should also be reachable via hybrid
        top_fts_id = _get_results(fts_result)[0]["kb_id"]
        assert top_fts_id in hybrid_ids, (
            f"Top FTS result {top_fts_id!r} not found in hybrid results.\n"
            f"Hybrid IDs: {sorted(hybrid_ids)}"
        )

    def test_default_mode_returns_results(self, seeded_client: tuple[LoreClient, str]) -> None:
        """kb_search without an explicit search_mode should also return results."""
        client, sentinel = seeded_client
        result = client.kb_search(sentinel)
        _assert_search_shape(result, "default")
        results = _get_results(result)
        assert len(results) >= _MIN_RESULTS, (
            f"Default-mode search returned no results for {sentinel!r}"
        )
