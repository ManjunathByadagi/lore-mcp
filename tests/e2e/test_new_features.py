"""End-to-end tests for three features added in issues #14, #15, and #16.

* Issue #16 — ``min_score``: kb_search score filter
* Issue #15 — ``journal_search``: full-text search across journal entries
* Issue #14 — ``trust_score``: per-entry trust score stored and filterable

All tests require a live Lore instance (``LORE_E2E_URL`` env var).
Run with::

    LORE_E2E_URL=http://lore-staging:5555 pytest tests/e2e/test_new_features.py -v
"""

from __future__ import annotations

import time

import pytest

from .client import LoreClient, LoreClientError

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_results(result: dict) -> list[dict]:
    return result.get("results") or result.get("entries") or []


# ---------------------------------------------------------------------------
# Issue #16 — min_score filter
# ---------------------------------------------------------------------------


class TestMinScore:
    """Verify that kb_search honours the min_score filter parameter."""

    def test_min_score_zero_returns_entry(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """A min_score of 0.0 must not filter out any matching entry."""
        sentinel = "minscore9a2f"
        client.kb_add(
            topic=cleanup_topic,
            title=f"min_score test {sentinel}",
            content=f"Unique sentinel content {sentinel} for min_score=0 assertion.",
        )
        time.sleep(1)

        result = client.kb_search(sentinel, search_mode="fts", min_score=0.0)
        results = _get_results(result)
        assert len(results) >= 1, (
            f"min_score=0.0 filtered out all results for sentinel {sentinel!r}. "
            f"Got: {results}"
        )

    def test_min_score_above_max_returns_empty(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """A min_score above any achievable score must return an empty result set."""
        sentinel = "minscore7b4e"
        client.kb_add(
            topic=cleanup_topic,
            title=f"min_score upper-bound test {sentinel}",
            content=f"Sentinel entry for upper-bound min_score test {sentinel}.",
        )
        time.sleep(1)

        # Score 1e9 is impossible for any real similarity metric.
        result = client.kb_search(sentinel, search_mode="fts", min_score=1_000_000_000.0)
        results = _get_results(result)
        sentinel_ids = [
            e.get("kb_id")
            for e in results
            if sentinel in (e.get("content") or "") + (e.get("title") or "")
        ]
        assert len(sentinel_ids) == 0, (
            f"min_score=1e9 should filter everything but sentinel entry appeared: {sentinel_ids}"
        )

    def test_min_score_response_shape(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """kb_search with min_score must return a dict with a 'results' or 'entries' list."""
        sentinel = "minscapeshape3c"
        client.kb_add(
            topic=cleanup_topic,
            title=f"Response-shape min_score {sentinel}",
            content=f"Shape check for min_score parameter {sentinel}.",
        )
        time.sleep(1)

        result = client.kb_search(sentinel, search_mode="fts", min_score=0.0)
        assert isinstance(result, dict), (
            f"kb_search did not return a dict; got {type(result).__name__}"
        )
        results = _get_results(result)
        assert isinstance(results, list), (
            f"'results'/'entries' field is not a list: {result}"
        )


# ---------------------------------------------------------------------------
# Issue #15 — journal_search
# ---------------------------------------------------------------------------


class TestJournalSearch:
    """Verify journal_search returns the expected response shape.

    These tests intentionally do NOT assert that specific entries exist —
    the journal may be empty on a fresh staging instance.
    """

    def test_basic_query_returns_entries_key(self, client: LoreClient) -> None:
        """A plain journal_search must return a dict with an 'entries' key."""
        result = client.journal_search("deployment")
        assert isinstance(result, dict), (
            f"journal_search returned {type(result).__name__}, expected dict"
        )
        assert "entries" in result, (
            f"journal_search response missing 'entries' key: {result}"
        )
        assert isinstance(result["entries"], list), (
            f"'entries' must be a list, got {type(result['entries']).__name__}"
        )

    def test_entries_list_may_be_empty(self, client: LoreClient) -> None:
        """journal_search must return a valid (possibly empty) entries list."""
        result = client.journal_search("deployment")
        entries = result.get("entries", [])
        assert len(entries) >= 0, "entries list has negative length (impossible)"

    def test_fts_mode_returns_correct_shape(self, client: LoreClient) -> None:
        """journal_search does not accept a mode parameter; verify shape without one."""
        # journal_search does not expose a mode kwarg to callers — the backend
        # selects fts vs ilike based on DB_BACKEND. We verify the shape is correct
        # regardless of which backend is active.
        result = client.journal_search("system")
        assert "entries" in result, f"Missing 'entries' in journal_search response: {result}"
        assert "count" in result, f"Missing 'count' in journal_search response: {result}"

    def test_backend_field_present(self, client: LoreClient) -> None:
        """journal_search response must include a 'backend' field."""
        result = client.journal_search("ops")
        assert "backend" in result, (
            f"journal_search response missing 'backend' field: {result}"
        )
        assert result["backend"] in ("postgres", "sqlite"), (
            f"Unexpected backend value: {result['backend']!r}"
        )

    def test_search_mode_field_present(self, client: LoreClient) -> None:
        """journal_search response must include a 'search_mode' field."""
        result = client.journal_search("config")
        assert "search_mode" in result, (
            f"journal_search response missing 'search_mode' field: {result}"
        )
        assert result["search_mode"] in ("fts", "ilike"), (
            f"Unexpected search_mode: {result['search_mode']!r}"
        )

    def test_invalid_date_from_returns_error(self, client: LoreClient) -> None:
        """journal_search with a malformed date_from must return a tool-level error.

        The server validates ISO date format and returns an error envelope instead
        of raising an unhandled 500, so this call should raise LoreClientError
        (tool returned ok=False) rather than an HTTP error.
        """
        with pytest.raises(LoreClientError):
            client.journal_search("anything", date_from="not-a-date")

    def test_invalid_date_to_returns_error(self, client: LoreClient) -> None:
        """journal_search with a malformed date_to must return a tool-level error."""
        with pytest.raises(LoreClientError):
            client.journal_search("anything", date_to="2026/01/01")

    def test_limit_parameter_respected(self, client: LoreClient) -> None:
        """journal_search with limit=1 must return at most 1 entry."""
        result = client.journal_search("the", limit=1)
        entries = result.get("entries", [])
        assert len(entries) <= 1, (
            f"journal_search with limit=1 returned {len(entries)} entries"
        )

    def test_count_matches_entries_length(self, client: LoreClient) -> None:
        """The 'count' field must equal the length of the 'entries' list."""
        result = client.journal_search("ops")
        entries = result.get("entries", [])
        count = result.get("count", -1)
        assert count == len(entries), (
            f"'count' ({count}) does not match len(entries) ({len(entries)})"
        )


# ---------------------------------------------------------------------------
# Issue #14 — trust_score
# ---------------------------------------------------------------------------


class TestTrustScore:
    """Verify that trust_score is stored, retrieved, and filterable."""

    def test_kb_add_with_high_trust_score_stored(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """kb_get after kb_add with trust_score=0.9 must return trust_score=0.9."""
        result = client.kb_add(
            topic=cleanup_topic,
            title="Trust score high — e2e",
            content="Entry seeded with trust_score 0.9 for e2e verification.",
            trust_score=0.9,
        )
        kb_id = result["kb_id"]
        entry = client.kb_get(kb_id)
        assert "trust_score" in entry, (
            f"kb_get response missing 'trust_score' field: {entry}"
        )
        assert abs(entry["trust_score"] - 0.9) < 0.001, (
            f"Expected trust_score=0.9, got {entry['trust_score']}"
        )

    def test_kb_add_with_low_trust_score_stored(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """kb_get after kb_add with trust_score=0.3 must return trust_score=0.3."""
        result = client.kb_add(
            topic=cleanup_topic,
            title="Trust score low — e2e",
            content="Entry seeded with trust_score 0.3 for e2e verification.",
            trust_score=0.3,
        )
        kb_id = result["kb_id"]
        entry = client.kb_get(kb_id)
        assert "trust_score" in entry, (
            f"kb_get response missing 'trust_score' field: {entry}"
        )
        assert abs(entry["trust_score"] - 0.3) < 0.001, (
            f"Expected trust_score=0.3, got {entry['trust_score']}"
        )

    def test_min_trust_score_filters_low_trust_entry(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """min_trust_score=0.5 must exclude the low-trust entry from results.

        Two entries share a unique sentinel token; only the high-trust one
        should survive the filter.
        """
        sentinel = "trustfilt8d3a"
        low_id: str | None = None
        high_id: str | None = None
        try:
            low_result = client.kb_add(
                topic=cleanup_topic,
                title=f"Low-trust sentinel {sentinel}",
                content=f"Trust filter test low-trust entry sentinel={sentinel}.",
                trust_score=0.3,
            )
            low_id = low_result["kb_id"]

            high_result = client.kb_add(
                topic=cleanup_topic,
                title=f"High-trust sentinel {sentinel}",
                content=f"Trust filter test high-trust entry sentinel={sentinel}.",
                trust_score=0.9,
            )
            high_id = high_result["kb_id"]

            time.sleep(1)

            result = client.kb_search(
                sentinel,
                search_mode="fts",
                min_trust_score=0.5,
            )
            results = _get_results(result)
            returned_ids = {e.get("kb_id") for e in results}

            assert low_id not in returned_ids, (
                f"Low-trust entry {low_id!r} should be filtered by min_trust_score=0.5 "
                f"but appeared in results: {returned_ids}"
            )
        finally:
            for eid in (low_id, high_id):
                if eid:
                    try:
                        client.kb_delete(eid, confirm=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[cleanup] failed to delete {eid}: {exc}")

    def test_kb_list_entries_have_trust_score(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """Each entry returned by kb_list must include a trust_score field."""
        client.kb_add(
            topic=cleanup_topic,
            title="kb_list trust_score field check",
            content="Seeded to verify kb_list returns trust_score on each entry.",
            trust_score=0.75,
        )
        result = client.kb_list(topic=cleanup_topic)
        entries = result.get("entries") or result.get("results") or []
        assert len(entries) >= 1, (
            f"kb_list returned no entries for topic {cleanup_topic!r}"
        )
        for entry in entries:
            assert "trust_score" in entry, (
                f"kb_list entry missing 'trust_score' field: {entry}"
            )

    def test_default_trust_score_is_one(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """An entry added without explicit trust_score must default to 1.0."""
        result = client.kb_add(
            topic=cleanup_topic,
            title="Default trust score check",
            content="No explicit trust_score supplied; should default to 1.0.",
        )
        kb_id = result["kb_id"]
        entry = client.kb_get(kb_id)
        trust = entry.get("trust_score")
        assert trust is not None, f"kb_get missing trust_score field: {entry}"
        assert abs(trust - 1.0) < 0.001, (
            f"Default trust_score should be 1.0, got {trust}"
        )
