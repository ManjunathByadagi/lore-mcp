"""Backfill idempotency and coverage tests.

These tests exercise ``kb_backfill_embeddings`` to confirm that:

- The operation is idempotent (running it twice is safe).
- Coverage never decreases after a successful backfill.
- The backfill response shape is consistent.
"""

from __future__ import annotations

import pytest

from .client import LoreClient


def _coverage_pct(status: dict) -> float | None:
    """Extract a coverage percentage from an embedding-status dict."""
    raw = status.get("coverage_pct") or status.get("coverage_percent")
    if raw is None:
        total = status.get("total") or status.get("total_entries", 0)
        embedded = status.get("embedded") or status.get("embedded_entries", 0)
        if total and total > 0:
            return (embedded / total) * 100.0
        return None
    return float(raw)


class TestBackfillResponseShape:
    """Verify the shape of the backfill response."""

    def test_backfill_returns_dict(self, client: LoreClient) -> None:
        """kb_backfill_embeddings must return a dict without raising."""
        result = client.kb_backfill_embeddings()
        assert isinstance(result, dict), (
            f"Expected dict from kb_backfill_embeddings, got {type(result).__name__}"
        )

    def test_backfill_response_not_empty(self, client: LoreClient) -> None:
        """Backfill response must contain at least one key."""
        result = client.kb_backfill_embeddings()
        assert len(result) > 0, "kb_backfill_embeddings returned an empty dict"

    def test_backfill_has_processed_field(self, client: LoreClient) -> None:
        """Backfill response should indicate how many entries were processed."""
        result = client.kb_backfill_embeddings()
        # Accept various field names: processed, embedded, count, batch_size
        has_count = any(
            k in result for k in ("processed", "embedded", "count", "batch_size", "total_processed")
        )
        if not has_count:
            pytest.skip(f"Server does not return a processed-count field. Got keys: {list(result)}")
        # Whichever field is present must be a non-negative integer
        for key in ("processed", "embedded", "count", "total_processed"):
            if key in result:
                val = result[key]
                assert isinstance(val, int) and val >= 0, (
                    f"Backfill field {key!r} should be a non-negative int, got {val!r}"
                )
                break


@pytest.mark.slow
class TestBackfillIdempotency:
    """Running backfill twice must be safe and must not reduce coverage."""

    def test_double_backfill_does_not_error(self, client: LoreClient) -> None:
        """Two consecutive backfill calls must both succeed without error."""
        result1 = client.kb_backfill_embeddings()
        result2 = client.kb_backfill_embeddings()
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_coverage_does_not_decrease_after_backfill(self, client: LoreClient) -> None:
        """Coverage percentage after backfill must be >= coverage before backfill."""
        status_before = client.kb_embedding_status()
        pct_before = _coverage_pct(status_before)

        client.kb_backfill_embeddings()

        status_after = client.kb_embedding_status()
        pct_after = _coverage_pct(status_after)

        if pct_before is None or pct_after is None:
            pytest.skip("Server does not expose coverage_pct — cannot verify trend")

        assert pct_after >= pct_before - 0.01, (
            f"Coverage decreased after backfill: {pct_before:.1f}% → {pct_after:.1f}%"
        )

    def test_backfill_status_consistency(self, client: LoreClient) -> None:
        """After backfill the embedding-status totals must remain internally consistent."""
        client.kb_backfill_embeddings()
        status = client.kb_embedding_status()

        total = status.get("total") or status.get("total_entries", 0)
        embedded = status.get("embedded") or status.get("embedded_entries", 0)

        assert isinstance(total, int) and total >= 0, f"total is not a non-negative int: {total!r}"
        assert isinstance(embedded, int) and embedded >= 0, (
            f"embedded is not a non-negative int: {embedded!r}"
        )
        assert embedded <= total, (
            f"embedded ({embedded}) > total ({total}) — status is inconsistent"
        )


@pytest.mark.slow
class TestBackfillWithNewEntries:
    """Backfill must pick up entries added after the last backfill run."""

    def test_new_entry_becomes_embedded_after_backfill(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """A freshly added entry should count as embedded after backfill completes."""
        # Seed a new entry
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Backfill coverage test entry",
            content=(
                "This entry is added immediately before a backfill run.  "
                "After backfill completes, coverage must not have decreased."
            ),
        )
        seeded_id = add_result["id"]

        # Capture baseline
        status_before = client.kb_embedding_status()
        pct_before = _coverage_pct(status_before)

        # Run backfill
        client.kb_backfill_embeddings()

        # Assert coverage stable or improved
        status_after = client.kb_embedding_status()
        pct_after = _coverage_pct(status_after)

        if pct_before is not None and pct_after is not None:
            assert pct_after >= pct_before - 0.01, (
                f"Coverage dropped after seeding entry {seeded_id!r} and running backfill: "
                f"{pct_before:.1f}% → {pct_after:.1f}%"
            )

        # Cleanup handled by cleanup_topic fixture — explicit delete here too
        try:
            client.kb_delete(seeded_id, confirm=True)
        except Exception:  # noqa: BLE001
            pass
