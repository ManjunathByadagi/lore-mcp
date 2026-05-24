"""Knowledge-base CRUD lifecycle tests.

These tests exercise the full create / read / update / delete cycle for
knowledge-base entries.  Each test is independent: it uses the ``cleanup_topic``
fixture to obtain a unique topic namespace and tears down its own data.
"""

from __future__ import annotations

import pytest

from .client import LoreClient, LoreClientError


class TestKbAdd:
    """Verify kb_add creates entries with the expected shape."""

    def test_add_returns_id(self, client: LoreClient, cleanup_topic: str) -> None:
        """A successful kb_add must return a non-empty id."""
        result = client.kb_add(
            topic=cleanup_topic,
            title="Lifecycle test: add returns id",
            content="This entry was created by test_add_returns_id.",
        )
        assert "id" in result, f"kb_add response missing 'id': {result}"
        assert result["id"], "kb_add returned an empty id"

    def test_add_status_field(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_add should indicate success in its response."""
        result = client.kb_add(
            topic=cleanup_topic,
            title="Lifecycle test: add status",
            content="Content for status field check.",
        )
        # Accept "added", "created", "ok", or a numeric 2xx-style code
        status = result.get("status") or result.get("result") or ""
        assert status, f"kb_add response has no status indicator: {result}"

    def test_add_multiple_entries_distinct_ids(
        self, client: LoreClient, cleanup_topic: str
    ) -> None:
        """Multiple kb_add calls under the same topic must yield distinct IDs."""
        ids = []
        for i in range(3):
            res = client.kb_add(
                topic=cleanup_topic,
                title=f"Distinct-id entry {i}",
                content=f"Entry body number {i}.",
            )
            ids.append(res["id"])
        assert len(set(ids)) == 3, f"Duplicate IDs returned by kb_add: {ids}"


class TestKbGet:
    """Verify kb_get retrieves entries by ID."""

    def test_get_returns_correct_entry(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_get must return the entry that was just added."""
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Lifecycle test: get by id",
            content="Content for get test.",
        )
        entry_id = add_result["id"]
        get_result = client.kb_get(entry_id)
        assert get_result.get("id") == entry_id, (
            f"kb_get returned wrong entry. Expected id={entry_id}, got {get_result.get('id')}"
        )

    def test_get_preserves_content(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_get must return the original content unchanged."""
        content = "Unique content sentinel: 7f3a9b2c."
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Content preservation check",
            content=content,
        )
        get_result = client.kb_get(add_result["id"])
        returned_content = get_result.get("content") or ""
        assert content in returned_content, (
            f"kb_get returned content does not contain original text.\n"
            f"Expected to find: {content!r}\n"
            f"Got: {returned_content!r}"
        )

    def test_get_nonexistent_id_returns_error(self, client: LoreClient) -> None:
        """kb_get with a non-existent ID must raise LoreClientError or return an error dict."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        try:
            result = client.kb_get(fake_id)
            # If no exception, the response should carry an error indicator
            assert result.get("error") or result.get("status") == "not_found", (
                f"kb_get on non-existent ID should signal an error, got: {result}"
            )
        except LoreClientError:
            pass  # Protocol-level error is also acceptable


class TestKbUpdate:
    """Verify kb_update modifies entry fields."""

    def test_update_title(self, client: LoreClient, cleanup_topic: str) -> None:
        """After kb_update, kb_get must reflect the new title."""
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Original title",
            content="Body text for update test.",
        )
        entry_id = add_result["id"]
        new_title = "Updated title — lifecycle test"

        client.kb_update(entry_id, title=new_title)

        get_result = client.kb_get(entry_id)
        returned_title = get_result.get("title") or ""
        assert new_title in returned_title, (
            f"kb_get after update did not reflect new title.\n"
            f"Expected: {new_title!r}\nGot: {returned_title!r}"
        )

    def test_update_content(self, client: LoreClient, cleanup_topic: str) -> None:
        """After kb_update with new content, kb_get must return the new content."""
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Content update test",
            content="Old content.",
        )
        entry_id = add_result["id"]
        new_content = "Replacement content sentinel: a1b2c3d4."

        client.kb_update(entry_id, content=new_content)

        get_result = client.kb_get(entry_id)
        returned = get_result.get("content") or ""
        assert new_content in returned, (
            f"kb_get after content update is stale.\nExpected: {new_content!r}\nGot: {returned!r}"
        )


class TestKbDelete:
    """Verify kb_delete removes entries."""

    def test_delete_returns_success(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_delete must complete without raising an exception."""
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Entry to be deleted",
            content="This entry will be deleted by the test.",
        )
        entry_id = add_result["id"]
        delete_result = client.kb_delete(entry_id, confirm=True)
        assert delete_result is not None, "kb_delete returned None"

    def test_deleted_entry_not_retrievable(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_get after kb_delete must return an error or not-found response."""
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="Entry to delete then fetch",
            content="Should be gone after deletion.",
        )
        entry_id = add_result["id"]
        client.kb_delete(entry_id, confirm=True)

        try:
            result = client.kb_get(entry_id)
            assert result.get("error") or result.get("status") in ("not_found", "deleted"), (
                f"kb_get on deleted entry should signal not-found, got: {result}"
            )
        except LoreClientError:
            pass  # Protocol-level error is also acceptable


class TestKbList:
    """Verify kb_list returns entries for a given topic."""

    def test_list_finds_added_entry(self, client: LoreClient, cleanup_topic: str) -> None:
        """An entry added to a topic must appear in kb_list for that topic."""
        client.kb_add(
            topic=cleanup_topic,
            title="List visibility test",
            content="Should appear in kb_list.",
        )
        result = client.kb_list(topic=cleanup_topic)
        entries = result.get("entries") or result.get("results") or []
        assert len(entries) >= 1, (
            f"kb_list for topic {cleanup_topic!r} returned no entries after kb_add"
        )

    def test_list_empty_topic_returns_empty(self, client: LoreClient) -> None:
        """kb_list on a topic that has never had entries must not error."""
        import uuid

        phantom_topic = f"phantom-{uuid.uuid4().hex}"
        result = client.kb_list(topic=phantom_topic)
        entries = result.get("entries") or result.get("results") or []
        assert isinstance(entries, list), f"kb_list result is not a list: {result}"
        assert len(entries) == 0, (
            f"Expected empty list for phantom topic, got {len(entries)} entries"
        )

    def test_list_respects_limit(self, client: LoreClient, cleanup_topic: str) -> None:
        """kb_list with limit=1 must return at most 1 entry."""
        for i in range(3):
            client.kb_add(
                topic=cleanup_topic,
                title=f"Limit test entry {i}",
                content=f"Body {i}",
            )
        result = client.kb_list(topic=cleanup_topic, limit=1)
        entries = result.get("entries") or result.get("results") or []
        assert len(entries) <= 1, f"kb_list with limit=1 returned {len(entries)} entries"


@pytest.mark.slow
class TestRoundTripIntegrity:
    """Slower round-trip tests that verify data integrity end-to-end."""

    def test_full_crud_cycle(self, client: LoreClient, cleanup_topic: str) -> None:
        """Add → get → update → get again → delete → confirm gone."""
        # 1. Add
        add_result = client.kb_add(
            topic=cleanup_topic,
            title="CRUD cycle — initial",
            content="Step 1: creation.",
        )
        entry_id = add_result["id"]

        # 2. Verify via get
        get1 = client.kb_get(entry_id)
        assert get1.get("id") == entry_id

        # 3. Update
        updated_content = "Step 3: updated content."
        client.kb_update(entry_id, content=updated_content)

        # 4. Verify update persisted
        get2 = client.kb_get(entry_id)
        assert updated_content in (get2.get("content") or ""), (
            f"Update did not persist. Got: {get2.get('content')!r}"
        )

        # 5. Delete
        client.kb_delete(entry_id, confirm=True)

        # 6. Confirm gone
        try:
            gone = client.kb_get(entry_id)
            assert gone.get("error") or gone.get("status") in ("not_found", "deleted")
        except LoreClientError:
            pass
