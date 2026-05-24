"""Regression tests driven by ``regression_corpus.yaml``.

For each corpus entry the test:

1. Seeds a KB article under an isolated topic.
2. Runs each configured query against the live server.
3. Asserts rank and match expectations.

False-negative queries (``expect_match: false``) verify that the seeded entry
does **not** appear in the top 10 results.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from .client import LoreClient

# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

CORPUS_PATH = Path(__file__).parent / "regression_corpus.yaml"
_TOP_N_FALSE_NEG = 10  # How many results to inspect for false-negative checks


def _load_corpus() -> list[dict[str, Any]]:
    with CORPUS_PATH.open() as fh:
        return yaml.safe_load(fh)  # type: ignore[return-value]


CORPUS: list[dict[str, Any]] = _load_corpus()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_results(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("results") or result.get("entries") or []


def _find_rank(
    results: list[dict[str, Any]],
    seeded_id: str,
) -> int | None:
    """Return 1-based rank of ``seeded_id`` in ``results``, or ``None``."""
    for rank, entry in enumerate(results, start=1):
        if entry.get("kb_id") == seeded_id:
            return rank
    return None


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRegressionCorpus:
    """Parameterised regression tests built from regression_corpus.yaml.

    Each parametrize ID is ``<corpus-entry-id>/<query-index>`` so failures
    are immediately traceable back to the YAML.
    """

    @staticmethod
    def _collect_params() -> list[pytest.param]:
        """Build pytest.param list for parametrize."""
        params: list[pytest.param] = []
        for entry in CORPUS:
            corpus_id = entry["id"]
            for q_idx, query in enumerate(entry.get("queries", [])):
                pid = f"{corpus_id}/q{q_idx}"
                params.append(pytest.param(entry, query, id=pid))
        return params

    @pytest.fixture(autouse=True)
    def _seed_and_cleanup(self, client: LoreClient, entry: dict[str, Any]) -> Any:
        """Seed the corpus entry and track the new ID for assertions."""
        unique_suffix = uuid.uuid4().hex[:8]
        topic = f"e2e-regression-{entry.get('topic', 'misc')}-{unique_suffix}"

        add_result = client.kb_add(
            topic=topic,
            title=entry.get("title", "regression entry"),
            content=entry.get("content", ""),
        )
        seeded_id: str = add_result["kb_id"]

        # Brief pause to allow embeddings to be computed (may be async on server)
        time.sleep(0.5)

        self._seeded_id = seeded_id
        self._topic = topic

        yield

        # Cleanup — best effort
        try:
            client.kb_delete(seeded_id, confirm=True)
        except Exception:  # noqa: BLE001
            pass

    @pytest.mark.parametrize("entry,query", _collect_params())
    def test_corpus_query(
        self,
        client: LoreClient,
        entry: dict[str, Any],
        query: dict[str, Any],
    ) -> None:
        """Run a single corpus query and assert rank / match expectation."""
        q_text: str = query["text"]
        mode: str = query.get("mode", "hybrid")
        expect_match: bool = query.get("expect_match", True)
        max_rank: int | None = query.get("max_rank")

        result = client.kb_search(q_text, search_mode=mode, top_k=max(20, _TOP_N_FALSE_NEG))
        results = _get_results(result)
        rank = _find_rank(results, self._seeded_id)

        if expect_match:
            assert rank is not None, (
                f"[{entry['id']}] Expected seeded entry {self._seeded_id!r} in {mode!r} "
                f"results for query {q_text!r}, but it was not found.\n"
                f"Top result IDs: {[e.get('kb_id') for e in results[:5]]}"
            )
            if max_rank is not None:
                assert rank <= max_rank, (
                    f"[{entry['id']}] Expected seeded entry within top {max_rank} for "
                    f"{mode!r} query {q_text!r}, but it appeared at rank {rank}.\n"
                    f"Top result IDs: {[e.get('kb_id') for e in results[: max_rank + 2]]}"
                )
        else:
            # False-negative check: entry must NOT appear in top N
            top_ids = {e.get("kb_id") for e in results[:_TOP_N_FALSE_NEG]}
            assert self._seeded_id not in top_ids, (
                f"[{entry['id']}] Seeded entry {self._seeded_id!r} appeared in "
                f"top-{_TOP_N_FALSE_NEG} results for unrelated {mode!r} query "
                f"{q_text!r} — possible recall contamination."
            )
