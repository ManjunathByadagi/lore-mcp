"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import os

import pytest

from lore.search import (
    candidate_pool_size,
    reciprocal_rank_fusion,
    rrf_k,
)


def test_rrf_empty_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_preserves_order():
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [kb_id for kb_id, _ in result]
    assert ids == ["a", "b", "c"]
    # Scores should be strictly decreasing.
    scores = [score for _, score in result]
    assert scores[0] > scores[1] > scores[2]


def test_rrf_overlap_ranks_intersection_higher():
    """An item in both lists ranks above items in only one."""
    result = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    scores = dict(result)
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_rrf_tie_breaking_is_stable():
    """Items at the same rank in different lists tie cleanly."""
    result = reciprocal_rank_fusion([["a"], ["b"]])
    scores = dict(result)
    # k+1 for both -> equal scores
    assert math.isclose(scores["a"], scores["b"])


def test_rrf_k_parameter_overrides_env(monkeypatch):
    """Explicit k= overrides LORE_RRF_K env var."""
    monkeypatch.setenv("LORE_RRF_K", "10")
    # k=1 makes top-of-list weight much heavier than k=100
    high_k = reciprocal_rank_fusion([["a", "b"]], k=100)
    low_k = reciprocal_rank_fusion([["a", "b"]], k=1)
    # Score gap between rank 1 and rank 2 is wider for small k.
    high_gap = dict(high_k)["a"] - dict(high_k)["b"]
    low_gap = dict(low_k)["a"] - dict(low_k)["b"]
    assert low_gap > high_gap


def test_rrf_env_k_default_when_unset(monkeypatch):
    monkeypatch.delenv("LORE_RRF_K", raising=False)
    assert rrf_k() == 10


def test_rrf_env_k_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("LORE_RRF_K", "not-an-int")
    assert rrf_k() == 10


def test_rrf_duplicate_in_single_list_counts_once():
    """Bug guard: duplicates in one ranking should not double-count."""
    result = reciprocal_rank_fusion([["a", "a", "b"]])
    scores = dict(result)
    # Only the first 'a' contributes; 'a' was at rank 1, 'b' at rank 3.
    expected_a = 1.0 / (rrf_k() + 1)
    expected_b = 1.0 / (rrf_k() + 3)
    assert math.isclose(scores["a"], expected_a)
    assert math.isclose(scores["b"], expected_b)


@pytest.mark.parametrize(
    "top_k,corpus_size,expected",
    [
        (5, 0, 50),  # empty corpus -> floor at top_k or 50
        (5, 10, 10),  # small corpus -> bounded by corpus
        (20, 1000, 100),  # mid corpus -> max(top_k*5, 50) = 100
        (50, 10000, 200),  # huge -> capped at 200
        (1, 1, 1),  # min case
    ],
)
def test_candidate_pool_size(top_k: int, corpus_size: int, expected: int):
    assert candidate_pool_size(top_k, corpus_size) == expected
