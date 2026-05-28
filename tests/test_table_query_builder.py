"""Unit tests for db_client.TableQuery's pure query-building methods.

TableQuery's filter setters (neq, gt, gte, lt, lte, like, ilike, in_, is_,
or_, order, limit, offset, maybe_single, upsert) and the WHERE/ORDER clause
builders (_build_where_clause, _build_order_clause) are pure state-manipulation
code that does NOT require a live database connection.

We construct TableQuery instances with a mock client so all execute() calls
that need a real cursor are NOT tested here (those require Postgres integration).
The goal is to cover the internal state machines and SQL string assembly.
"""

from __future__ import annotations

import pytest

from lore.db_client import TableQuery

# ---------------------------------------------------------------------------
# Minimal mock client (never calls _get_connection)
# ---------------------------------------------------------------------------


class _MockClient:
    """Fake LocalPostgresClient: provides _extras stub only."""

    class _extras:
        RealDictCursor = None  # not called by query-builder pure methods


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tq(table: str = "test_table") -> TableQuery:
    """Return a fresh TableQuery against the mock client."""
    return TableQuery(_MockClient(), table)


# ---------------------------------------------------------------------------
# Filter setter methods — verify _filters list state
# ---------------------------------------------------------------------------


def test_eq_appends_filter():
    q = _tq().select("*").eq("topic", "python")
    assert ("topic", "=", "python") in q._filters


def test_neq_appends_filter():
    q = _tq().neq("topic", "rust")
    assert ("topic", "!=", "rust") in q._filters


def test_gt_appends_filter():
    q = _tq().gt("trust_score", 0.5)
    assert ("trust_score", ">", 0.5) in q._filters


def test_gte_appends_filter():
    q = _tq().gte("trust_score", 0.5)
    assert ("trust_score", ">=", 0.5) in q._filters


def test_lt_appends_filter():
    q = _tq().lt("trust_score", 0.5)
    assert ("trust_score", "<", 0.5) in q._filters


def test_lte_appends_filter():
    q = _tq().lte("trust_score", 0.5)
    assert ("trust_score", "<=", 0.5) in q._filters


def test_like_appends_filter():
    q = _tq().like("title", "Python%")
    assert ("title", "LIKE", "Python%") in q._filters


def test_ilike_appends_filter():
    q = _tq().ilike("title", "python%")
    assert ("title", "ILIKE", "python%") in q._filters


def test_in_appends_filter_as_tuple():
    q = _tq().in_("topic", ["python", "rust"])
    # in_ stores value as tuple
    assert ("topic", "IN", ("python", "rust")) in q._filters


def test_is_null_appends_filter():
    q = _tq().is_("author", None)
    assert ("author", "IS", None) in q._filters


def test_is_not_null_appends_filter():
    q = _tq().is_("author", "NOT NULL")
    assert ("author", "IS", "NOT NULL") in q._filters


def test_chaining_multiple_filters():
    q = _tq().gt("trust_score", 0.5).lt("trust_score", 0.9)
    assert len(q._filters) == 2


# ---------------------------------------------------------------------------
# Ordering / paging state
# ---------------------------------------------------------------------------


def test_order_asc():
    q = _tq().order("created_at", desc=False)
    assert ("created_at", "ASC") in q._order_by


def test_order_desc():
    q = _tq().order("created_at", desc=True)
    assert ("created_at", "DESC") in q._order_by


def test_limit_sets_value():
    q = _tq().limit(42)
    assert q._limit_val == 42


def test_offset_sets_value():
    q = _tq().offset(10)
    assert q._offset_val == 10


def test_maybe_single_sets_flags():
    q = _tq().maybe_single()
    assert q._single_result is True
    assert q._limit_val == 1


# ---------------------------------------------------------------------------
# or_ filter storage
# ---------------------------------------------------------------------------


def test_or_stores_raw_condition():
    q = _tq().or_("title.wfts.python,content.wfts.python")
    assert len(q._or_filters) == 1
    assert q._or_filters[0] == "title.wfts.python,content.wfts.python"


# ---------------------------------------------------------------------------
# Operation setters
# ---------------------------------------------------------------------------


def test_select_sets_operation():
    q = _tq().select("kb_id, title")
    assert q._operation == "select"
    assert q._columns == "kb_id, title"


def test_select_with_count_mode():
    q = _tq().select("*", count="exact")
    assert q._count_mode == "exact"


def test_insert_sets_operation():
    q = _tq().insert({"kb_id": "x", "title": "T"})
    assert q._operation == "insert"
    assert len(q._data) == 1


def test_insert_list_sets_operation():
    q = _tq().insert([{"kb_id": "a"}, {"kb_id": "b"}])
    assert q._operation == "insert"
    assert len(q._data) == 2


def test_insert_upsert_true_sets_upsert_operation():
    q = _tq().insert({"kb_id": "x"}, upsert=True)
    assert q._operation == "upsert"


def test_upsert_sets_operation_and_conflict():
    q = _tq().upsert({"kb_id": "x", "title": "T"}, on_conflict="kb_id")
    assert q._operation == "upsert"
    assert q._on_conflict == "kb_id"


def test_update_sets_operation():
    q = _tq().update({"title": "New"})
    assert q._operation == "update"
    assert q._data == {"title": "New"}


def test_delete_sets_operation():
    q = _tq().delete()
    assert q._operation == "delete"


# ---------------------------------------------------------------------------
# _build_where_clause — SQL string construction (no DB needed)
# ---------------------------------------------------------------------------


def test_build_where_clause_empty():
    q = _tq()
    clause, values = q._build_where_clause()
    assert clause == ""
    assert values == []


def test_build_where_clause_eq():
    q = _tq().eq("topic", "python")
    clause, values = q._build_where_clause()
    assert "WHERE" in clause
    assert "topic = %s" in clause
    assert values == ["python"]


def test_build_where_clause_neq():
    q = _tq().neq("topic", "rust")
    clause, values = q._build_where_clause()
    assert "topic != %s" in clause
    assert values == ["rust"]


def test_build_where_clause_gt():
    q = _tq().gt("trust_score", 0.5)
    clause, values = q._build_where_clause()
    assert "trust_score > %s" in clause
    assert values == [0.5]


def test_build_where_clause_gte():
    q = _tq().gte("trust_score", 0.5)
    clause, values = q._build_where_clause()
    assert "trust_score >= %s" in clause


def test_build_where_clause_lt():
    q = _tq().lt("trust_score", 0.5)
    clause, values = q._build_where_clause()
    assert "trust_score < %s" in clause


def test_build_where_clause_lte():
    q = _tq().lte("trust_score", 0.5)
    clause, values = q._build_where_clause()
    assert "trust_score <= %s" in clause


def test_build_where_clause_like():
    q = _tq().like("title", "Python%")
    clause, values = q._build_where_clause()
    assert "title LIKE %s" in clause
    assert values == ["Python%"]


def test_build_where_clause_ilike():
    q = _tq().ilike("title", "python%")
    clause, values = q._build_where_clause()
    assert "title ILIKE %s" in clause


def test_build_where_clause_in():
    q = _tq().in_("topic", ["python", "rust"])
    clause, values = q._build_where_clause()
    assert "topic IN (" in clause
    assert "%s, %s" in clause or "%s" in clause  # at least one placeholder
    assert "python" in values
    assert "rust" in values


def test_build_where_clause_is_null():
    q = _tq().is_("author", None)
    clause, values = q._build_where_clause()
    assert "author IS NULL" in clause
    # IS NULL doesn't add a value to the params list
    assert values == []


def test_build_where_clause_is_not_null():
    q = _tq().is_("author", "something")
    clause, values = q._build_where_clause()
    assert "author IS NOT NULL" in clause


def test_build_where_clause_multiple_filters_joined_with_and():
    q = _tq().eq("topic", "python").gt("trust_score", 0.5)
    clause, values = q._build_where_clause()
    assert " AND " in clause
    assert len(values) == 2


def test_build_where_clause_or_filter_wfts():
    q = _tq().or_("title.wfts.python,content.wfts.python")
    clause, values = q._build_where_clause()
    assert "websearch_to_tsquery" in clause
    assert "python" in values


def test_build_where_clause_or_filter_plfts():
    q = _tq().or_("title.plfts.python,content.plfts.python")
    clause, values = q._build_where_clause()
    assert "plainto_tsquery" in clause


# ---------------------------------------------------------------------------
# _build_order_clause
# ---------------------------------------------------------------------------


def test_build_order_clause_empty():
    q = _tq()
    assert q._build_order_clause() == ""


def test_build_order_clause_single_asc():
    q = _tq().order("created_at")
    clause = q._build_order_clause()
    assert "ORDER BY" in clause
    assert "created_at ASC" in clause


def test_build_order_clause_single_desc():
    q = _tq().order("updated_at", desc=True)
    clause = q._build_order_clause()
    assert "updated_at DESC" in clause


def test_build_order_clause_multiple():
    q = _tq().order("topic").order("created_at", desc=True)
    clause = q._build_order_clause()
    assert "topic ASC" in clause
    assert "created_at DESC" in clause


# ---------------------------------------------------------------------------
# _serialize_value
# ---------------------------------------------------------------------------


def test_serialize_value_dict_to_json():
    q = _tq()
    result = q._serialize_value({"key": "val"})
    import json
    assert json.loads(result) == {"key": "val"}


def test_serialize_value_list_passthrough():
    q = _tq()
    result = q._serialize_value(["a", "b"])
    assert result == ["a", "b"]


def test_serialize_value_primitive_passthrough():
    q = _tq()
    assert q._serialize_value(42) == 42
    assert q._serialize_value("hello") == "hello"
    assert q._serialize_value(None) is None
    assert q._serialize_value(3.14) == 3.14
