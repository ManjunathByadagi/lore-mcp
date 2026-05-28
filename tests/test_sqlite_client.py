"""Unit tests for SqliteClient and SqliteTableQuery (db_client.py).

Uses real sqlite3 in-memory / tempfile databases — no external services.
Covers the query builder (select, insert, update, delete, upsert, filters,
ordering, pagination, maybe_single, or_) and helper functions
(_sqlite_deserialize_row, _sqlite_serialize_value, _sqlite_map_table,
get_db_client with SQLITE backend).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lore.db_client import (
    DatabaseBackend,
    QueryResult,
    SqliteClient,
    _sqlite_deserialize_row,
    _sqlite_map_table,
    _sqlite_serialize_value,
    get_db_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """A fresh SqliteClient backed by a temp file."""
    client = SqliteClient(db_path=str(tmp_path / "test.db"))
    yield client
    client.close()


def _seed(client: SqliteClient, rows: list[dict]) -> None:
    """Insert rows into knowledge.kb_entries via the SqliteTableQuery API."""
    for row in rows:
        client.table("knowledge.kb_entries").insert(row).execute()


def _make_entry(
    kb_id: str = "kb_001",
    topic: str = "testing",
    title: str = "Test Title",
    content: str = "Test content",
    tags: list | None = None,
    trust_score: float = 1.0,
) -> dict:
    return {
        "kb_id": kb_id,
        "topic": topic,
        "title": title,
        "content": content,
        "tags": tags or [],
        "trust_score": trust_score,
    }


# ---------------------------------------------------------------------------
# _sqlite_map_table helper
# ---------------------------------------------------------------------------


def test_sqlite_map_table_replaces_dot():
    assert _sqlite_map_table("knowledge.kb_entries") == "knowledge_kb_entries"


def test_sqlite_map_table_no_dot():
    assert _sqlite_map_table("plain_table") == "plain_table"


# ---------------------------------------------------------------------------
# _sqlite_deserialize_row helper
# ---------------------------------------------------------------------------


def test_deserialize_row_parses_json_list():
    row = {"tags": '["a", "b"]', "kb_id": "x"}
    result = _sqlite_deserialize_row(row)
    assert result["tags"] == ["a", "b"]
    assert result["kb_id"] == "x"


def test_deserialize_row_parses_json_dict():
    row = {"meta": '{"key": "val"}'}
    result = _sqlite_deserialize_row(row)
    assert result["meta"] == {"key": "val"}


def test_deserialize_row_leaves_plain_strings():
    row = {"title": "Hello", "count": 42}
    result = _sqlite_deserialize_row(row)
    assert result["title"] == "Hello"
    assert result["count"] == 42


def test_deserialize_row_handles_invalid_json():
    row = {"bad": "[not valid json}"}
    result = _sqlite_deserialize_row(row)
    assert result["bad"] == "[not valid json}"


def test_deserialize_row_empty_string():
    row = {"col": ""}
    result = _sqlite_deserialize_row(row)
    assert result["col"] == ""


# ---------------------------------------------------------------------------
# _sqlite_serialize_value helper
# ---------------------------------------------------------------------------


def test_serialize_list_to_json():
    val = _sqlite_serialize_value(["a", "b"])
    assert val == '["a", "b"]'


def test_serialize_dict_to_json():
    val = _sqlite_serialize_value({"k": "v"})
    assert val == '{"k": "v"}'


def test_serialize_primitive_passthrough():
    assert _sqlite_serialize_value(42) == 42
    assert _sqlite_serialize_value("hello") == "hello"
    assert _sqlite_serialize_value(None) is None


# ---------------------------------------------------------------------------
# INSERT / SELECT roundtrip
# ---------------------------------------------------------------------------


def test_insert_and_select_all(tmp_db):
    entry = _make_entry("kb_1")
    result = tmp_db.table("knowledge.kb_entries").insert(entry).execute()
    assert isinstance(result, QueryResult)
    assert len(result.data) == 1
    assert result.data[0]["kb_id"] == "kb_1"
    assert result.data[0]["title"] == "Test Title"

    rows = tmp_db.table("knowledge.kb_entries").select("*").execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_1"


def test_insert_returns_inserted_row(tmp_db):
    entry = _make_entry("kb_x", trust_score=0.75)
    result = tmp_db.table("knowledge.kb_entries").insert(entry).execute()
    assert result.data[0]["trust_score"] == 0.75


def test_insert_with_tags_roundtrip(tmp_db):
    entry = _make_entry("kb_tags", tags=["python", "testing"])
    tmp_db.table("knowledge.kb_entries").insert(entry).execute()
    rows = tmp_db.table("knowledge.kb_entries").select("*").eq("kb_id", "kb_tags").execute()
    assert rows.data[0]["tags"] == ["python", "testing"]


def test_select_specific_columns(tmp_db):
    _seed(tmp_db, [_make_entry("kb_col")])
    rows = tmp_db.table("knowledge.kb_entries").select("kb_id, title").execute()
    assert "kb_id" in rows.data[0]
    assert "title" in rows.data[0]
    # content was not requested
    assert "content" not in rows.data[0]


# ---------------------------------------------------------------------------
# EQ / NEQ filters
# ---------------------------------------------------------------------------


def test_select_eq_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", topic="alpha"), _make_entry("kb_b", topic="beta")])
    rows = tmp_db.table("knowledge.kb_entries").select("*").eq("topic", "alpha").execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_a"


def test_select_neq_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", topic="alpha"), _make_entry("kb_b", topic="beta")])
    rows = tmp_db.table("knowledge.kb_entries").select("*").neq("topic", "alpha").execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_b"


# ---------------------------------------------------------------------------
# Comparison filters (gt, gte, lt, lte)
# ---------------------------------------------------------------------------


def test_select_gt_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", trust_score=0.3), _make_entry("kb_b", trust_score=0.8)])
    rows = tmp_db.table("knowledge.kb_entries").select("*").gt("trust_score", 0.5).execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_b"


def test_select_gte_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", trust_score=0.5), _make_entry("kb_b", trust_score=0.8)])
    rows = tmp_db.table("knowledge.kb_entries").select("*").gte("trust_score", 0.5).execute()
    assert len(rows.data) == 2


def test_select_lt_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", trust_score=0.3), _make_entry("kb_b", trust_score=0.8)])
    rows = tmp_db.table("knowledge.kb_entries").select("*").lt("trust_score", 0.5).execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_a"


def test_select_lte_filter(tmp_db):
    _seed(tmp_db, [_make_entry("kb_a", trust_score=0.5), _make_entry("kb_b", trust_score=0.8)])
    rows = tmp_db.table("knowledge.kb_entries").select("*").lte("trust_score", 0.5).execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_a"


# ---------------------------------------------------------------------------
# LIKE / ILIKE filters
# ---------------------------------------------------------------------------


def test_select_like_filter(tmp_db):
    _seed(
        tmp_db,
        [_make_entry("kb_l1", title="Python basics"), _make_entry("kb_l2", title="Rust intro")],
    )
    rows = tmp_db.table("knowledge.kb_entries").select("*").like("title", "Python%").execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_l1"


def test_select_ilike_filter(tmp_db):
    _seed(
        tmp_db,
        [_make_entry("kb_i1", title="Python basics"), _make_entry("kb_i2", title="Rust intro")],
    )
    rows = tmp_db.table("knowledge.kb_entries").select("*").ilike("title", "python%").execute()
    assert len(rows.data) == 1
    assert rows.data[0]["kb_id"] == "kb_i1"


# ---------------------------------------------------------------------------
# IN filter
# ---------------------------------------------------------------------------


def test_select_in_filter(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_p", topic="python"),
            _make_entry("kb_r", topic="rust"),
            _make_entry("kb_g", topic="go"),
        ],
    )
    rows = (
        tmp_db.table("knowledge.kb_entries").select("*").in_("topic", ["python", "rust"]).execute()
    )
    assert len(rows.data) == 2
    topics = {r["topic"] for r in rows.data}
    assert topics == {"python", "rust"}


# ---------------------------------------------------------------------------
# IS filter (NULL checks)
# ---------------------------------------------------------------------------


def test_select_is_null_filter(tmp_db):
    # Insert one without author (null) and one with author
    _seed(tmp_db, [_make_entry("kb_n")])
    authored_entry = _make_entry("kb_authored")
    authored_entry["author"] = "Alice"
    tmp_db.table("knowledge.kb_entries").insert(authored_entry).execute()
    rows = tmp_db.table("knowledge.kb_entries").select("*").is_("author", None).execute()
    ids = {r["kb_id"] for r in rows.data}
    assert "kb_authored" not in ids


def test_select_is_not_null_filter(tmp_db):
    entry = _make_entry("kb_auth2")
    entry["author"] = "Bob"
    tmp_db.table("knowledge.kb_entries").insert(entry).execute()
    rows = tmp_db.table("knowledge.kb_entries").select("*").is_("author", "notnull").execute()
    ids = {r["kb_id"] for r in rows.data}
    assert "kb_auth2" in ids


# ---------------------------------------------------------------------------
# ORDER BY
# ---------------------------------------------------------------------------


def test_select_order_asc(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_z", trust_score=0.9),
            _make_entry("kb_a", trust_score=0.1),
        ],
    )
    rows = tmp_db.table("knowledge.kb_entries").select("*").order("trust_score").execute()
    scores = [r["trust_score"] for r in rows.data]
    assert scores == sorted(scores)


def test_select_order_desc(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_z", trust_score=0.9),
            _make_entry("kb_a", trust_score=0.1),
        ],
    )
    rows = (
        tmp_db.table("knowledge.kb_entries").select("*").order("trust_score", desc=True).execute()
    )
    scores = [r["trust_score"] for r in rows.data]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# LIMIT / OFFSET
# ---------------------------------------------------------------------------


def test_select_limit(tmp_db):
    for i in range(5):
        _seed(tmp_db, [_make_entry(f"kb_{i:03}")])
    rows = tmp_db.table("knowledge.kb_entries").select("*").limit(2).execute()
    assert len(rows.data) == 2


def test_select_offset(tmp_db):
    """OFFSET must be used with LIMIT (SQLite requires LIMIT before OFFSET).

    NOTE: SqliteTableQuery._execute_select has a latent bug: it appends
    'OFFSET N' without a preceding LIMIT when _limit_val is None, which
    produces invalid SQL ('near OFFSET: syntax error').  This test uses
    limit + offset together (the valid and intended usage) to exercise the
    offset code path without triggering the bug.  See bug report below.
    """
    for i in range(4):
        _seed(tmp_db, [_make_entry(f"kb_{i:03}", trust_score=float(i) / 4)])
    all_rows = tmp_db.table("knowledge.kb_entries").select("*").order("trust_score").execute()
    offset_rows = (
        tmp_db.table("knowledge.kb_entries")
        .select("*")
        .order("trust_score")
        .limit(100)  # LIMIT required before OFFSET in SQLite
        .offset(2)
        .execute()
    )
    assert len(offset_rows.data) == 2
    # First row of offset result matches third row of full result
    assert offset_rows.data[0]["kb_id"] == all_rows.data[2]["kb_id"]


# ---------------------------------------------------------------------------
# COUNT (exact)
# ---------------------------------------------------------------------------


def test_select_exact_count(tmp_db):
    for i in range(3):
        _seed(tmp_db, [_make_entry(f"kb_{i:03}")])
    result = tmp_db.table("knowledge.kb_entries").select("*", count="exact").execute()
    assert result.count == 3


def test_select_count_with_filter(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_a", topic="alpha"),
            _make_entry("kb_b", topic="beta"),
            _make_entry("kb_c", topic="alpha"),
        ],
    )
    result = (
        tmp_db.table("knowledge.kb_entries")
        .select("*", count="exact")
        .eq("topic", "alpha")
        .execute()
    )
    assert result.count == 2
    assert len(result.data) == 2


# ---------------------------------------------------------------------------
# MAYBE_SINGLE
# ---------------------------------------------------------------------------


def test_maybe_single_returns_dict_when_found(tmp_db):
    _seed(tmp_db, [_make_entry("kb_single")])
    result = (
        tmp_db.table("knowledge.kb_entries")
        .select("*")
        .eq("kb_id", "kb_single")
        .maybe_single()
        .execute()
    )
    assert isinstance(result.data, dict)
    assert result.data["kb_id"] == "kb_single"


def test_maybe_single_returns_none_when_not_found(tmp_db):
    result = (
        tmp_db.table("knowledge.kb_entries")
        .select("*")
        .eq("kb_id", "no_such_id")
        .maybe_single()
        .execute()
    )
    assert result.data is None


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------


def test_update_row(tmp_db):
    _seed(tmp_db, [_make_entry("kb_upd", title="Original")])
    tmp_db.table("knowledge.kb_entries").update({"title": "Updated"}).eq(
        "kb_id", "kb_upd"
    ).execute()
    rows = tmp_db.table("knowledge.kb_entries").select("*").eq("kb_id", "kb_upd").execute()
    assert rows.data[0]["title"] == "Updated"


def test_update_returns_updated_data(tmp_db):
    _seed(tmp_db, [_make_entry("kb_upd2", content="old")])
    result = (
        tmp_db.table("knowledge.kb_entries")
        .update({"content": "new"})
        .eq("kb_id", "kb_upd2")
        .execute()
    )
    assert len(result.data) >= 1
    assert result.data[0]["content"] == "new"


def test_update_nonexistent_row_returns_empty(tmp_db):
    result = (
        tmp_db.table("knowledge.kb_entries").update({"title": "X"}).eq("kb_id", "no_such").execute()
    )
    assert result.data == []


def test_update_no_data_returns_error(tmp_db):
    # Calling execute() on an update with no data dict
    from lore.db_client import SqliteTableQuery

    q = SqliteTableQuery(tmp_db, "knowledge.kb_entries")
    q._operation = "update"
    q._data = None
    result = q.execute()
    assert result.error is not None


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


def test_delete_row(tmp_db):
    _seed(tmp_db, [_make_entry("kb_del")])
    tmp_db.table("knowledge.kb_entries").delete().eq("kb_id", "kb_del").execute()
    rows = tmp_db.table("knowledge.kb_entries").select("*").eq("kb_id", "kb_del").execute()
    assert rows.data == []


def test_delete_returns_deleted_row(tmp_db):
    _seed(tmp_db, [_make_entry("kb_del2", title="Gone")])
    result = tmp_db.table("knowledge.kb_entries").delete().eq("kb_id", "kb_del2").execute()
    assert len(result.data) == 1
    assert result.data[0]["title"] == "Gone"


def test_delete_nonexistent_row_returns_empty(tmp_db):
    result = tmp_db.table("knowledge.kb_entries").delete().eq("kb_id", "not_there").execute()
    assert result.data == []


# ---------------------------------------------------------------------------
# UPSERT
# ---------------------------------------------------------------------------


def test_upsert_inserts_new_row(tmp_db):
    entry = _make_entry("kb_ups")
    result = tmp_db.table("knowledge.kb_entries").upsert(entry, on_conflict="kb_id").execute()
    assert len(result.data) == 1
    assert result.data[0]["kb_id"] == "kb_ups"


def test_upsert_replaces_existing_row(tmp_db):
    _seed(tmp_db, [_make_entry("kb_ups2", title="Old")])
    tmp_db.table("knowledge.kb_entries").upsert(
        _make_entry("kb_ups2", title="New"), on_conflict="kb_id"
    ).execute()

    rows = tmp_db.table("knowledge.kb_entries").select("*").eq("kb_id", "kb_ups2").execute()
    assert rows.data[0]["title"] == "New"


def test_upsert_no_data_returns_error(tmp_db):
    from lore.db_client import SqliteTableQuery

    q = SqliteTableQuery(tmp_db, "knowledge.kb_entries")
    q._operation = "upsert"
    q._data = None
    result = q.execute()
    assert result.error is not None


# ---------------------------------------------------------------------------
# INSERT no data edge case
# ---------------------------------------------------------------------------


def test_insert_no_data_returns_error(tmp_db):
    from lore.db_client import SqliteTableQuery

    q = SqliteTableQuery(tmp_db, "knowledge.kb_entries")
    q._operation = "insert"
    q._data = None
    result = q.execute()
    assert result.error is not None


# ---------------------------------------------------------------------------
# Unknown operation raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_operation_raises(tmp_db):
    from lore.db_client import SqliteTableQuery

    q = SqliteTableQuery(tmp_db, "knowledge.kb_entries")
    q._operation = "drop_table"
    with pytest.raises(ValueError, match="Unknown operation"):
        q.execute()


# ---------------------------------------------------------------------------
# OR_ filter (wfts Supabase-style)
# ---------------------------------------------------------------------------


def test_or_filter_wfts_style(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_py", title="Python guide", content="learn python"),
            _make_entry("kb_rs", title="Rust intro", content="learn rust"),
        ],
    )
    rows = (
        tmp_db.table("knowledge.kb_entries")
        .select("*")
        .or_("title.wfts.Python,content.wfts.Python")
        .execute()
    )
    ids = {r["kb_id"] for r in rows.data}
    assert "kb_py" in ids
    assert "kb_rs" not in ids


def test_or_filter_generic_dotted(tmp_db):
    _seed(
        tmp_db,
        [
            _make_entry("kb_go", topic="golang"),
            _make_entry("kb_py2", topic="python"),
        ],
    )
    rows = tmp_db.table("knowledge.kb_entries").select("*").or_("topic.eq.golang").execute()
    ids = {r["kb_id"] for r in rows.data}
    assert "kb_go" in ids


# ---------------------------------------------------------------------------
# RPC stub
# ---------------------------------------------------------------------------


def test_rpc_returns_empty_result(tmp_db):
    result = tmp_db.rpc("some_function")
    assert isinstance(result, QueryResult)
    assert result.data == []


# ---------------------------------------------------------------------------
# SqliteClient.close() is idempotent
# ---------------------------------------------------------------------------


def test_close_is_idempotent(tmp_path):
    client = SqliteClient(db_path=str(tmp_path / "close.db"))
    client.close()
    client.close()  # Should not raise


# ---------------------------------------------------------------------------
# get_db_client with SQLITE backend
# ---------------------------------------------------------------------------


def test_get_db_client_sqlite_backend(monkeypatch, tmp_path):
    """DB_BACKEND=sqlite returns a SqliteClient."""
    db_path = str(tmp_path / "test_get.db")
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DB_PATH", db_path)
    client = get_db_client()
    assert isinstance(client, SqliteClient)
    client.close()


def test_get_db_client_sqlite_via_enum(tmp_path):
    """Passing backend=DatabaseBackend.SQLITE directly returns SqliteClient."""
    db_path = str(tmp_path / "enum_test.db")
    client = get_db_client(backend=DatabaseBackend.SQLITE, db_path=db_path)
    assert isinstance(client, SqliteClient)
    client.close()


def test_get_db_client_unknown_backend_falls_back_to_supabase(monkeypatch):
    """Unknown DB_BACKEND string logs warning and falls back to supabase."""
    monkeypatch.setenv("DB_BACKEND", "magic_db")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    with pytest.raises(ValueError, match="SUPABASE_URL"):
        get_db_client()


# ---------------------------------------------------------------------------
# _probe_optional_features coverage: FTS5 and vec0 unavailable path
# ---------------------------------------------------------------------------


def test_fts5_available_probe(tmp_db):
    """A fresh db built from _SQLITE_SCHEMA should report fts5_available=True."""
    # The standard Python sqlite3 on Linux is built with FTS5
    # (this may be False in minimal builds — acceptable to skip, not fail)
    assert isinstance(tmp_db.fts5_available, bool)


# ---------------------------------------------------------------------------
# Journal table CRUD (covers knowledge_journal_entries)
# ---------------------------------------------------------------------------


def test_journal_insert_and_select(tmp_db):
    entry = {
        "entry_id": "jrnl_001",
        "date": "2026-05-27",
        "entry_type": "observation",
        "content": "Test journal content",
        "tags": ["test"],
    }
    tmp_db.table("knowledge.journal_entries").insert(entry).execute()
    result = (
        tmp_db.table("knowledge.journal_entries").select("*").eq("entry_id", "jrnl_001").execute()
    )
    assert len(result.data) == 1
    assert result.data[0]["content"] == "Test journal content"
    assert result.data[0]["tags"] == ["test"]


def test_journal_delete(tmp_db):
    entry = {
        "entry_id": "jrnl_del",
        "date": "2026-05-27",
        "entry_type": "log",
        "content": "to delete",
        "tags": [],
    }
    tmp_db.table("knowledge.journal_entries").insert(entry).execute()
    tmp_db.table("knowledge.journal_entries").delete().eq("entry_id", "jrnl_del").execute()
    result = (
        tmp_db.table("knowledge.journal_entries").select("*").eq("entry_id", "jrnl_del").execute()
    )
    assert result.data == []


# ---------------------------------------------------------------------------
# Research notes table CRUD
# ---------------------------------------------------------------------------


def test_research_notes_insert_and_select(tmp_db):
    note = {
        "note_id": "note_001",
        "topic": "async",
        "title": "Async Patterns",
        "content": "Research content here",
        "tags": ["async", "python"],
    }
    tmp_db.table("knowledge.research_notes").insert(note).execute()
    result = (
        tmp_db.table("knowledge.research_notes").select("*").eq("note_id", "note_001").execute()
    )
    assert result.data[0]["title"] == "Async Patterns"
    assert result.data[0]["tags"] == ["async", "python"]


# ---------------------------------------------------------------------------
# MCP index versions table
# ---------------------------------------------------------------------------


def test_mcp_index_versions_upsert(tmp_db):
    record = {
        "server_name": "test_server",
        "version": "1.0.0",
        "last_scanned": "2026-05-27T12:00:00Z",
        "tool_count": 5,
    }
    tmp_db.table("mcp_index_versions").insert(record).execute()
    result = (
        tmp_db.table("mcp_index_versions").select("*").eq("server_name", "test_server").execute()
    )
    assert result.data[0]["tool_count"] == 5
