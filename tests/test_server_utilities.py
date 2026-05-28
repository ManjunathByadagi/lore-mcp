"""Unit tests for server.py utility functions.

Targets previously uncovered branches in:
- json_serializer (datetime, date, UUID, Decimal, unknown type)
- _coerce_arguments (empty args, array coercion from JSON/CSV/space, boolean
  coercion, object coercion, non-string items schema skip)
- _is_production / _production_guard (env-var branching)
- _sanitize_search_query (metacharacter stripping)
- _backend_kind

These are pure functions with no DB dependency; they run without a live
db connection. Server-level module globals (db) are left as None since
none of these functions touch them.
"""

from __future__ import annotations

import decimal
import json
import uuid
from datetime import date, datetime

import pytest

import lore.server as srv

# ---------------------------------------------------------------------------
# json_serializer
# ---------------------------------------------------------------------------


def test_json_serializer_datetime():
    dt = datetime(2025, 5, 26, 12, 0, 0)
    result = srv.json_serializer(dt)
    assert result == dt.isoformat()


def test_json_serializer_date():
    d = date(2025, 5, 26)
    result = srv.json_serializer(d)
    assert result == "2025-05-26"


def test_json_serializer_uuid():
    u = uuid.uuid4()
    result = srv.json_serializer(u)
    assert result == str(u)


def test_json_serializer_decimal():
    d = decimal.Decimal("3.14159")
    result = srv.json_serializer(d)
    assert isinstance(result, float)
    assert abs(result - 3.14159) < 1e-4


def test_json_serializer_unknown_type_raises():
    with pytest.raises(TypeError, match="not JSON serializable"):
        srv.json_serializer(object())


def test_json_serializer_works_in_json_dumps():
    payload = {"ts": datetime(2025, 1, 1), "id": uuid.UUID("12345678-1234-5678-1234-567812345678")}
    result = json.dumps(payload, default=srv.json_serializer)
    decoded = json.loads(result)
    assert decoded["ts"] == "2025-01-01T00:00:00"
    assert decoded["id"] == "12345678-1234-5678-1234-567812345678"


# ---------------------------------------------------------------------------
# _is_production / _production_guard
# ---------------------------------------------------------------------------


def test_is_production_true_when_unset(monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)
    assert srv._is_production() is True


def test_is_production_true_explicit(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "production")
    assert srv._is_production() is True


def test_is_production_false_staging(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "staging")
    assert srv._is_production() is False


def test_is_production_false_development(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "development")
    assert srv._is_production() is False


def test_production_guard_blocks_in_production(monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)  # defaults to production
    result = srv._production_guard("kb_delete", confirm_production=False)
    assert result is not None
    assert result["ok"] is False
    assert "confirm_production" in result["message"]


def test_production_guard_passes_with_confirmation(monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)
    result = srv._production_guard("kb_delete", confirm_production=True)
    assert result is None


def test_production_guard_passes_with_dry_run(monkeypatch):
    monkeypatch.delenv("LORE_ENV", raising=False)
    result = srv._production_guard("kb_delete", confirm_production=False, dry_run=True)
    assert result is None


def test_production_guard_passes_outside_production(monkeypatch):
    monkeypatch.setenv("LORE_ENV", "development")
    result = srv._production_guard("kb_delete", confirm_production=False)
    assert result is None


# ---------------------------------------------------------------------------
# _sanitize_search_query
# ---------------------------------------------------------------------------


def test_sanitize_strips_metacharacters():
    q = "hello, world (test) [bracket]"
    result = srv._sanitize_search_query(q)
    assert "," not in result
    assert "(" not in result
    assert ")" not in result
    assert "[" not in result
    assert "]" not in result


def test_sanitize_strips_postgrest_operators():
    q = "eq.status,ilike.query"
    result = srv._sanitize_search_query(q)
    assert "eq" not in result
    assert "ilike" not in result


def test_sanitize_collapses_whitespace():
    q = "  hello   world  "
    result = srv._sanitize_search_query(q)
    assert result == "hello world"


def test_sanitize_plain_query_untouched():
    q = "Python asyncio patterns"
    result = srv._sanitize_search_query(q)
    assert result == q


def test_sanitize_empty_string():
    assert srv._sanitize_search_query("") == ""


# ---------------------------------------------------------------------------
# _backend_kind
# ---------------------------------------------------------------------------


def test_backend_kind_sqlite(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    assert srv._backend_kind() == "sqlite"


def test_backend_kind_postgres_variants(monkeypatch):
    for v in ("local", "postgres", "postgresql"):
        monkeypatch.setenv("DB_BACKEND", v)
        assert srv._backend_kind() == "postgres"


def test_backend_kind_empty_when_unset(monkeypatch):
    monkeypatch.delenv("DB_BACKEND", raising=False)
    assert srv._backend_kind() == ""


def test_backend_kind_unknown_value(monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "mysql")
    assert srv._backend_kind() == ""


# ---------------------------------------------------------------------------
# _coerce_arguments
# ---------------------------------------------------------------------------


def test_coerce_empty_arguments_returns_empty():
    result = srv._coerce_arguments({}, {"properties": {"tags": {"type": "array"}}})
    assert result == {}


def test_coerce_none_arguments_returns_none():
    result = srv._coerce_arguments(None, {"properties": {}})
    assert result is None


def test_coerce_empty_schema_passthrough():
    args = {"tags": "a,b,c"}
    result = srv._coerce_arguments(args, {})
    assert result == args


def test_coerce_array_from_json_string():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": '["alpha","beta","gamma"]'}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["alpha", "beta", "gamma"]


def test_coerce_array_from_comma_separated():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": "alpha,beta,gamma"}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["alpha", "beta", "gamma"]


def test_coerce_array_from_space_separated():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": "alpha beta gamma"}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["alpha", "beta", "gamma"]


def test_coerce_array_invalid_json_falls_back_to_split():
    """Malformed JSON that is not a list should fall back to comma/space split."""
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": "not-json-at-all"}
    result = srv._coerce_arguments(args, schema)
    assert isinstance(result["tags"], list)
    assert result["tags"] == ["not-json-at-all"]


def test_coerce_array_skips_non_string_items():
    """Non-string items schema should not do plain-text split."""
    schema = {"properties": {"ids": {"type": "array", "items": {"type": "integer"}}}}
    args = {"ids": "1 2 3"}  # Can't safely split to integers
    result = srv._coerce_arguments(args, schema)
    # Without string items, it should not split (leave as string or return original)
    # The coerce logic only splits when items.type == "string"
    assert result["ids"] == "1 2 3"


def test_coerce_object_from_json_string():
    schema = {"properties": {"meta": {"type": "object"}}}
    args = {"meta": '{"key": "value", "count": 5}'}
    result = srv._coerce_arguments(args, schema)
    assert result["meta"] == {"key": "value", "count": 5}


def test_coerce_object_invalid_json_leaves_as_string():
    schema = {"properties": {"meta": {"type": "object"}}}
    args = {"meta": "not-valid-json"}
    result = srv._coerce_arguments(args, schema)
    assert result["meta"] == "not-valid-json"


def test_coerce_boolean_from_true_string():
    schema = {"properties": {"flag": {"type": "boolean"}}}
    for val in ("true", "True", "TRUE", "1", "yes"):
        args = {"flag": val}
        result = srv._coerce_arguments(args, schema)
        assert result["flag"] is True, f"Expected True for {val!r}"


def test_coerce_boolean_from_false_string():
    schema = {"properties": {"flag": {"type": "boolean"}}}
    for val in ("false", "False", "FALSE", "0", "no"):
        args = {"flag": val}
        result = srv._coerce_arguments(args, schema)
        assert result["flag"] is False, f"Expected False for {val!r}"


def test_coerce_boolean_invalid_string_left_as_is():
    schema = {"properties": {"flag": {"type": "boolean"}}}
    args = {"flag": "maybe"}
    result = srv._coerce_arguments(args, schema)
    assert result["flag"] == "maybe"


def test_coerce_already_proper_types_unchanged():
    """Already-correct types must not be mutated."""
    schema = {
        "properties": {
            "tags": {"type": "array"},
            "flag": {"type": "boolean"},
            "meta": {"type": "object"},
        }
    }
    args = {"tags": ["a", "b"], "flag": True, "meta": {"x": 1}}
    result = srv._coerce_arguments(args, schema)
    assert result["tags"] == ["a", "b"]
    assert result["flag"] is True
    assert result["meta"] == {"x": 1}


def test_coerce_does_not_mutate_original():
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    original = {"tags": "a,b,c"}
    original_copy = dict(original)
    srv._coerce_arguments(original, schema)
    assert original == original_copy  # unchanged


def test_coerce_field_absent_in_args_skipped():
    """Fields defined in schema but absent in args are not added."""
    schema = {"properties": {"tags": {"type": "array"}}}
    args = {"title": "Hello"}
    result = srv._coerce_arguments(args, schema)
    assert "tags" not in result
    assert result["title"] == "Hello"


def test_coerce_empty_string_array_not_split():
    """Empty string value for array field should not produce ['']."""
    schema = {"properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    args = {"tags": ""}
    result = srv._coerce_arguments(args, schema)
    # Empty stripped string should not produce a list with one empty string
    # (the code checks `if stripped:`)
    assert result["tags"] == "" or result["tags"] == []


# ---------------------------------------------------------------------------
# _pg_format_vector_literal in server.py (server-side copy)
# ---------------------------------------------------------------------------


def test_server_pg_format_vector_literal_roundtrips():
    vec = [0.25, -0.5, 1.0]
    result = srv._pg_format_vector_literal(vec)
    inner = result[1:-1].split(",")
    parsed = [float(x) for x in inner]
    for orig, got in zip(vec, parsed):
        assert abs(orig - got) < 1e-9
