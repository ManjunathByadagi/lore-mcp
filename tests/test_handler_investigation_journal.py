"""Unit tests for Investigation, Journal, and Snapshot handlers in server.py.

Uses the same fake-db pattern as test_server_handlers.py. No live database
or network connections.  Assertions target the ResponseEnvelope contract:
ok, error, message, data keys.
"""

from __future__ import annotations

import pytest

import lore.server as srv
from lore.db_client import QueryResult

# ---------------------------------------------------------------------------
# Minimal fluent fake — same pattern as other handler test files
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, db: _FakeDb):
        self._db = db
        self._op = "select"

    def select(self, *_a, **_k):
        return self

    def insert(self, data):
        self._op = "insert"
        self._db.inserts.append(data)
        return self

    def update(self, data):
        self._op = "update"
        return self

    def delete(self):
        self._op = "delete"
        return self

    def eq(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def offset(self, *_a, **_k):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        if self._op == "insert":
            return QueryResult(data=self._db.inserts or [])
        return QueryResult(data=self._db.current_row, count=self._db.count)


class _FakeDb:
    def __init__(self, current_row=None, count: int | None = None):
        self.current_row = current_row if current_row is not None else []
        self.count = count
        self.inserts: list = []

    def table(self, _name):
        return _FakeQuery(self)


class _ErrorDb:
    """A db that always raises on table()."""

    def __init__(self, exc=RuntimeError("db error")):
        self._exc = exc

    def table(self, _name):
        raise self._exc


# ---------------------------------------------------------------------------
# Investigation handlers
# ---------------------------------------------------------------------------


def test_investigation_add_success(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_investigation_add(
        topic="performance", title="Profiling notes", content="cProfile results here"
    )
    assert resp["ok"] is True
    assert resp["data"]["note_id"].startswith("note_")
    assert "Profiling notes" in resp["message"]


def test_investigation_add_with_tags(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_investigation_add(
        topic="async", title="Async perf", content="content", tags=["asyncio", "benchmark"]
    )
    assert resp["ok"] is True
    assert resp["data"]["note_id"].startswith("note_")


def test_investigation_add_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_investigation_add(topic="t", title="T", content="c")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_investigation_list_success(monkeypatch):
    notes = [
        {"note_id": "note_1", "topic": "t", "title": "A"},
        {"note_id": "note_2", "topic": "t", "title": "B"},
    ]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=notes))
    resp = srv.handle_investigation_list()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 2
    assert len(resp["data"]["investigations"]) == 2


def test_investigation_list_with_topic_filter(monkeypatch):
    notes = [{"note_id": "note_py", "topic": "python", "title": "Python notes"}]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=notes))
    resp = srv.handle_investigation_list(topic="python")
    assert resp["ok"] is True
    assert resp["data"]["count"] == 1


def test_investigation_list_empty(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_investigation_list()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 0


def test_investigation_list_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_investigation_list()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_investigation_get_success(monkeypatch):
    note = {"note_id": "note_42", "topic": "perf", "title": "Perf study", "content": "data"}
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=note))
    resp = srv.handle_investigation_get(note_id="note_42")
    assert resp["ok"] is True
    assert resp["data"]["note_id"] == "note_42"
    assert "Perf study" in resp["message"]


def test_investigation_get_not_found_none(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=None))
    resp = srv.handle_investigation_get(note_id="note_missing")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_investigation_get_not_found_empty_dict(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row={}))
    resp = srv.handle_investigation_get(note_id="note_empty")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_investigation_get_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_investigation_get(note_id="note_x")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_investigation_log_experiment_success(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_investigation_log_experiment(
        title="Cache experiment",
        hypothesis="LRU cache reduces DB calls by 30%",
        methodology="Instrument DB calls before/after",
        results={"db_calls_before": 100, "db_calls_after": 70},
        conclusion="Confirmed 30% reduction",
    )
    assert resp["ok"] is True
    assert resp["data"]["experiment_id"].startswith("exp_")
    assert "Cache experiment" in resp["message"]


def test_investigation_log_experiment_minimal(monkeypatch):
    """Only title is required; others are optional."""
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_investigation_log_experiment(title="Minimal experiment")
    assert resp["ok"] is True
    assert resp["data"]["experiment_id"].startswith("exp_")


def test_investigation_log_experiment_db_error(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_investigation_log_experiment(title="X")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_investigation_list_experiments_success(monkeypatch):
    experiments = [
        {"experiment_id": "exp_1", "title": "Exp A"},
        {"experiment_id": "exp_2", "title": "Exp B"},
    ]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=experiments))
    resp = srv.handle_investigation_list_experiments()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 2


def test_investigation_list_experiments_empty(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_investigation_list_experiments()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 0


def test_investigation_list_experiments_db_error(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_investigation_list_experiments()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# Journal handlers
# ---------------------------------------------------------------------------


def test_journal_append_success(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_journal_append(
        entry_type="observation", content="Noticed latency spike at 14:00"
    )
    assert resp["ok"] is True
    assert resp["data"]["entry_id"].startswith("jrnl_")
    assert "date" in resp["data"]


def test_journal_append_with_tags(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_journal_append(
        entry_type="milestone", content="v1.0 shipped", tags=["release", "v1"]
    )
    assert resp["ok"] is True


def test_journal_append_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_journal_append(entry_type="log", content="data")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_journal_list_success(monkeypatch):
    entries = [
        {"entry_id": "jrnl_1", "date": "2026-05-27", "entry_type": "observation"},
        {"entry_id": "jrnl_2", "date": "2026-05-26", "entry_type": "milestone"},
    ]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries))
    resp = srv.handle_journal_list()
    assert resp["ok"] is True
    assert resp["data"]["count"] == 2


def test_journal_list_custom_limit(monkeypatch):
    entries = [{"entry_id": "jrnl_1"}]
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entries))
    resp = srv.handle_journal_list(limit=5)
    assert resp["ok"] is True


def test_journal_list_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_journal_list()
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


def test_journal_get_success(monkeypatch):
    entry = {
        "entry_id": "jrnl_42",
        "date": "2026-05-27",
        "entry_type": "observation",
        "content": "content here",
    }
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=entry))
    resp = srv.handle_journal_get(entry_id="jrnl_42")
    assert resp["ok"] is True
    assert resp["data"]["entry_id"] == "jrnl_42"
    assert "2026-05-27" in resp["message"]


def test_journal_get_not_found_none(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=None))
    resp = srv.handle_journal_get(entry_id="jrnl_ghost")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_journal_get_not_found_empty(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb(current_row={}))
    resp = srv.handle_journal_get(entry_id="jrnl_empty")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_journal_get_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_journal_get(entry_id="jrnl_x")
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# handle_snapshot_config
# ---------------------------------------------------------------------------


def test_snapshot_config_success(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_snapshot_config(
        config_name="lore_prod",
        config_data={"db_backend": "postgres", "semantic": True},
    )
    assert resp["ok"] is True
    assert resp["data"]["entry_id"].startswith("jrnl_")
    assert "lore_prod" in resp["message"]


def test_snapshot_config_empty_config_data(monkeypatch):
    monkeypatch.setattr(srv, "db", _FakeDb())
    resp = srv.handle_snapshot_config(config_name="empty_cfg", config_data={})
    assert resp["ok"] is True


def test_snapshot_config_db_error_returns_envelope(monkeypatch):
    monkeypatch.setattr(srv, "db", _ErrorDb())
    resp = srv.handle_snapshot_config(config_name="cfg", config_data={"k": "v"})
    assert resp["ok"] is False
    assert resp["error"] == "unexpected_exception"


# ---------------------------------------------------------------------------
# _journal_like_score helper (coverage for the scoring heuristic)
# ---------------------------------------------------------------------------


def test_journal_like_score_empty_query():
    assert srv._journal_like_score("some content", "") == 0.0


def test_journal_like_score_empty_content():
    assert srv._journal_like_score("", "query") == 0.0


def test_journal_like_score_phrase_match():
    score = srv._journal_like_score("asyncio gather is great", "asyncio gather")
    # phrase match adds a bonus + individual term matches
    assert score > 0.0


def test_journal_like_score_no_match():
    score = srv._journal_like_score("completely unrelated text", "zzz_not_found")
    assert score == 0.0


def test_journal_like_score_case_insensitive():
    score_lower = srv._journal_like_score("Python is fun", "python")
    score_upper = srv._journal_like_score("Python is fun", "PYTHON")
    assert score_lower == score_upper


def test_journal_like_score_multiple_terms():
    # "python" appears twice, "fast" once
    score = srv._journal_like_score("python is fast and python is popular", "python fast")
    # phrase "python fast" appears 0 times, individual terms: python=2, fast=1
    assert score >= 3.0


# ---------------------------------------------------------------------------
# handle_kb_sync_status — no dir_path and no env var
# ---------------------------------------------------------------------------


def test_kb_sync_status_no_dir_and_no_env(monkeypatch):
    """When dir_path is absent and LORE_SYNC_DIR is unset, return not_configured."""
    monkeypatch.delenv("LORE_SYNC_DIR", raising=False)
    monkeypatch.delenv("LORE_KB_DIR", raising=False)
    resp = srv.handle_kb_sync_status()
    assert resp["ok"] is False
    assert resp["error"] == "not_configured"


def test_kb_sync_status_empty_string_dir_uses_env(monkeypatch, tmp_path):
    """Empty-string dir_path falls back to LORE_SYNC_DIR env var."""
    monkeypatch.setenv("LORE_SYNC_DIR", str(tmp_path))
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_kb_sync_status(dir_path="")
    assert resp["ok"] is True


def test_kb_sync_status_directory_not_found(monkeypatch):
    resp = srv.handle_kb_sync_status(dir_path="/nonexistent/path/xyz_lore_test")
    assert resp["ok"] is False
    assert resp["error"] == "not_found"


def test_kb_sync_status_empty_dir(monkeypatch, tmp_path):
    """An existing empty directory returns all-zero counts."""
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_kb_sync_status(dir_path=str(tmp_path))
    assert resp["ok"] is True
    assert resp["data"]["total_docs"] == 0
    assert resp["data"]["synced"] == 0
    assert resp["data"]["new"] == 0


def test_kb_sync_status_new_files_detected(monkeypatch, tmp_path):
    """Markdown files not in sync records appear as 'new'."""
    (tmp_path / "notes.md").write_text("# Notes")
    (tmp_path / "guide.md").write_text("# Guide")
    monkeypatch.setattr(srv, "db", _FakeDb(current_row=[]))
    resp = srv.handle_kb_sync_status(dir_path=str(tmp_path))
    assert resp["ok"] is True
    assert resp["data"]["new"] == 2
    assert resp["data"]["total_docs"] == 2
