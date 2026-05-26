"""Schema/migration tests for the trust_score column (Issue #14).

Covers both backends' migration surface without a live PostgreSQL:

* SQLite — a real ``SqliteClient`` against a tempfile DB. Verifies the column
  exists on a fresh database and that ``_migrate_trust_score`` adds it (with the
  DEFAULT 1.0 picked up by pre-existing rows) on a legacy database created
  before the column existed.
* PostgreSQL — the ``KB_ENTRIES_TRUST_SCORE_DDL`` constant matches
  ``migrations/009_trust_score.sql`` byte-for-byte (sans comments), mirroring
  the parity guarantee enforced for the telemetry migrations.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from lore import db_client


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_sqlite_db_has_trust_score_column():
    """A brand-new SQLite database includes trust_score from the base schema."""
    with tempfile.TemporaryDirectory() as tmp:
        client = db_client.SqliteClient(db_path=str(Path(tmp) / "fresh.db"))
        try:
            conn = client._get_connection()
            assert "trust_score" in _columns(conn, "knowledge_kb_entries")
        finally:
            client.close()


def test_sqlite_trust_score_defaults_to_one_on_insert():
    """Inserting a row without trust_score yields the DEFAULT 1.0."""
    with tempfile.TemporaryDirectory() as tmp:
        client = db_client.SqliteClient(db_path=str(Path(tmp) / "default.db"))
        try:
            conn = client._get_connection()
            conn.execute(
                "INSERT INTO knowledge_kb_entries (kb_id, topic, title, content) "
                "VALUES (?, ?, ?, ?)",
                ("kb_x", "t", "T", "c"),
            )
            conn.commit()
            row = conn.execute(
                "SELECT trust_score FROM knowledge_kb_entries WHERE kb_id = ?", ("kb_x",)
            ).fetchone()
            assert row[0] == 1.0
        finally:
            client.close()


def test_sqlite_migration_adds_trust_score_to_legacy_db():
    """A legacy DB lacking trust_score gets the column added idempotently.

    We hand-build a knowledge_kb_entries table *without* trust_score (the
    pre-Issue-#14 shape) and seed a row, then open it through SqliteClient.
    _init_schema -> _migrate_trust_score must add the column, and the existing
    row must pick up the DEFAULT 1.0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "legacy.db")

        # 1) Create the old-shape table directly (no trust_score column) and seed.
        raw = sqlite3.connect(db_path)
        raw.execute(
            "CREATE TABLE knowledge_kb_entries ("
            "    kb_id TEXT PRIMARY KEY,"
            "    topic TEXT NOT NULL,"
            "    title TEXT NOT NULL,"
            "    content TEXT NOT NULL,"
            "    verified INTEGER"
            ")"
        )
        raw.execute(
            "INSERT INTO knowledge_kb_entries (kb_id, topic, title, content) VALUES (?, ?, ?, ?)",
            ("kb_legacy", "t", "Legacy", "old content"),
        )
        raw.commit()
        assert "trust_score" not in _columns(raw, "knowledge_kb_entries")
        raw.close()

        # 2) Open through SqliteClient — _migrate_trust_score runs at init.
        client = db_client.SqliteClient(db_path=db_path)
        try:
            conn = client._get_connection()
            assert "trust_score" in _columns(conn, "knowledge_kb_entries")
            # Pre-existing row picks up the DEFAULT 1.0 (fully trusted).
            row = conn.execute(
                "SELECT trust_score FROM knowledge_kb_entries WHERE kb_id = ?",
                ("kb_legacy",),
            ).fetchone()
            assert row[0] == 1.0
        finally:
            client.close()


def test_sqlite_migration_is_idempotent():
    """Calling _migrate_trust_score twice is a harmless no-op the second time."""
    with tempfile.TemporaryDirectory() as tmp:
        client = db_client.SqliteClient(db_path=str(Path(tmp) / "idem.db"))
        try:
            conn = client._get_connection()
            # Already migrated during init; a second call must not raise.
            client._migrate_trust_score(conn)
            assert "trust_score" in _columns(conn, "knowledge_kb_entries")
        finally:
            client.close()


def _normalize_sql(text: str) -> str:
    """Collapse whitespace and drop a trailing semicolon for byte-comparison."""
    return " ".join(text.split()).rstrip(";").strip()


def test_pg_trust_score_ddl_matches_migration_009():
    """migration 009 SQL (its ALTER statement) must match the code constant."""
    migration = Path(__file__).resolve().parents[1] / "migrations" / "009_trust_score.sql"
    text = migration.read_text()
    sql_start = text.index("ALTER TABLE")
    # The migration's first statement ends at the first semicolon; trailing
    # comment lines that follow are documentation, not DDL.
    sql_end = text.index(";", sql_start) + 1
    migration_ddl = _normalize_sql(text[sql_start:sql_end])
    assert migration_ddl == _normalize_sql(db_client.KB_ENTRIES_TRUST_SCORE_DDL)
    # The constant is the exact single ALTER ... ADD COLUMN statement.
    assert db_client.KB_ENTRIES_TRUST_SCORE_DDL == (
        "ALTER TABLE knowledge.kb_entries ADD COLUMN IF NOT EXISTS trust_score REAL DEFAULT 1.0;"
    )
