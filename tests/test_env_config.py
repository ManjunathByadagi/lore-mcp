"""Unit tests for lore.env_config.

env_config.py was at 31.6% (26 stmts). Tests cover:
- load_env_file: reads KEY=VALUE pairs, skips comments/blanks, env takes precedence
- get_env: optional default
- require_env: raises ValueError when missing
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------


def test_load_env_file_sets_variables(tmp_path, monkeypatch):
    """A .env file with KEY=VALUE sets those variables in os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_VAR=hello\nANOTHER_VAR=world\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    monkeypatch.delenv("ANOTHER_VAR", raising=False)

    # Re-import to trigger the module-level load with the patched ENV_FILE.
    import importlib

    import lore.env_config as ec

    # Manually call since we've already imported
    ec.ENV_FILE = env_file
    ec.load_env_file()

    assert os.environ.get("MY_TEST_VAR") == "hello"
    assert os.environ.get("ANOTHER_VAR") == "world"

    # Clean up
    os.environ.pop("MY_TEST_VAR", None)
    os.environ.pop("ANOTHER_VAR", None)


def test_load_env_file_skips_comments_and_blanks(tmp_path):
    """Comment lines (#...) and blank lines are silently ignored."""
    env_file = tmp_path / ".env"
    env_file.write_text("# This is a comment\n\nVALID_KEY=value\n")

    import lore.env_config as ec

    ec.ENV_FILE = env_file
    os.environ.pop("VALID_KEY", None)
    ec.load_env_file()

    assert os.environ.get("VALID_KEY") == "value"
    os.environ.pop("VALID_KEY", None)


def test_load_env_file_env_takes_precedence(tmp_path, monkeypatch):
    """When a variable is already set in os.environ, the .env value must not override it."""
    env_file = tmp_path / ".env"
    env_file.write_text("OVERRIDE_ME=from_file\n")

    import lore.env_config as ec

    monkeypatch.setenv("OVERRIDE_ME", "from_env")
    ec.ENV_FILE = env_file
    ec.load_env_file()

    assert os.environ.get("OVERRIDE_ME") == "from_env"


def test_load_env_file_nonexistent_is_noop(tmp_path):
    """When the .env file does not exist, load_env_file returns without error."""
    import lore.env_config as ec

    ec.ENV_FILE = tmp_path / "nonexistent.env"
    # Should not raise
    ec.load_env_file()


def test_load_env_file_handles_equals_in_value(tmp_path):
    """Values that themselves contain '=' must be preserved whole."""
    env_file = tmp_path / ".env"
    env_file.write_text("TOKEN=abc=def=ghi\n")

    import lore.env_config as ec

    os.environ.pop("TOKEN", None)
    ec.ENV_FILE = env_file
    ec.load_env_file()
    assert os.environ.get("TOKEN") == "abc=def=ghi"
    os.environ.pop("TOKEN", None)


# ---------------------------------------------------------------------------
# get_env
# ---------------------------------------------------------------------------


def test_get_env_returns_value_when_set(monkeypatch):
    from lore.env_config import get_env

    monkeypatch.setenv("LORE_TEST_KEY", "test_value")
    assert get_env("LORE_TEST_KEY") == "test_value"


def test_get_env_returns_default_when_unset(monkeypatch):
    from lore.env_config import get_env

    monkeypatch.delenv("LORE_TEST_KEY", raising=False)
    assert get_env("LORE_TEST_KEY", "my_default") == "my_default"


def test_get_env_returns_none_without_default(monkeypatch):
    from lore.env_config import get_env

    monkeypatch.delenv("LORE_TEST_KEY", raising=False)
    assert get_env("LORE_TEST_KEY") is None


# ---------------------------------------------------------------------------
# require_env
# ---------------------------------------------------------------------------


def test_require_env_returns_value_when_set(monkeypatch):
    from lore.env_config import require_env

    monkeypatch.setenv("LORE_REQUIRED_KEY", "required_value")
    assert require_env("LORE_REQUIRED_KEY") == "required_value"


def test_require_env_raises_when_unset(monkeypatch):
    from lore.env_config import require_env

    monkeypatch.delenv("LORE_MISSING_KEY", raising=False)
    with pytest.raises(ValueError, match="LORE_MISSING_KEY"):
        require_env("LORE_MISSING_KEY")


def test_require_env_error_message_includes_key_name(monkeypatch):
    from lore.env_config import require_env

    monkeypatch.delenv("MY_SECRET_KEY", raising=False)
    with pytest.raises(ValueError) as exc_info:
        require_env("MY_SECRET_KEY")
    assert "MY_SECRET_KEY" in str(exc_info.value)
