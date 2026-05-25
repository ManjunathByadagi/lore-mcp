"""Standard response envelope and error codes."""

import os
from typing import Any


def current_env() -> str:
    """Return the active environment label for tool responses.

    Sourced from ``LORE_ENV``; defaults to ``"production"`` when unset so that
    an unconfigured deployment is treated as production (fail-safe). This value
    is stamped into every tool response so callers can immediately see which
    environment they are operating against.
    """
    return os.getenv("LORE_ENV", "production").strip().lower()


class ResponseEnvelope:
    """Standard response envelope for all tools."""

    @staticmethod
    def success(message: str, data: Any = None) -> dict:
        """Create a success response."""
        return {
            "ok": True,
            "error": None,
            "message": message,
            "env": current_env(),
            "data": data or {},
        }

    @staticmethod
    def ok(message: str, data: Any = None) -> dict:
        """Alias for success() - create a success response."""
        return ResponseEnvelope.success(message, data)

    @staticmethod
    def error(code: str, message: str, data: Any = None) -> dict:
        """Create an error response."""
        return {
            "ok": False,
            "error": code,
            "message": message,
            "env": current_env(),
            "data": data or {},
        }


class ErrorCodes:
    """Common error codes across all servers."""

    UNEXPECTED_EXCEPTION = "unexpected_exception"
    INVALID_ARGUMENT = "invalid_argument"
    INVALID_INPUT = "invalid_input"  # Alias for INVALID_ARGUMENT
    NOT_FOUND = "not_found"
    IO_ERROR = "io_error"
    NONZERO_EXIT = "nonzero_exit"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"
    POLICY_VIOLATION = "policy_violation"
    UNIT_NOT_ALLOWED = "unit_not_allowed"
    INTERNAL_ERROR = "internal_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    PRODUCTION_GUARD = "production_guard"
