"""Opt-in bearer-token authentication for Lore's HTTP transport (P1-8).

Applied to the consolidated FastMCP HTTP surface (P1-3):

  * ``lore.server_fastmcp``    (FastMCP HTTP — what production runs)

Design (do NOT break existing deployments):

  * Auth is OPT-IN. It is enforced ONLY when ``LORE_API_KEY`` is set to a
    non-empty value. When unset, the HTTP surfaces behave exactly as before
    (open) so the live Hermes -> Lore connection — which sends no auth header
    today — keeps working.
  * When set, requests must carry ``Authorization: Bearer <key>``; the token is
    compared in constant time (``hmac.compare_digest``). Missing/wrong tokens
    get a ``401`` with JSON body ``{"error": "unauthorized"}``.
  * Health/readiness endpoints are always exempt so liveness probes work even
    when a key is configured.
  * stdio mode is unaffected (no network surface) — only the HTTP paths get the
    check.

It also centralises the CORS configuration and fixes a latent bug: the prior
``allow_origins=["*"]`` combined with ``allow_credentials=True`` is invalid per
the CORS spec and is rejected by browsers. We force ``allow_credentials=False``
whenever origins are wildcard, and allow explicit origins (via
``LORE_CORS_ORIGINS``) to opt back into credentials.
"""

from __future__ import annotations

import hmac
import logging

from starlette.types import ASGIApp, Receive, Scope, Send

from .env_config import get_env

logger = logging.getLogger(__name__)

# Paths that never require auth: liveness/readiness probes and the root ping.
# Matching is exact (after stripping a trailing slash) so a protected route is
# never accidentally exempted by a prefix collision.
_EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/healthz", "/ready", "/readyz", "/"})

# Hosts that are considered local (no insecure-bind warning needed).
_LOCAL_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1", ""})

_UNAUTHORIZED_BODY = b'{"error":"unauthorized"}'


def bearer_token() -> str | None:
    """Return the configured ``LORE_API_KEY`` or ``None`` when auth is off.

    A whitespace-only value is treated as unset so an empty assignment in a
    ``.env`` file does not silently enable auth with a blank key.
    """
    raw = get_env("LORE_API_KEY", None)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def auth_enabled() -> bool:
    """Whether bearer auth should be enforced (i.e. a key is configured)."""
    return bearer_token() is not None


def _is_exempt(path: str) -> bool:
    """Whether ``path`` is an always-open health/readiness endpoint."""
    if path != "/":
        path = path.rstrip("/")
    return path in _EXEMPT_PATHS


def _token_matches(header_value: str | None, expected: str) -> bool:
    """Constant-time check of an ``Authorization: Bearer <key>`` header."""
    if not header_value:
        return False
    parts = header_value.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    # RFC 6750: exactly one space between scheme and token; do NOT strip so
    # "Bearer  secret" (double space) is correctly rejected.
    presented = parts[1]
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


class BearerAuthMiddleware:
    """Pure-ASGI middleware enforcing opt-in bearer auth.

    Implemented at the ASGI level (rather than as a Starlette
    ``BaseHTTPMiddleware``) so it composes cleanly with all three transports,
    including FastMCP's streaming responses, without buffering bodies.

    No-op unless ``LORE_API_KEY`` is set, so the open back-compat path adds only
    a single env read per request and never alters behaviour.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only guard HTTP requests; pass websockets/lifespan straight through.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        expected = bearer_token()
        if expected is None:
            # Auth disabled — behave exactly as before (open).
            await self.app(scope, receive, send)
            return

        if _is_exempt(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        header_value = _extract_authorization(scope)
        if _token_matches(header_value, expected):
            await self.app(scope, receive, send)
            return

        await _send_unauthorized(send)


def _extract_authorization(scope: Scope) -> str | None:
    """Pull the ``Authorization`` header value out of an ASGI scope."""
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            try:
                return value.decode("latin-1")
            except UnicodeDecodeError:
                return None
    return None


async def _send_unauthorized(send: Send) -> None:
    """Emit a ``401`` JSON response without touching the wrapped app.

    Includes ``WWW-Authenticate: Bearer realm="lore"`` per RFC 6750 §3 so
    clients know the required authentication scheme.
    """
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode("ascii")),
                (b"www-authenticate", b'Bearer realm="lore"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def cors_config() -> dict:
    """Return CORS kwargs for ``CORSMiddleware`` with the credentials bug fixed.

    Origins come from ``LORE_CORS_ORIGINS`` (comma-separated); default ``*``.
    ``allow_credentials`` is forced ``False`` whenever origins are wildcard,
    because ``allow_origins=["*"]`` + ``allow_credentials=True`` is invalid per
    the CORS spec and rejected by browsers. Token auth does not need cookies, so
    disabling credentials for the wildcard case is both correct and sufficient.
    """
    raw = get_env("LORE_CORS_ORIGINS", "*") or "*"
    origins = [o.strip() for o in raw.split(",") if o.strip()] or ["*"]
    wildcard = "*" in origins
    return {
        "allow_origins": origins,
        "allow_credentials": not wildcard,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def warn_if_insecure_bind(host: str | None) -> bool:
    """Warn (and return True) if binding to a non-local host without a key.

    Helps operators notice that an open Lore is now reachable on the LAN. Does
    nothing when a key is configured or when bound to a loopback address.
    """
    if auth_enabled():
        return False
    if (host or "").strip() in _LOCAL_HOSTS:
        return False
    logger.warning(
        "Lore HTTP is binding to non-localhost host %r WITHOUT LORE_API_KEY set: "
        "the server is reachable on your network with NO authentication. "
        "Set LORE_API_KEY to require 'Authorization: Bearer <key>' on requests.",
        host,
    )
    return True
