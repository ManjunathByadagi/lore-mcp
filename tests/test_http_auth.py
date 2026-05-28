"""Unit tests for opt-in LORE_API_KEY bearer auth on HTTP transports (P1-8).

Auth is OPT-IN: it is enforced only when ``LORE_API_KEY`` is set. When the key
is unset the HTTP surfaces behave exactly as before (open), so the live
Hermes -> Lore connection (which sends no auth header) keeps working.

These tests cover the shared helpers in ``lore.http_auth`` and verify the
middleware wired into each of the three HTTP transports:

  1. lore.server          (--host/--port HTTP mode, delegates to the SSE wrapper)
  2. lore.server_fastmcp   (FastMCP HTTP — what production runs)
  3. lore.mcp_http_wrapper_sse (the SSE wrapper)

Behaviour matrix:
  * key UNSET  -> request without header succeeds (back-compat preserved)
  * key SET    -> missing/wrong bearer -> 401; correct bearer -> 200
  * health endpoints reachable without auth even when the key is set
  * CORS: allow_credentials is False whenever origins are wildcard "*"
"""

from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from lore import http_auth

API_KEY = "s3cret-key"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _build_app() -> Starlette:
    """A minimal Starlette app mirroring the real transports' route shape.

    Exposes a protected ``/mcp`` route plus exempt ``/health`` and ``/`` routes,
    wrapped in the shared BearerAuthMiddleware exactly as the real servers do.
    """

    async def protected(request):  # noqa: ANN001
        return JSONResponse({"ok": True})

    async def health(request):  # noqa: ANN001
        return JSONResponse({"status": "healthy"})

    routes = [
        Route("/mcp", protected, methods=["POST"]),
        Route("/jsonrpc", protected, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/", health, methods=["GET"]),
    ]
    middleware = [Middleware(http_auth.BearerAuthMiddleware)]
    return Starlette(routes=routes, middleware=middleware)


@pytest.fixture
def client_no_key(monkeypatch):
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    return TestClient(_build_app())


@pytest.fixture
def client_with_key(monkeypatch):
    monkeypatch.setenv("LORE_API_KEY", API_KEY)
    return TestClient(_build_app())


# ---------------------------------------------------------------------------
# bearer_token() helper
# ---------------------------------------------------------------------------


def test_bearer_token_none_when_unset(monkeypatch):
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    assert http_auth.bearer_token() is None


def test_bearer_token_none_when_blank(monkeypatch):
    """A whitespace-only key is treated as unset (open mode)."""
    monkeypatch.setenv("LORE_API_KEY", "   ")
    assert http_auth.bearer_token() is None


def test_bearer_token_value_when_set(monkeypatch):
    monkeypatch.setenv("LORE_API_KEY", API_KEY)
    assert http_auth.bearer_token() == API_KEY


# ---------------------------------------------------------------------------
# Back-compat: key UNSET -> open (Hermes -> Lore must keep working)
# ---------------------------------------------------------------------------


def test_unset_key_allows_request_without_header(client_no_key):
    resp = client_no_key.post("/mcp", json={"jsonrpc": "2.0", "id": 1})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_unset_key_ignores_any_header(client_no_key):
    """A stray Authorization header must not break the open path."""
    resp = client_no_key.post("/mcp", json={}, headers={"Authorization": "Bearer whatever"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# key SET -> require Authorization: Bearer <key>
# ---------------------------------------------------------------------------


def test_set_key_rejects_missing_header(client_with_key):
    resp = client_with_key.post("/mcp", json={})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_set_key_rejects_wrong_token(client_with_key):
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_set_key_rejects_malformed_scheme(client_with_key):
    """Token without the ``Bearer`` scheme is rejected."""
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": API_KEY})
    assert resp.status_code == 401


def test_set_key_accepts_correct_token(client_with_key):
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": f"Bearer {API_KEY}"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_set_key_bearer_scheme_case_insensitive(client_with_key):
    """The ``Bearer`` scheme keyword is matched case-insensitively per RFC 6750."""
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": f"bearer {API_KEY}"})
    assert resp.status_code == 200


def test_set_key_protects_jsonrpc_route(client_with_key):
    resp = client_with_key.post("/jsonrpc", json={})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Health endpoints are exempt even when the key is set
# ---------------------------------------------------------------------------


def test_health_exempt_when_key_set(client_with_key):
    resp = client_with_key.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_root_exempt_when_key_set(client_with_key):
    resp = client_with_key.get("/")
    assert resp.status_code == 200


def test_health_works_without_key(client_no_key):
    resp = client_no_key.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CORS config: fix the allow_origins=["*"] + allow_credentials=True bug
# ---------------------------------------------------------------------------


def test_cors_wildcard_disables_credentials(monkeypatch):
    monkeypatch.delenv("LORE_CORS_ORIGINS", raising=False)
    cfg = http_auth.cors_config()
    assert cfg["allow_origins"] == ["*"]
    assert cfg["allow_credentials"] is False


def test_cors_explicit_origins_allow_credentials(monkeypatch):
    monkeypatch.setenv("LORE_CORS_ORIGINS", "https://a.example, https://b.example")
    cfg = http_auth.cors_config()
    assert cfg["allow_origins"] == ["https://a.example", "https://b.example"]
    assert cfg["allow_credentials"] is True


def test_cors_explicit_wildcard_still_disables_credentials(monkeypatch):
    """Even an explicit ``*`` must not be paired with credentials (CORS spec)."""
    monkeypatch.setenv("LORE_CORS_ORIGINS", "*")
    cfg = http_auth.cors_config()
    assert cfg["allow_origins"] == ["*"]
    assert cfg["allow_credentials"] is False


# ---------------------------------------------------------------------------
# Non-localhost bind warning when no key is set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.21", "10.0.0.5"])
def test_warns_on_non_localhost_bind_without_key(monkeypatch, caplog, host):
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        warned = http_auth.warn_if_insecure_bind(host)
    assert warned is True
    assert any("LORE_API_KEY" in r.message for r in caplog.records)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_no_warning_on_localhost_bind(monkeypatch, caplog, host):
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        warned = http_auth.warn_if_insecure_bind(host)
    assert warned is False


def test_no_warning_when_key_set_even_on_public_bind(monkeypatch, caplog):
    monkeypatch.setenv("LORE_API_KEY", API_KEY)
    with caplog.at_level("WARNING"):
        warned = http_auth.warn_if_insecure_bind("0.0.0.0")
    assert warned is False


# ---------------------------------------------------------------------------
# Integration: the real FastMCP app (production transport) honours the key
# ---------------------------------------------------------------------------


def test_fastmcp_app_enforces_auth_when_key_set(monkeypatch):
    monkeypatch.setenv("LORE_API_KEY", API_KEY)
    from lore import server_fastmcp

    app = server_fastmcp.build_http_app()
    with TestClient(app) as client:
        # Health is exempt.
        assert client.get("/health").status_code == 200
        # /mcp without a token is rejected.
        unauth = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert unauth.status_code == 401
        # /mcp with the correct token succeeds.
        ok = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert ok.status_code == 200
        assert ok.json()["result"]["tools"]


def test_fastmcp_app_open_when_key_unset(monkeypatch):
    """Back-compat: no key -> /mcp reachable with no header (Hermes path)."""
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    from lore import server_fastmcp

    app = server_fastmcp.build_http_app()
    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["result"]["tools"]


# ---------------------------------------------------------------------------
# Integration: the SSE wrapper app honours the key
# ---------------------------------------------------------------------------


def test_sse_wrapper_enforces_auth_when_key_set(monkeypatch):
    monkeypatch.setenv("LORE_API_KEY", API_KEY)
    from lore import mcp_http_wrapper_sse, server

    app = mcp_http_wrapper_sse.create_app(server.app, "lore.server")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/mcp", json={}).status_code == 401
    ok = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialized"},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert ok.status_code == 200


def test_sse_wrapper_open_when_key_unset(monkeypatch):
    monkeypatch.delenv("LORE_API_KEY", raising=False)
    from lore import mcp_http_wrapper_sse, server

    app = mcp_http_wrapper_sse.create_app(server.app, "lore.server")
    client = TestClient(app)
    resp = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialized"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fix 2: strict RFC 6750 bearer parse — double-space must be rejected
# ---------------------------------------------------------------------------


def test_correct_bearer_returns_200(client_with_key):
    """Normal ``Bearer <key>`` (single space) must still be accepted."""
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": f"Bearer {API_KEY}"})
    assert resp.status_code == 200


def test_double_space_bearer_returns_401(client_with_key):
    """``Bearer  <key>`` (double space) must be rejected per RFC 6750."""
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": f"Bearer  {API_KEY}"})
    assert resp.status_code == 401


def test_lowercase_bearer_scheme_accepted(client_with_key):
    """Case-insensitive scheme matching (``bearer``) is intentional and must remain."""
    resp = client_with_key.post("/mcp", json={}, headers={"Authorization": f"bearer {API_KEY}"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Fix 3: WWW-Authenticate header on 401 responses (RFC 6750 §3)
# ---------------------------------------------------------------------------


def test_401_includes_www_authenticate_header(client_with_key):
    """Every 401 must include ``WWW-Authenticate: Bearer realm="lore"`` per RFC 6750 §3."""
    resp = client_with_key.post("/mcp", json={})
    assert resp.status_code == 401
    www_auth = resp.headers.get("www-authenticate", "")
    assert "Bearer" in www_auth
    assert 'realm="lore"' in www_auth


# ---------------------------------------------------------------------------
# Fix 4: exempt-path lookalike paths must NOT bypass auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/healthILY", "/health/extra", "/healthz/sub", "/healthx"],
)
def test_exempt_path_lookalikes_are_protected(client_with_key, path):
    """Paths that look like exempt paths but are not exact matches must require auth."""
    resp = client_with_key.get(path)
    assert resp.status_code == 401
