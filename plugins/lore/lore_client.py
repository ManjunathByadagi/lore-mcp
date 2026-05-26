"""HTTP client for the Lore KB-MCP server + dedup logic.

Lore (http://192.168.1.21:5555) is MCP-only — it exposes no REST API
(``/``, ``/docs``, ``/openapi.json`` all 404; only ``/health`` is 200 and
``/mcp`` speaks JSON-RPC over StreamableHTTP). So this client POSTs
JSON-RPC ``tools/call`` requests to ``/mcp`` directly via httpx, handling
both plain-JSON and SSE-framed (``data: {...}``) responses, and unwraps
Lore's ``{ok, error, message, env, data}`` envelope.

IMPORTANT — parameter and scoring facts verified live against Lore
(2026-05-25):
  * kb_search params are ``query``, ``topic``, ``top_k`` (NOT ``limit``)
    and ``search_mode`` in {"fts","semantic","hybrid"} (NOT ``mode``).
  * Result fields: kb_id, title, topic, tags, author, source_type,
    verified, score, rrf_score. ``content`` is absent in list results.
  * In hybrid mode ``rrf_score`` is present and HIGHER = more similar.
    In semantic mode score/rrf_score are None. So dedup uses hybrid +
    ``rrf_score >= DEDUP_THRESHOLD``.

This module imports nothing from hermes, so it is independently testable.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Calibrated default. rrf_score for hybrid search lands in (0, ~0.3].
# Live observation: a near-duplicate top hit fuses to rrf_score ~0.15-0.18,
# while unrelated content scores <~0.05. 0.10 cleanly separates the two
# bands with margin on both sides. Overridable via config (dedup_threshold).
DEFAULT_LORE_URL = "http://192.168.1.21:5555"
DEDUP_THRESHOLD = 0.10

_VALID_SEARCH_MODES = ("fts", "semantic", "hybrid")


class LoreClient:
    """Thin MCP-over-HTTP client for Lore's kb_* tools."""

    def __init__(self, base_url: str = DEFAULT_LORE_URL, *, timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._req_id = 0

    # -- transport -----------------------------------------------------------

    def is_available(self) -> bool:
        """Cheap reachability check against Lore's /health endpoint.

        Network call kept here (not in the provider's is_available, which
        the ABC says must avoid network I/O) — used opportunistically by
        write paths, not during agent init.
        """
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.debug("Lore /health check failed: %s", exc)
            return False

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON-RPC tools/call and return the unwrapped ``data`` dict.

        Raises RuntimeError on transport/protocol/tool errors so callers
        can decide whether to degrade.
        """
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        resp = httpx.post(
            f"{self.base_url}/mcp", json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        envelope = self._parse_mcp_body(resp.text)

        if "error" in envelope and envelope["error"] is not None:
            raise RuntimeError(f"Lore MCP error: {envelope['error']}")

        result = envelope.get("result", {})
        inner = self._extract_structured(result)
        if inner is None:
            raise RuntimeError("Lore returned no structured content")

        if inner.get("ok") is False:
            raise RuntimeError(f"Lore tool error: {inner.get('error')}")
        # Unwrap the {ok, error, message, env, data} envelope.
        return inner.get("data", inner)

    @staticmethod
    def _parse_mcp_body(text: str) -> dict[str, Any]:
        """Parse a StreamableHTTP body: plain JSON or SSE ``data:`` frame."""
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise RuntimeError("Unrecognized MCP response body")

    @staticmethod
    def _extract_structured(result: dict[str, Any]) -> dict[str, Any] | None:
        """Pull the structured tool result out of an MCP result block."""
        sc = result.get("structuredContent")
        if isinstance(sc, dict):
            return sc
        for block in result.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except (ValueError, KeyError):
                    continue
        return None

    # -- kb_* tools ----------------------------------------------------------

    def kb_search(
        self,
        query: str,
        *,
        search_mode: str = "hybrid",
        topic: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if search_mode not in _VALID_SEARCH_MODES:
            search_mode = "hybrid"
        args: dict[str, Any] = {
            "query": query,
            "search_mode": search_mode,
            "top_k": top_k,
        }
        if topic is not None:
            args["topic"] = topic
        data = self._call_tool("kb_search", args)
        results = data.get("results", [])
        return results if isinstance(results, list) else []

    def kb_add(
        self,
        *,
        topic: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        author: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"topic": topic, "title": title, "content": content}
        if tags is not None:
            args["tags"] = tags
        if author is not None:
            args["author"] = author
        return self._call_tool("kb_add", args)

    def kb_update(
        self,
        kb_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {"kb_id": kb_id}
        if content is not None:
            args["content"] = content
        if title is not None:
            args["title"] = title
        if tags is not None:
            args["tags"] = tags
        return self._call_tool("kb_update", args)

    def kb_get(self, kb_id: str) -> dict[str, Any]:
        return self._call_tool("kb_get", {"kb_id": kb_id})


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def add_or_update(
    client: Any,
    *,
    topic: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    author: str | None = None,
    threshold: float = DEDUP_THRESHOLD,
) -> dict[str, Any]:
    """Add ``content`` to Lore, or update a near-duplicate instead.

    Probes Lore with a hybrid search for the content. If the top hit's
    ``rrf_score`` is at or above ``threshold`` (higher = more similar in
    hybrid mode), the existing entry is updated rather than creating a
    near-duplicate. Degrades gracefully: if Lore is unreachable, returns
    {"action": "skipped"} without raising.

    Returns a dict with ``action`` in {"added", "updated", "skipped"} and,
    for added/updated, the ``kb_id``.
    """
    if not client.is_available():
        return {"action": "skipped", "reason": "lore_unavailable"}

    try:
        hits = client.kb_search(content, search_mode="hybrid", topic=topic, top_k=3)
    except Exception as exc:  # noqa: BLE001 - dedup probe is best-effort
        logger.debug("Lore dedup probe failed, falling back to add: %s", exc)
        hits = []

    top = hits[0] if hits else None
    rrf = top.get("rrf_score") if isinstance(top, dict) else None

    if top is not None and isinstance(rrf, (int, float)) and rrf >= threshold:
        kb_id = top.get("kb_id")
        client.kb_update(kb_id, content=content, title=title, tags=tags)
        return {"action": "updated", "kb_id": kb_id}

    resp = client.kb_add(topic=topic, title=title, content=content, tags=tags, author=author)
    return {"action": "added", "kb_id": resp.get("kb_id")}
