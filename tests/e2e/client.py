"""Thin synchronous HTTP client for the Lore MCP JSON-RPC endpoint.

The Lore server exposes a JSON-RPC 2.0 interface at ``POST /mcp``.  Each
response carries the MCP-protocol envelope::

    {
        "jsonrpc": "2.0",
        "id": <int>,
        "result": {
            "content": [
                {"type": "text", "text": "<json-encoded payload>"}
            ]
        }
    }

Inside that TextContent wrapper Lore adds its own business envelope::

    {"ok": True, "data": {...}, "message": "...", "error": null}

:class:`LoreClient` transparently unwraps both layers so callers receive the
inner ``data`` dict directly.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class LoreClientError(Exception):
    """Raised when the MCP server returns a JSON-RPC error."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.data = data
        super().__init__(f"MCP error {code}: {message}")


class LoreClient:
    """Synchronous MCP HTTP client for the Lore knowledge-base server.

    Args:
        url: Base URL of the Lore instance, e.g. ``http://localhost:5555``.
              A trailing slash is stripped automatically.
        timeout: HTTP request timeout in seconds (default 30 s).

    Example::

        client = LoreClient("http://lore-staging:5555")
        result = client.kb_add(topic="ops", title="DNS fix", content="…")
        print(result["kb_id"])
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._id = 0

    # ------------------------------------------------------------------
    # Low-level JSON-RPC helpers
    # ------------------------------------------------------------------

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and return the raw response dict."""
        self._id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
            "params": params or {},
        }
        resp = self._client.post(f"{self.url}/mcp", json=payload)
        resp.raise_for_status()

        # Server returns SSE-framed responses: "event: message\ndata: {...}\n\n"
        # Extract the JSON payload from the "data: " line.
        raw = resp.text
        body_str: str | None = None
        for line in raw.splitlines():
            if line.startswith("data: "):
                body_str = line[len("data: ") :]
                break
        if body_str is None:
            # Server returned bare JSON (alternative endpoint or future behaviour change)
            body_str = raw

        body: dict[str, Any] = json.loads(body_str)

        # Surface JSON-RPC-level errors
        if "error" in body and body["error"] is not None:
            err = body["error"]
            raise LoreClientError(
                code=err.get("code", -1),
                message=err.get("message", "unknown error"),
                data=err.get("data"),
            )

        return body

    def _unwrap(self, rpc_response: dict[str, Any]) -> dict[str, Any]:
        """Unwrap the MCP TextContent envelope.

        Lore wraps every tool result as::

            result.content[0].text  ->  JSON-encoded inner payload

        This method extracts and JSON-parses that inner payload.  If the
        response does not follow the MCP envelope shape the raw ``result``
        dict is returned instead, so callers are insulated from minor
        protocol variations.
        """
        result = rpc_response.get("result", {})
        content = result.get("content")
        if content and isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                raw_text = first.get("text", "")
                try:
                    return json.loads(raw_text)  # type: ignore[return-value]
                except json.JSONDecodeError:
                    # Return the raw string wrapped in a dict so tests can still inspect
                    return {"text": raw_text}
        # Fallback: return the result dict as-is
        return result  # type: ignore[return-value]

    def _unwrap_tool_result(self, jsonrpc_result: dict[str, Any]) -> dict[str, Any]:
        """Unwrap Lore's business data envelope.

        Every Lore tool response has the shape::

            {"ok": True, "data": {…}, "message": "…", "error": null}

        This method extracts ``data`` on success and raises
        :class:`LoreClientError` when ``ok`` is ``False``.
        """
        if not isinstance(jsonrpc_result, dict):
            return jsonrpc_result  # type: ignore[return-value]
        if jsonrpc_result.get("ok") is False:
            raise LoreClientError(
                code=-1,
                message=(
                    f"Tool failed: {jsonrpc_result.get('error')} — {jsonrpc_result.get('message')}"
                ),
            )
        if "data" in jsonrpc_result and isinstance(jsonrpc_result["data"], dict):
            return jsonrpc_result["data"]
        return jsonrpc_result

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a named MCP tool and return the unwrapped inner payload.

        Args:
            name: Tool name (e.g. ``"kb_add"``).
            arguments: Keyword arguments forwarded to the tool.

        Returns:
            The inner payload dict after unwrapping both the MCP TextContent
            envelope and Lore's business data envelope.
        """
        rpc = self._call("tools/call", {"name": name, "arguments": arguments or {}})
        unwrapped = self._unwrap(rpc)
        return self._unwrap_tool_result(unwrapped)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tools advertised by the server.

        Returns:
            List of tool descriptor dicts from ``tools/list``.
        """
        rpc = self._call("tools/list")
        return rpc.get("result", {}).get("tools", [])

    def ping(self) -> bool:
        """Return ``True`` if the server responds to ``tools/list`` without error."""
        try:
            self.list_tools()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Knowledge-base convenience wrappers
    # ------------------------------------------------------------------

    def kb_add(
        self,
        *,
        topic: str,
        title: str,
        content: str,
        **kw: Any,
    ) -> dict[str, Any]:
        """Add a new knowledge-base entry.

        Args:
            topic: Logical topic / namespace for the entry.
            title: Short human-readable title.
            content: Full text content of the entry.
            **kw: Additional arguments forwarded verbatim (e.g. ``tags``).

        Returns:
            Unwrapped tool response dict with ``kb_id``, ``topic``,
            ``embedded``, etc.
        """
        return self.tool("kb_add", {"topic": topic, "title": title, "content": content, **kw})

    def kb_search(self, query: str, **kw: Any) -> dict[str, Any]:
        """Search the knowledge base.

        Args:
            query: Free-text search query.
            **kw: Optional overrides: ``search_mode`` (``"fts"``, ``"semantic"``,
                  ``"hybrid"``), ``top_k``, ``topic``, etc.

        Returns:
            Unwrapped search result dict with ``results`` list.
        """
        return self.tool("kb_search", {"query": query, **kw})

    def kb_get(self, kb_id: str) -> dict[str, Any]:
        """Retrieve a single knowledge-base entry by ID.

        Args:
            kb_id: The entry ID (e.g. ``"kb_abc123"``).

        Returns:
            Unwrapped entry dict.
        """
        return self.tool("kb_get", {"kb_id": kb_id})

    def kb_update(self, kb_id: str, **kw: Any) -> dict[str, Any]:
        """Update fields of an existing knowledge-base entry.

        Args:
            kb_id: The entry ID (e.g. ``"kb_abc123"``).
            **kw: Fields to update: ``content``, ``tags``, ``topic``,
                  ``verified``, ``metadata``.

        Returns:
            Unwrapped update-result dict.
        """
        return self.tool("kb_update", {"kb_id": kb_id, **kw})

    def kb_delete(self, kb_id: str, confirm: bool = True) -> dict[str, Any]:
        """Delete a knowledge-base entry.

        Args:
            kb_id: The entry ID (e.g. ``"kb_abc123"``).
            confirm: Safety flag — must be ``True`` (default) to actually delete.

        Returns:
            Unwrapped deletion-confirmation dict.
        """
        return self.tool("kb_delete", {"kb_id": kb_id, "confirm": confirm})

    def kb_list(self, **kw: Any) -> dict[str, Any]:
        """List knowledge-base entries.

        Args:
            **kw: Optional filters such as ``topic``.

        Returns:
            Unwrapped list-result dict with ``entries`` list.
        """
        return self.tool("kb_list", {**kw})

    def kb_embedding_status(self) -> dict[str, Any]:
        """Return the current embedding-coverage status report.

        Returns:
            Dict with ``total_entries``, ``embedded``, ``missing``,
            ``coverage_pct``, etc.
        """
        return self.tool("kb_embedding_status", {})

    def kb_backfill_embeddings(self, **kw: Any) -> dict[str, Any]:
        """Trigger a backfill run to embed any un-embedded entries.

        Args:
            **kw: Optional params such as ``batch_size``, ``dry_run``.

        Returns:
            Unwrapped backfill-result dict.
        """
        return self.tool("kb_backfill_embeddings", {**kw})

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._client.close()

    def __enter__(self) -> LoreClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
