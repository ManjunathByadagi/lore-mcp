"""lore — Lore memory provider plugin for Nous Hermes.

Backs the Hermes agent (hermes-agent 0.14.0) with Lore (a KB-MCP server at
http://192.168.1.21:5555) as its persistent memory backend. Forked in
structure from the bundled Holographic provider, but talks to a remote
KB-MCP over HTTP instead of a local SQLite store.

Behavior:
  * prefetch(query)      — hybrid kb_search, format top hits into a
                           fence-wrapped recall block for the system prompt.
  * sync_turn(...)       — capture raw structured turns (no LLM call),
                           stripping any injected memory between fence
                           markers so recalled context is never re-stored.
  * on_session_end(...)  — flush captured turns to Lore under topic
                           'hermes-conversations' (the hermes-scheduler job
                           later summarizes them; the provider never calls
                           an LLM itself).
  * lore_remember tool   — explicit user-triggered storage with dedup.

Config (config.yaml plugins.lore, or plugins/lore/config.json):
  recall_mode (default "hybrid"), write_frequency (default "turn"),
  dedup_threshold (default calibrated float), lore_url.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

# Hermes interfaces. On CT 133 these resolve to the real hermes-agent
# modules; in local tests they resolve to tests/_hermes_stubs/ (wired by
# conftest.py). The fallback keeps the module importable even if neither is
# present at definition time.
try:  # pragma: no cover - exercised on CT 133 / via stubs
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
except Exception:  # pragma: no cover - last-resort shim
    from abc import ABC as MemoryProvider  # type: ignore[assignment]

    def tool_error(message, **extra) -> str:  # type: ignore[misc]
        result = {"error": str(message)}
        if extra:
            result.update(extra)
        return json.dumps(result, ensure_ascii=False)


from .lore_client import (  # noqa: E402 - after hermes import guard
    DEDUP_THRESHOLD,
    DEFAULT_LORE_URL,
    LoreClient,
    add_or_update,
)

logger = logging.getLogger(__name__)

# Context-fence markers. prefetch() wraps recalled memory in these so
# sync_turn() can strip it before persisting — recalled memory is never
# re-stored as a new "fact". HTML-comment style survives prompt assembly.
MEMORY_FENCE_START = "<!-- MEMORY_INJECT_START -->"
MEMORY_FENCE_END = "<!-- MEMORY_INJECT_END -->"

CONVERSATIONS_TOPIC = "hermes-conversations"

_FENCE_RE = re.compile(
    re.escape(MEMORY_FENCE_START) + r".*?" + re.escape(MEMORY_FENCE_END),
    re.DOTALL,
)


def strip_memory_fence(text: str) -> str:
    """Remove any fenced memory-injection block(s) from ``text``."""
    if not text or MEMORY_FENCE_START not in text:
        return text
    return _FENCE_RE.sub("", text)


# ---------------------------------------------------------------------------
# Tool schema (explicit user-triggered storage)
# ---------------------------------------------------------------------------

LORE_REMEMBER_SCHEMA = {
    "name": "lore_remember",
    "description": (
        "Store a durable memory in Lore (persistent knowledge base). Use "
        "when the user explicitly asks you to remember something, or for a "
        "fact they would expect recalled in a future session. Dedupes "
        "near-identical entries automatically (updates instead of "
        "duplicating)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or memory to store.",
            },
            "title": {
                "type": "string",
                "description": "Short title for the memory (optional).",
            },
            "topic": {
                "type": "string",
                "description": (
                    "Topic bucket (optional; defaults to a general hermes-memory topic)."
                ),
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags.",
            },
        },
        "required": ["content"],
    },
}

DEFAULT_MEMORY_TOPIC = "hermes-memory"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_plugin_config() -> dict:
    """Load plugins/lore/config.json from HERMES_HOME if present."""
    try:
        from hermes_constants import get_hermes_home

        cfg_file = get_hermes_home() / "plugins" / "lore" / "config.json"
        if cfg_file.exists():
            return json.loads(cfg_file.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - missing config is fine
        logger.debug("Lore config load skipped: %s", exc)
    return {}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LoreMemoryProvider(MemoryProvider):
    """Memory provider backed by the Lore KB-MCP server."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._lore_url = self._config.get("lore_url", DEFAULT_LORE_URL)
        self._recall_mode = self._config.get("recall_mode", "hybrid")
        self._write_frequency = self._config.get("write_frequency", "turn")
        try:
            self._dedup_threshold = float(self._config.get("dedup_threshold", DEDUP_THRESHOLD))
        except (TypeError, ValueError):
            self._dedup_threshold = DEDUP_THRESHOLD
        self._client: LoreClient | None = None
        self._session_id: str = ""
        self._captured_turns: list[dict[str, str]] = []

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "lore"

    def is_available(self) -> bool:
        # Per the ABC, this must not make network calls — just confirm we
        # have a URL configured and httpx importable. Reachability is
        # checked at write/read time via LoreClient.is_available().
        return bool(self._lore_url)

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or self._session_id
        self._client = LoreClient(self._lore_url)

    def shutdown(self) -> None:
        # Best-effort flush of any unpersisted turns on clean exit.
        if self._captured_turns:
            try:
                self._persist_turns()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Lore shutdown flush failed: %s", exc)
        self._client = None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        # Flush the current session's turns before switching so they land
        # under the right session record.
        if self._captured_turns:
            try:
                self._persist_turns()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Lore session-switch flush failed: %s", exc)
        self._captured_turns = []
        self._session_id = new_session_id

    # -- recall --------------------------------------------------------------

    def system_prompt_block(self) -> str:
        return (
            "# Lore Memory\n"
            "Active. Persistent recall is backed by Lore (knowledge base). "
            "Relevant memories are injected automatically before each turn. "
            "Use lore_remember to explicitly store something durable."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or self._client is None:
            return ""
        if not self._client.is_available():
            return ""
        try:
            results = self._client.kb_search(
                query, search_mode=self._recall_mode, topic=None, top_k=5
            )
        except Exception as exc:  # noqa: BLE001 - recall is best-effort
            logger.debug("Lore prefetch failed: %s", exc)
            return ""
        if not results:
            return ""
        lines = []
        for r in results:
            title = r.get("title") or r.get("kb_id", "")
            topic = r.get("topic", "")
            prefix = f"[{topic}] " if topic else ""
            entry_text = f"**{prefix}{title}**"
            # kb_search list results omit content — fetch the full entry so
            # the model sees actual stored facts, not just titles.
            content = r.get("content", "")
            if not content:
                kb_id = r.get("kb_id", "")
                if kb_id:
                    # Use a short-lived client with a 2s timeout so N kb_get
                    # calls in prefetch cannot block longer than 2s each.
                    try:
                        fast_client = type(self._client)(self._lore_url, timeout=2.0)
                        full = fast_client.kb_get(kb_id)
                        content = (full.get("content") or "").strip()
                    except Exception as exc:  # noqa: BLE001 - best-effort
                        logger.warning("prefetch kb_get(%s) failed: %s", kb_id, exc)
                        content = ""
                else:
                    logger.debug("prefetch: kb_id missing for result %s", r)
            if content:
                truncated = content[:400] + ("…" if len(content) > 400 else "")
                entry_text += f"\n{truncated}"
            lines.append(entry_text)
        if lines:
            body = "## Recalled from Lore\n\n" + "\n\n".join(lines)
        else:
            body = "## Recalled from Lore\n\n(No relevant entries found)"
        return f"{MEMORY_FENCE_START}\n{body}\n{MEMORY_FENCE_END}"

    # -- write ---------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Capture raw turn for persistence. No LLM call.
        # Strip injected memory so recalled context is never re-stored.
        user_clean = strip_memory_fence(user_content or "").strip()
        assistant_clean = (assistant_content or "").strip()
        if not user_clean and not assistant_clean:
            return
        self._captured_turns.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "session_id": session_id or self._session_id,
                "user": user_clean,
                "assistant": assistant_clean,
            }
        )
        # write_frequency "turn": persist immediately after each turn.
        # "session" (or any other value): defer until on_session_end. Never
        # raise into the agent — persistence is best-effort.
        if self._write_frequency == "turn":
            try:
                self._persist_turns()
            except Exception as exc:  # noqa: BLE001 - never raise into the agent
                logger.debug("Lore per-turn flush failed: %s", exc)
            finally:
                self._captured_turns = []

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if not self._captured_turns:
            return
        try:
            self._persist_turns()
        except Exception as exc:  # noqa: BLE001 - never raise into the agent
            logger.debug("Lore on_session_end flush failed: %s", exc)
        finally:
            self._captured_turns = []

    def _persist_turns(self) -> None:
        """Store captured turns to Lore as one structured conversation entry."""
        if not self._captured_turns or self._client is None:
            return
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        title = f"Session {self._session_id} — {date}"
        # Render turns as a readable, structured body. The hermes-scheduler
        # job summarizes these later; the provider never calls an LLM.
        parts: list[str] = []
        for i, turn in enumerate(self._captured_turns, 1):
            parts.append(
                f"### Turn {i} ({turn['timestamp']})\n"
                f"USER: {turn['user']}\n"
                f"ASSISTANT: {turn['assistant']}"
            )
        content = "\n\n".join(parts)
        add_or_update(
            self._client,
            topic=CONVERSATIONS_TOPIC,
            title=title,
            content=content,
            tags=["hermes-session", self._session_id] if self._session_id else ["hermes-session"],
            author="hermes",
            threshold=self._dedup_threshold,
        )

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [LORE_REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name == "lore_remember":
            return self._handle_lore_remember(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_lore_remember(self, args: dict[str, Any]) -> str:
        if self._client is None:
            return tool_error("Lore provider not initialized")
        content = args.get("content")
        if not content:
            return tool_error("Missing required argument: content")
        title = args.get("title") or content[:60]
        topic = args.get("topic") or DEFAULT_MEMORY_TOPIC
        tags = args.get("tags")
        try:
            result = add_or_update(
                self._client,
                topic=topic,
                title=title,
                content=content,
                tags=tags,
                author="hermes",
                threshold=self._dedup_threshold,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return tool_error(str(exc))

    # -- config --------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "lore_url",
                "description": "Base URL of the Lore KB-MCP server",
                "default": DEFAULT_LORE_URL,
            },
            {
                "key": "recall_mode",
                "description": "kb_search mode used for prefetch recall",
                "default": "hybrid",
                "choices": ["fts", "semantic", "hybrid"],
            },
            {
                "key": "write_frequency",
                "description": "When to persist turns to Lore",
                "default": "turn",
                "choices": ["turn", "session"],
            },
            {
                "key": "dedup_threshold",
                "description": (
                    "rrf_score at/above which a new entry is treated as a "
                    "near-duplicate (higher = more similar, hybrid mode)"
                ),
                "default": str(DEDUP_THRESHOLD),
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        cfg_dir = Path(hermes_home) / "plugins" / "lore"
        try:
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "config.json").write_text(
                json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Lore save_config failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the Lore memory provider with the Hermes plugin system."""
    config = _load_plugin_config()
    provider = LoreMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
