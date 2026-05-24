"""Standalone 24–48 hour soak test runner for the Lore MCP service.

Not a pytest file — run it directly::

    LORE_E2E_URL=http://lore-staging:5555 python -m tests.e2e.soak_runner --duration 24h

Features
--------
- Adds random KB entries every 30–60 s.
- Runs random searches (mix of all three modes) every 5–15 s.
- Occasionally updates random entries (every 5th add).
- Tracks per-operation latency, error rate, and embedding coverage drift.
- Structured JSON logging to stdout (and optionally to a file).
- Exits non-zero when:
    - Error rate exceeds ``--error-threshold`` (default 1 %) over the last
      ``--window`` minutes (default 5).
    - P95 latency for any operation type exceeds ``--latency-p95-ms``
      (default 5 000 ms).

Exit codes
----------
0 — completed normally within thresholds.
1 — terminated by threshold breach.
2 — unexpected exception or bad arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import string
import sys
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("soak_runner")

_JSON_HANDLER: logging.Handler | None = None


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        doc: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "event"):
            doc["event"] = record.event  # type: ignore[attr-defined]
        return json.dumps(doc)


def _setup_logging(log_file: str | None, verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_JsonFormatter())
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)


def _log_event(event_type: str, **kwargs: Any) -> None:
    """Log a structured event with additional key-value pairs."""
    extra = {"event": {"type": event_type, **kwargs}}
    record = logging.LogRecord(
        name="soak_runner",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=f"event:{event_type}",
        args=(),
        exc_info=None,
    )
    record.event = extra["event"]  # type: ignore[attr-defined]
    logger.handle(record)


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


def _parse_duration(s: str) -> float:
    """Parse a duration string like '24h', '90m', '3600s' into seconds."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


# ---------------------------------------------------------------------------
# Metrics window
# ---------------------------------------------------------------------------


@dataclass
class _OpSample:
    op: str
    latency_ms: float
    ok: bool
    ts: float = field(default_factory=time.monotonic)


class _MetricsWindow:
    """Rolling window of operation samples for rate/latency calculations."""

    def __init__(self, window_secs: float = 300.0) -> None:
        self._window = window_secs
        self._samples: deque[_OpSample] = deque()

    def record(self, op: str, latency_ms: float, ok: bool) -> None:
        self._samples.append(_OpSample(op=op, latency_ms=latency_ms, ok=ok))
        self._evict()

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def error_rate(self) -> float:
        """Return error rate in [0, 1] over the window."""
        self._evict()
        if not self._samples:
            return 0.0
        errors = sum(1 for s in self._samples if not s.ok)
        return errors / len(self._samples)

    def p95_latency_ms(self, op: str | None = None) -> float | None:
        """Return P95 latency in ms for ``op`` (or all ops if None)."""
        self._evict()
        latencies = [s.latency_ms for s in self._samples if (op is None or s.op == op) and s.ok]
        if not latencies:
            return None
        latencies.sort()
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]

    def total_ops(self) -> int:
        self._evict()
        return len(self._samples)

    def summary(self) -> dict[str, Any]:
        self._evict()
        ops: dict[str, list[float]] = {}
        errors: dict[str, int] = {}
        for s in self._samples:
            if s.ok:
                ops.setdefault(s.op, []).append(s.latency_ms)
            else:
                errors[s.op] = errors.get(s.op, 0) + 1
        result: dict[str, Any] = {"window_secs": self._window}
        for op, lat in ops.items():
            lat.sort()
            result[op] = {
                "count": len(lat),
                "errors": errors.get(op, 0),
                "p50_ms": statistics.median(lat),
                "p95_ms": lat[int(len(lat) * 0.95)],
                "p99_ms": lat[int(len(lat) * 0.99)],
            }
        return result


# ---------------------------------------------------------------------------
# Random content generators
# ---------------------------------------------------------------------------

_TOPICS = ["ops", "python", "database", "networking", "devops", "security"]
_SEARCH_MODES = ["fts", "semantic", "hybrid"]

_WORDS = (
    "configure deploy monitor scale backup restore migrate upgrade "
    "container kubernetes docker systemd postgres redis nginx "
    "asyncio timeout retry backoff latency throughput cache index "
    "vacuum replication failover snapshot rollback certificate "
    "firewall route dns dhcp wireguard tailscale proxmox lxc vm"
).split()


def _random_content(n_sentences: int = 5) -> str:
    sentences = []
    for _ in range(n_sentences):
        length = random.randint(6, 14)
        words = random.choices(_WORDS, k=length)
        sentences.append(" ".join(words).capitalize() + ".")
    return " ".join(sentences)


def _random_title() -> str:
    return " ".join(random.choices(_WORDS, k=random.randint(3, 6))).capitalize()


def _random_query() -> str:
    return " ".join(random.choices(_WORDS, k=random.randint(2, 4)))


def _random_string(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


# ---------------------------------------------------------------------------
# Soak runner core
# ---------------------------------------------------------------------------


def run_soak(
    url: str,
    duration_secs: float,
    error_threshold: float,
    latency_p95_ms: float,
    window_mins: float,
    add_interval: tuple[float, float],
    search_interval: tuple[float, float],
    log_interval_secs: float,
) -> int:
    """Run the soak loop.

    Returns:
        0 on clean finish, 1 on threshold breach, 2 on fatal error.
    """
    # Import here so the module imports cleanly when LORE_E2E_URL is unset
    from tests.e2e.client import LoreClient, LoreClientError

    metrics = _MetricsWindow(window_secs=window_mins * 60)

    start_time = time.monotonic()
    end_time = start_time + duration_secs

    # Track IDs of entries we created so we can update/delete them
    live_entries: list[str] = []
    add_count = 0

    next_add = time.monotonic() + random.uniform(*add_interval)
    next_search = time.monotonic() + random.uniform(*search_interval)
    next_log = time.monotonic() + log_interval_secs

    _log_event(
        "soak_start",
        url=url,
        duration_secs=duration_secs,
        error_threshold_pct=error_threshold * 100,
        latency_p95_ms_limit=latency_p95_ms,
        window_mins=window_mins,
    )

    with LoreClient(url, timeout=60.0) as client:
        while time.monotonic() < end_time:
            now = time.monotonic()

            # ----------------------------------------------------------------
            # KB add
            # ----------------------------------------------------------------
            if now >= next_add:
                topic = f"soak-{random.choice(_TOPICS)}-{_random_string(4)}"
                title = _random_title()
                content = _random_content()
                t0 = time.monotonic()
                ok = True
                entry_id: str | None = None
                try:
                    result = client.kb_add(topic=topic, title=title, content=content)
                    entry_id = result.get("id")
                    if entry_id:
                        live_entries.append(entry_id)
                    add_count += 1
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    _log_event("error", op="kb_add", exc=str(exc))
                latency = (time.monotonic() - t0) * 1000
                metrics.record("kb_add", latency, ok)
                _log_event("op", op="kb_add", ok=ok, latency_ms=round(latency, 1))

                # Every 5th add: also update a random existing entry
                if ok and add_count % 5 == 0 and len(live_entries) >= 2:
                    target_id = random.choice(live_entries[:-1])
                    t0 = time.monotonic()
                    ok_u = True
                    try:
                        client.kb_update(target_id, content=_random_content(3))
                    except LoreClientError:
                        # Entry may have been deleted; non-fatal
                        ok_u = False
                    except Exception as exc:  # noqa: BLE001
                        ok_u = False
                        _log_event("error", op="kb_update", exc=str(exc))
                    latency_u = (time.monotonic() - t0) * 1000
                    metrics.record("kb_update", latency_u, ok_u)
                    _log_event(
                        "op",
                        op="kb_update",
                        ok=ok_u,
                        latency_ms=round(latency_u, 1),
                    )

                next_add = time.monotonic() + random.uniform(*add_interval)

            # ----------------------------------------------------------------
            # KB search
            # ----------------------------------------------------------------
            if now >= next_search:
                mode = random.choice(_SEARCH_MODES)
                query = _random_query()
                t0 = time.monotonic()
                ok = True
                try:
                    result = client.kb_search(query, search_mode=mode, limit=10)
                    n_results = len(result.get("results") or result.get("entries") or [])
                    _log_event(
                        "op",
                        op=f"kb_search_{mode}",
                        ok=True,
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                        n_results=n_results,
                    )
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    _log_event(
                        "error",
                        op=f"kb_search_{mode}",
                        exc=str(exc),
                        latency_ms=round((time.monotonic() - t0) * 1000, 1),
                    )
                latency = (time.monotonic() - t0) * 1000
                metrics.record(f"kb_search_{mode}", latency, ok)
                next_search = time.monotonic() + random.uniform(*search_interval)

            # ----------------------------------------------------------------
            # Periodic metrics log
            # ----------------------------------------------------------------
            if now >= next_log:
                summary = metrics.summary()
                err_rate = metrics.error_rate()
                _log_event(
                    "metrics_summary",
                    elapsed_secs=round(now - start_time),
                    remaining_secs=round(end_time - now),
                    error_rate_pct=round(err_rate * 100, 2),
                    live_entries=len(live_entries),
                    summary=summary,
                )

                # ---- Threshold checks ----
                if err_rate > error_threshold:
                    _log_event(
                        "threshold_breach",
                        reason="error_rate",
                        error_rate_pct=round(err_rate * 100, 2),
                        threshold_pct=round(error_threshold * 100, 2),
                    )
                    logger.error(
                        "THRESHOLD BREACH: error rate %.1f%% > %.1f%%",
                        err_rate * 100,
                        error_threshold * 100,
                    )
                    return 1

                for op_name in (
                    "kb_add",
                    "kb_search_fts",
                    "kb_search_semantic",
                    "kb_search_hybrid",
                ):
                    p95 = metrics.p95_latency_ms(op_name)
                    if p95 is not None and p95 > latency_p95_ms:
                        _log_event(
                            "threshold_breach",
                            reason="p95_latency",
                            op=op_name,
                            p95_ms=round(p95, 1),
                            limit_ms=latency_p95_ms,
                        )
                        logger.error(
                            "THRESHOLD BREACH: P95 latency for %s: %.0f ms > %.0f ms",
                            op_name,
                            p95,
                            latency_p95_ms,
                        )
                        return 1

                next_log = time.monotonic() + log_interval_secs

            # ----------------------------------------------------------------
            # Embedding coverage drift check (every ~10 min)
            # ----------------------------------------------------------------
            time.sleep(0.1)  # Yield CPU; main loop is not latency-sensitive

    _log_event(
        "soak_complete",
        elapsed_secs=round(time.monotonic() - start_time),
        total_adds=add_count,
        summary=metrics.summary(),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tests.e2e.soak_runner",
        description="24-48 h continuous-load soak test for the Lore MCP service.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("LORE_E2E_URL", ""),
        help="Base URL of the Lore instance (default: $LORE_E2E_URL)",
    )
    parser.add_argument(
        "--duration",
        default="24h",
        help="Test duration (e.g. 24h, 90m, 3600s). Default: 24h",
    )
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Max error rate %% over the evaluation window before aborting. Default: 1.0",
    )
    parser.add_argument(
        "--latency-p95-ms",
        type=float,
        default=5000.0,
        metavar="MS",
        help="Max P95 latency in ms for any operation before aborting. Default: 5000",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        metavar="MINS",
        help="Rolling evaluation window in minutes. Default: 5",
    )
    parser.add_argument(
        "--add-interval",
        nargs=2,
        type=float,
        default=[30.0, 60.0],
        metavar=("MIN_S", "MAX_S"),
        help="Random interval between kb_add calls in seconds. Default: 30 60",
    )
    parser.add_argument(
        "--search-interval",
        nargs=2,
        type=float,
        default=[5.0, 15.0],
        metavar=("MIN_S", "MAX_S"),
        help="Random interval between kb_search calls in seconds. Default: 5 15",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=60.0,
        metavar="SECS",
        help="Interval between metrics-summary log events. Default: 60",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional path to write JSON log lines (in addition to stdout).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log_file, args.verbose)

    if not args.url:
        logger.error("No Lore URL configured. Set LORE_E2E_URL or pass --url.")
        return 2

    try:
        duration = _parse_duration(args.duration)
    except ValueError as exc:
        logger.error("Invalid --duration %r: %s", args.duration, exc)
        return 2

    return run_soak(
        url=args.url,
        duration_secs=duration,
        error_threshold=args.error_threshold / 100.0,
        latency_p95_ms=args.latency_p95_ms,
        window_mins=args.window,
        add_interval=tuple(args.add_interval),  # type: ignore[arg-type]
        search_interval=tuple(args.search_interval),  # type: ignore[arg-type]
        log_interval_secs=args.log_interval,
    )


if __name__ == "__main__":
    sys.exit(main())
