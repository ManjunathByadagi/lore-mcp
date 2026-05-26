# Lore

**`lore-knowledge-mcp`** · Operational knowledge layer for engineering teams and their AI agents.

[![Version](https://img.shields.io/badge/version-0.6.0-blue)](https://github.com/davidgut1982/lore-mcp)
[![CI](https://github.com/davidgut1982/lore-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/davidgut1982/lore-mcp/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)
[![Hybrid Search](https://img.shields.io/badge/search-hybrid%20%2B%20semantic-blueviolet)](https://github.com/davidgut1982/lore-mcp#semantic-search-v060)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Lore demo](docs/demo.gif)

---

## The Problem

Your agents start every session knowing nothing about your systems. Every runbook you've written. Every gotcha you've hit. Every incident you've debugged. None of it carries forward.

You re-explain. They re-discover. Context vanishes when the session ends.

**Lore fixes that.**

```
Without Lore                     With Lore
─────────────────────────────    ──────────────────────────────────
Agent starts fresh every time    Agent queries Lore on startup
"How does our infra work?"       Gets: topology, gotchas, runbooks,
You re-explain everything        past incidents, verified decisions
Context lost at session end      Knowledge persists across all sessions
```

---

## How It's Different

| Tool | Built for | What it remembers | Agent-native |
|---|---|---|---|
| OB1 / personal memory | One person | Your thoughts and captures | No |
| Mem0 / Zep | App developers | User preferences, conversations | Partially |
| Confluence / Notion | Human teams | Documentation (human-browsed) | No |
| **Lore** | **Engineering teams + AI agents** | **How your systems actually work — searchable by meaning, not just keywords** | **Yes** |

Lore is not a second brain. It's the operational intelligence your agents need to work in *your* environment — not just any environment.

---

## What Lore Does

### Knowledge Base

Your team's operational knowledge — always queryable by any agent. Capture the things that matter: runbooks, hard-won gotchas, architecture decisions, deployment state. Every entry carries attribution so agents know who wrote it and whether a human has verified it.

### Investigations

When something breaks, open a structured investigation. Document the symptom, test hypotheses, record what you tried and what you found. Six months later when the same issue resurfaces — different engineer, different agent — the trail is there.

### Journal

A permanent record of milestones, architecture decisions, and buying decisions. The kind of thing that lives in someone's head until they leave the team.

---

## Built for Multi-Agent Systems

In a multi-agent environment, provenance matters. Every Lore entry carries `author`, `source_type`, and `verified`.

```
kb_search("proxmox lxc dns")

  [1] "LXC inherits host resolv.conf — Tailscale breaks containers"
      david · human · ✓ verified

  [2] "LXC DNS fix after Tailscale install"
      engineer-agent · agent · unreviewed

  [3] "LXC DNS configuration reference"
      research-agent · agent · ✗ disputed
```

Your agents know: result 1 is production-safe. Result 2, spot-check before acting. Result 3, review first.

---

## Semantic Search (v0.6.0+)

Lore finds entries by meaning, not just keywords. Search "DNS broken in containers" and it returns an entry titled "LXC containers inherit resolv.conf from the host" — no keyword overlap required.

Powered by local sentence-transformers embeddings (no API key, no external calls), combined with FTS5 lexical search and Reciprocal Rank Fusion. The same model used by mcp-memory-service, fully self-hosted.

### Enable it

> **Heads up:** Semantic search is feature-complete and shipping in a future release. The v0.6.0 release that included it was yanked from PyPI on 2026-05-24 while we set up a proper staging and end-to-end testing pipeline. You can run it from source today by cloning the repo and running `pip install -e ".[semantic]"`.

```bash
# Once a stable release is published:
pip install lore-knowledge-mcp[semantic]
LORE_SEMANTIC_SEARCH=true lore-mcp
```

### What you get

| Mode | When to use |
|---|---|
| `fts` | Exact term matches (default when semantic is off) |
| `semantic` | Meaning-based retrieval, no keyword overlap needed |
| `hybrid` | Best of both — FTS5 + vector via RRF (recommended) |

### Backfill existing KB

If you already have entries, generate embeddings for them:

```
kb_backfill_embeddings()    # idempotent, safe to re-run
kb_embedding_status()       # check coverage
```

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `LORE_SEMANTIC_SEARCH` | `false` | Master switch — off = current behavior unchanged |
| `LORE_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 384d, ~90MB, English-optimized |
| `LORE_RRF_K` | `10` | Increase to 30–60 for corpora >10k entries |

For multilingual content, set `LORE_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` (same 384d, no schema change).

---

## Automating Lore in Your Workflow

Add one line to every agent's system prompt and one entry to `~/.mcp.json` — that's the entire integration. Each phase of your engineering workflow reads prior knowledge from Lore and writes its findings back, so nothing is re-discovered from scratch.

→ **[How to wire Lore into a 6-phase multi-agent pipeline](docs/multi-agent-workflow.md)** — full walkthrough with code examples for every phase: research, architecture review, implementation, adversarial code review, QA, and documentation.

---

## Quick Start

**No database setup required.** Lore runs out of the box with SQLite.

### 1. Install

```bash
pip install lore-knowledge-mcp
```

### Optional: semantic search

> **Note:** The v0.6.0 PyPI release was yanked — see [Semantic Search](#semantic-search-v060) for current install status.

```bash
pip install lore-knowledge-mcp[semantic]
```

Then set `LORE_SEMANTIC_SEARCH=true`. See [Semantic Search](#semantic-search-v060) for details.

### 2. Start the server

```bash
# Stdio mode (for local MCP clients like Claude Code)
lore-mcp

# HTTP mode (for remote or multi-agent access)
lore-mcp --host 0.0.0.0 --port 8000
```

### 3. Add to your MCP client

**Claude Code / Claude Desktop** — add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "lore": {
      "type": "stdio",
      "command": "lore-mcp"
    }
  }
}
```

Or for HTTP mode (recommended for teams):

```json
{
  "mcpServers": {
    "lore": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

That’s it. Lore is ready.

---

## Tool Reference

### Knowledge Base
| Tool | What it does |
|---|---|
| `kb_add` | Add an entry. Accepts `author`, `source_type` for attribution. |
| `kb_search` | Semantic search with optional topic filter. |
| `kb_get` | Fetch full entry by ID. |
| `kb_list` | List entries, filter by topic. |
| `kb_update` | Update content, tags, or set `verified` flag. |
| `kb_delete` | Delete entry (requires `confirm=true`). |

### Investigations
| Tool | What it does |
|---|---|
| `investigation_add` | Open or add to an investigation. |
| `investigation_list` | List investigations, filter by topic. |
| `investigation_get` | Fetch full investigation by ID. |
| `investigation_log_experiment` | Log a structured hypothesis → result → conclusion. |
| `investigation_list_experiments` | List all logged experiments. |

### Journal
| Tool | What it does |
|---|---|
| `journal_append` | Add a milestone, decision, or reflection. |
| `journal_list` | List recent entries (default 20). |
| `journal_get` | Fetch entry by ID. |
| `snapshot_config` | Snapshot a config object to the journal. |

### Document Ingestion
| Tool | What it does |
|---|---|
| `kb_ingest_doc` | Ingest a markdown file into the KB. |
| `kb_ingest_dir` | Batch-ingest a directory, with change detection. |
| `kb_sync_status` | Check what's changed since last sync. |

### MCP Index
| Tool | What it does |
|---|---|
| `mcp_index_scan` | Scan all configured MCP servers and index their tools. |
| `mcp_index_search` | Search indexed tools by description. |
| `mcp_index_get_server` | Get all tools for a specific MCP server. |
| `mcp_index_rebuild` | Force a full rescan. |

### Search
| Tool | What it does |
|---|---|
| `multi_search` | Search across KB, investigations, journal, and transcripts at once. |
| `search_local` | Search local files by content. |
| `search_transcripts` | Search Whisper transcript segments. |
| `deduplicate_results` | Deduplicate a result set by similarity threshold. |
| `cluster_results` | Cluster results by topic. |

---

## Backends

| | SQLite | PostgreSQL |
|---|---|---|
| Setup required | None | Existing PostgreSQL instance |
| Best for | Solo developers, local use | Teams, shared agents, production |
| Config | `DB_BACKEND=sqlite` (default) | `DB_BACKEND=postgres` + connection vars |
| Data location | `~/.local/share/lore/` | Your database |

**SQLite is the default.** No configuration needed — just install and run.

**PostgreSQL** is for teams who want a shared knowledge layer accessible from multiple machines or agents simultaneously.

```bash
# PostgreSQL setup
export DB_BACKEND=postgres
export DB_HOST=your-db-host
export DB_PORT=5432
export DB_NAME=lore
export DB_USER=your-user
export DB_PASSWORD=your-password
lore-mcp
```

---

## Hermes Memory Provider

A Hermes agent memory provider plugin that backs conversation memory with Lore is available as a separate package:

**[hermes-lore-plugin](https://github.com/davidgut1982/hermes-lore-plugin)** — drop-in memory provider for the [Hermes agent](https://github.com/NousResearch/Hermes). Stores KB entries in Lore, prefetches relevant context on session start, and deduplicates before storing.

---

## License

MIT — see [LICENSE](LICENSE)
