# Lore

**The operational knowledge layer for engineers and their AI agents.**

[![Version](https://img.shields.io/badge/version-0.5.0-blue)](https://github.com/davidgut1982/lore-mcp)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-supported-blue)](https://postgresql.org)
[![CI](https://github.com/davidgut1982/lore-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/davidgut1982/lore-mcp/actions/workflows/ci.yml)

---

## The Problem

Your agents are capable. But every session, they start from zero — no knowledge of your infrastructure, no memory of what broke last month, no idea why you made that architecture call.

You re-explain. They re-discover. Context vanishes when the session ends.

**Lore fixes that.**

```
Without Lore:                    With Lore:
─────────────────                ──────────────────────────────────
Agent session starts             Agent queries Lore on startup
"How does our infra work?"       Gets: topology, gotchas, runbooks,
You re-explain everything        recent incidents, verified decisions
Session ends, context lost       Knowledge persists across all sessions
                                 Next agent starts informed
```

---

## What Lore Does

### Knowledge Base

Structured operational knowledge your agents query in real time.

```python
# Capture a hard-won gotcha
kb_add(topic="proxmox", title="LXC resolv.conf inherits from host — breaks without Tailscale",
       content="...", author="david", source_type="human")

# Agent queries before touching anything
kb_search(query="proxmox lxc dns")  # returns the gotcha instantly
```

### Investigations

Structured debugging with a traceable paper trail from symptom to resolution.

```python
investigation_add(topic="anki-media", title="Media not loading after container restart",
                  content="Hypothesis: mount collision between container volumes...")

investigation_log_experiment(title="Mount collision test", hypothesis="...",
                              results={"collision": True}, conclusion="Confirmed. Remove duplicate mount.")
```

Six months later, a different engineer hits the same issue. The trail is there.

### Journal

Permanent record of milestones, architecture decisions, and buying decisions.

```python
journal_append(entry_type="milestone",
               content="Migrated monitoring stack to ops bastion. Rationale: centralized visibility.",
               tags=["monitoring", "migration"])
```

---

## Attribution

In a multi-agent system, provenance matters. Every Lore entry carries `author`, `source_type`, and `verified`.

```
kb_search("proxmox lxc networking")

Results:
  [1] "LXC resolv.conf inherits from host — Tailscale breaks containers"
      author: david | source_type: human | verified: true

  [2] "LXC container DNS fix after Tailscale install"
      author: engineer-agent | source_type: agent | verified: null

  [3] "LXC DNS configuration reference"
      author: research-agent | source_type: agent | verified: false
```

Agents know: result 1 is production-safe. Result 2, spot-check first. Result 3, review before acting.

---

## Quick Start

### Solo (SQLite — no server needed)

```bash
pip install lore-mcp
export DB_BACKEND=sqlite KNOWLEDGE_DATA_DIR=~/.lore
lore-mcp
```

Add to Claude Code (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "lore": {
      "command": "lore-mcp",
      "env": {
        "DB_BACKEND": "sqlite",
        "KNOWLEDGE_DATA_DIR": "/home/yourname/.lore"
      }
    }
  }
}
```

### Team (PostgreSQL)

```bash
pip install lore-mcp
export DB_BACKEND=local DB_HOST=localhost DB_PORT=5432 \
       DB_NAME=lore DB_USER=lore_user DB_PASSWORD=yourpassword
lore-mcp --host 0.0.0.0 --port 5555
```

All agents on your team point at `http://your-server:5555/mcp`. One shared knowledge layer.

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
| `investigation_add` | Open or add to an investigation (topic, title, content, tags). |
| `investigation_list` | List investigations, filter by topic. |
| `investigation_get` | Fetch full investigation by ID. |
| `investigation_log_experiment` | Log a structured experiment with hypothesis, methodology, results, conclusion. |
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
| `deduplicate_results` | Deduplicate a result set by similarity. |
| `cluster_results` | Cluster results by topic. |

---

## Backends

| Backend | Use case | Setup |
|---|---|---|
| SQLite | Solo / local dev / single machine | No server, one env var |
| PostgreSQL | Team / shared / production | Self-hosted DB |
| Supabase | Cloud PostgreSQL | Managed, zero-ops |

---

## How It's Different

| Tool | Built for | What it remembers | Agent-native |
|---|---|---|---|
| OB1 / personal memory | One person | Your thoughts and captures | No |
| Mem0 / Zep | App developers | User preferences, conversations | Partially |
| Confluence / Notion | Human teams | Documentation (human-browsed) | No |
| **Lore** | **Engineering teams + AI agents** | **How your systems work** | **Yes** |

Lore is not a second brain. It's the operational intelligence layer your agents need to work in your environment — not just any environment.
