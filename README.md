# Lore

**The operational knowledge layer for engineers and their AI agents.**

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/davidgut1982/advanced-knowledge-mcp)
[![Python](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-supported-blue)](https://postgresql.org)

---

## The Problem

Your AI agents are smart. But they start every session operationally blind.

They don't know how your infrastructure is built. They don't know why you made that architecture decision six months ago. They don't know what broke last time and how you fixed it. They don't know which engineer — or which agent — wrote that runbook, or whether it's been verified.

Every session, they start from zero.

**Lore fixes that.**

---

## What Lore Is

Lore is a persistent operational knowledge layer your entire team — humans and AI agents — shares and queries.

It's not a personal second brain. It's not conversation history. It's not a document store for humans to browse.

It's the **lore of your stack**: how things work, why decisions were made, and what happened when things broke — structured so AI agents can query it instantly and act on it correctly.

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

## Three Things Lore Does

### 1. Knowledge Base

The core. Structured operational knowledge your agents query in real time.

```python
# Capture a hard-won gotcha
kb_add(
    topic="pfsense",
    title="HAProxy load-server-state-from-file overrides cfg changes on reload",
    content="If load-server-state-from-file is enabled, a HAProxy reload will restore...",
    tags=["pfsense", "haproxy", "gotcha"],
    author="david",
    source_type="human",
)

# Agent queries before touching HAProxy
kb_search(query="HAProxy reload behavior")
# Returns the gotcha entry instantly
```

Every entry carries **attribution** — who wrote it (human or agent name), what type of source it is, and whether it's been human-verified. In a multi-agent system, your agents know the difference between:

- A runbook written and verified by a senior engineer
- A deployment note written by an agent, unreviewed
- An auto-captured system state entry (reference only)

### 2. Investigations

Structured ops debugging with a paper trail. When something breaks, you don't want scattered notes — you want a traceable trail from symptom to root cause to resolution.

```python
# Open an investigation
investigation_add(
    topic="anki-media-loading",
    title="Anki Media Loading Root Cause Analysis",
    content="Symptom: media files not loading after container restart. "
            "Hypothesis: mount collision between container volumes...",
    tags=["anki", "docker", "urgent"],
)

# Log the experiment
investigation_log_experiment(
    title="Mount collision test",
    hypothesis="Overlapping volume mounts cause file descriptor exhaustion",
    methodology="Reproduced with minimal compose config, isolated variables",
    results={"fd_count": 1024, "collision": True, "resolution": "Remove duplicate mount"},
    conclusion="Confirmed. Fix: remove /data volume from service B.",
)

# Final resolution entry
investigation_add(
    topic="anki-media-loading",
    title="RESOLVED: Anki Media Loading Fix",
    content="Root cause was mount collision. Fix verified in production.",
    tags=["anki", "resolved"],
)
```

Next time a similar issue happens — six months later, different engineer — the investigation trail is there.

### 3. Journal

Major milestones and inflection points. Architecture decisions. Buying decisions. Things you want a permanent record of.

```python
journal_append(
    entry_type="milestone",
    content="Migrated monitoring stack from latvian-vm to ops bastion. "
            "Rationale: centralized visibility, reduced per-VM overhead. "
            "All Grafana dashboards updated.",
    tags=["monitoring", "migration", "bastion"],
)
```

---

## Attribution: Built for Multi-Agent Systems

In a multi-agent environment, provenance matters. Lore tracks who wrote what.

```
kb_search("proxmox lxc networking")

Results:
  [1] "Proxmox LXC inherits host resolv.conf — Tailscale breaks containers"
      author: david | source_type: human | verified: true

  [2] "LXC container DNS fix after Tailscale install"
      author: engineer-agent | source_type: agent | verified: null

  [3] "LXC DNS configuration reference"
      author: research-agent | source_type: agent | verified: false
```

Your agents understand: trust level 1 is production-safe. Trust level 2, spot-check before acting. Trust level 3, do not follow without review.

---

## Quick Start

### Solo / Local (SQLite — no server needed)

```bash
git clone https://github.com/davidgut1982/advanced-knowledge-mcp.git
cd advanced-knowledge-mcp
pip install -e .

export DB_BACKEND=sqlite
export KNOWLEDGE_DATA_DIR=~/.lore

knowledge-mcp  # starts on stdio
```

Add to Claude Code:
```json
{
  "mcpServers": {
    "lore": {
      "command": "knowledge-mcp",
      "env": {
        "DB_BACKEND": "sqlite",
        "KNOWLEDGE_DATA_DIR": "/home/yourname/.lore"
      }
    }
  }
}
```

### Team / Shared (PostgreSQL)

```bash
# On your server / LXC:
git clone https://github.com/davidgut1982/advanced-knowledge-mcp.git
cd advanced-knowledge-mcp
pip install -e .

# Set up PostgreSQL (or use docker-compose)
docker-compose up -d postgres

export DB_BACKEND=local
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=lore
export DB_USER=lore_user
export DB_PASSWORD=yourpassword

knowledge-mcp --host 0.0.0.0 --port 5555
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
| `kb_search` | Search the knowledge base. |
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

---

## Version

`0.4.0` — Lore rebrand. Stripped to three focused systems (KB, Investigations, Journal). Added attribution model (author, source_type, verified). Removed knowledge graph and source tracking.
