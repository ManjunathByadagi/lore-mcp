# Automating Lore Across a Multi-Agent Engineering Workflow

**TL;DR:** Every agent in a multi-agent pipeline starts cold — no memory of what the previous agent discovered, decided, or warned about. Lore fixes this by giving all agents a shared, searchable knowledge base. Wire it in once, and your agents stop reinventing the wheel on every run.

---

## The Cold-Start Problem

You have six agents working a single feature: research, architecture review, implementation, adversarial code review, QA, and documentation. Each one is a clean context window. The research agent discovers that your codebase already has a JWT refresh token pattern in `lib/auth/`. It writes a detailed analysis. Then it exits.

The engineer agent starts fresh. Nothing was passed automatically. It doesn't know about `lib/auth/`. It writes a second implementation from scratch. The code critic flags the duplication. The QA agent hits inconsistent behavior between the two implementations. The documentation agent documents the wrong one.

This isn't hypothetical — it's the default behavior of any orchestrated multi-agent system that treats agents as stateless workers. The agents are competent individually. The workflow is broken structurally.

---

## Setup: One Config Entry

Install Lore and start it:

```bash
pip install lore-knowledge-mcp

# Local (stdio)
lore-mcp

# Team (HTTP, accessible to all agents)
lore-mcp --host 0.0.0.0 --port 8000
```

Add to `~/.mcp.json`:

```json
{
  "mcpServers": {
    "lore": {
      "type": "http",
      "url": "http://your-lore-server:8000/mcp"
    }
  }
}
```

Add to every agent's system prompt:

```
At the start of every task, search Lore for relevant prior work.
At the end, write your key findings back to Lore with your agent name as author.
```

That's the entire integration. What follows is what it looks like across a real six-phase engineering workflow ([claude-mpm](https://github.com/bobmatnyc/claude-mpm)).

---

## The Six Phases

### Phase 1 — Research

The research agent investigates the codebase and gathers requirements. Without Lore it starts from zero every time. With Lore, it checks what's already been learned before spending time re-discovering it.

```python
# Start: search before investigating
kb_search(query="authentication patterns", tags=["auth"])
kb_search(query="feature requirements user-profile", tags=["requirements"])

# End: write findings so the engineer doesn't repeat the work
kb_add(
  title="JWT refresh token pattern — lib/auth/",
  content="Existing implementation in lib/auth/refresh.py. Uses 15m access / 7d refresh. Do not duplicate.",
  tags=["auth", "patterns", "do-not-duplicate"],
  author="research-agent"
)
```

### Phase 2 — Code Analysis

The architecture review agent reads the research findings and produces a verdict: APPROVED, NEEDS_IMPROVEMENT, or BLOCKED. It writes the verdict and its reasoning to Lore so the engineer gets it automatically.

```python
# Start: pull research findings
kb_search(query="architectural patterns api design", tags=["architecture"])

# End: record the verdict and the reasoning
kb_add(
  title="Architecture verdict: user-profile feature",
  content="APPROVED. Extend existing UserService. Do not create UserProfileService — violates single-responsibility boundary established in ADR-012.",
  tags=["verdict", "architecture", "user-profile"],
  author="architecture-agent"
)
```

The engineer in Phase 3 will find this. They won't create `UserProfileService`.

### Phase 3 — Implementation

The engineer reads the research findings and the architecture verdict before writing a single line.

```python
# Start: get full context before touching code
multi_search(queries=[
  "user-profile implementation patterns",
  "architecture verdict user-profile",
  "do-not-duplicate patterns"
])

# End: record what was built and any decisions made
kb_add(
  title="Implementation: user-profile endpoint",
  content="Extended UserService.get_profile(). Added caching via existing CacheLayer. Edge case: deleted users return 410 Gone, not 404 — per product spec.",
  tags=["implementation", "user-profile", "edge-cases"],
  author="engineer-agent"
)
```

### Phase 3.5 — Code Critic (Adversarial Review)

The code critic is independent of the implementer by design — it shouldn't be anchored by the engineer's framing. But it still needs context to distinguish intentional decisions from bugs.

```python
# Start: read implementation decisions
kb_search(query="implementation decisions user-profile")
kb_search(query="architecture verdict approved patterns")

# End: record what was flagged and what was intentional
kb_add(
  title="Code review: user-profile — 410 edge case intentional",
  content="Verified with architect notes. 410 Gone for deleted users is intentional per product spec. Not a bug.",
  tags=["code-review", "user-profile", "intentional-decisions"],
  author="critic-agent"
)
```

### Phase 4 — QA

The QA agent needs to know what was implemented, what edge cases exist, and what the code critic flagged.

```python
# Start: pull everything relevant before writing test plans
multi_search(queries=[
  "edge cases user-profile",
  "implementation decisions user-profile",
  "code review findings"
])

# End: record test results and any regressions
kb_add(
  title="QA results: user-profile — 4 scenarios tested",
  content="PASS: 200 active user, 410 deleted user, 401 unauthorized, 404 nonexistent. Cache invalidation confirmed on profile update.",
  tags=["qa", "test-results", "user-profile"],
  author="qa-agent"
)
```

### Phase 5 — Documentation

The documentation agent doesn't need to read the code. It reads Lore.

```python
# Start: gather all phase outputs
multi_search(queries=[
  "user-profile feature all phases",
  "intentional decisions user-profile",
  "test results user-profile"
])

# End: record what was documented
kb_add(
  title="Documentation written: user-profile API",
  content="Updated docs/api/user-profile.md. Included 410 behavior, cache invalidation note, example responses for all 4 QA scenarios.",
  tags=["documentation", "user-profile", "completed"],
  author="documentation-agent"
)
```

---

## The Investigation Thread Pattern

For complex bugs or multi-session features, Lore's investigation tracker is more appropriate than scattered `kb_add` entries. You open an investigation once and log experiments against it across as many sessions as it takes.

```python
# Open the investigation (usually Phase 1 or when the bug is identified)
investigation_start(
  title="Intermittent 503s on profile endpoint under load",
  description="P95 latency spikes to 8s under 500 concurrent users. Suspected: connection pool exhaustion.",
  tags=["bug", "performance", "profile-endpoint"],
  author="research-agent"
)

# Each agent logs what they tried
investigation_log_experiment(
  investigation_id="inv_503_profile",
  experiment="Increased connection pool from 10 to 50",
  result="503s reduced 60% but not eliminated",
  author="engineer-agent"
)

investigation_log_experiment(
  investigation_id="inv_503_profile",
  experiment="Added connection timeout of 2s with retry",
  result="503s eliminated. P95 back to 200ms under 500 concurrent.",
  author="engineer-agent"
)

# Record the conclusion
investigation_log_experiment(
  investigation_id="inv_503_profile",
  experiment="ROOT CAUSE CONFIRMED",
  result="Default timeout was infinite. Under pool exhaustion, requests queued indefinitely. 2s timeout + retry resolves.",
  author="qa-agent"
)
```

Next week, when a different feature triggers the same symptom, the research agent searches Lore and finds the full thread — every experiment, every result, the root cause, and who found it. No one repeats the investigation.

---

## The Compounding Effect

The first run through this workflow, agents write to Lore. The second run, they find something. By the tenth run, the knowledge base contains:

- Architectural decisions with verdicts and reasoning
- Implementation patterns that exist and shouldn't be duplicated
- Edge cases discovered the hard way
- QA scenarios that proved tricky
- Full investigation threads for every non-trivial bug

Attribution matters here. When you see `author="critic-agent"` on a note about a particular pattern, you know it came from adversarial review — not the engineer who implemented it. That's a different trust signal. When you see `author="research-agent"` and `verified=false`, you know a human hasn't signed off on it yet.

Teams running five or more agents across multiple sessions see this most clearly. By session ten, the research phase is faster because half the groundwork is already documented. The engineer doesn't rediscover existing patterns. QA doesn't re-derive known edge cases.

**The agents aren't smarter. The workflow is.**

---

## See Also

- [Quick Start](../README.md#quick-start)
- [Tool Reference](../README.md#tool-reference)
- [claude-mpm](https://github.com/bobmatnyc/claude-mpm) — the multi-agent PM framework this example is based on
