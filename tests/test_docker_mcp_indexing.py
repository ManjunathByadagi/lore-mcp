"""Docker-MCP indexing test (legacy).

The original test depended on a real ~/.claude/mcp.json layout and a
'supabase' global on lore.server. The MCP-index path was rewritten to
use the unified db_client in v0.4.0, so this test no longer maps
cleanly onto the codebase.

TODO(v0.5.0): rewrite against the current MCPIndexScanner API using
the lore.db_client interface (sqlite backend) and a fixture mcp.json
under tmp_path. Until then this whole module is skipped.
"""

import pytest

pytest.skip(
    "needs rewrite for v0.5.0 (MCPIndexScanner now uses db_client, not raw supabase)",
    allow_module_level=True,
)
