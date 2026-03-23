#!/bin/bash
# Creates project context directory for Claude Code Channel Mode.
# Usage: ./scripts/init-context.sh /path/to/assistant_context

set -euo pipefail

CONTEXT_DIR="${1:?Usage: $0 /path/to/context-dir}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MCP_JSON="$SCRIPT_DIR/config/mcp.json"
CHANNEL_DIR="$SCRIPT_DIR/channel"
PA_BRIDGE="$CHANNEL_DIR/pa-bridge.ts"

if [ ! -f "$MCP_JSON" ]; then
    echo "Error: $MCP_JSON not found. Run from the personal_assistant repo." >&2
    exit 1
fi

echo "Creating assistant context at: $CONTEXT_DIR"

mkdir -p "$CONTEXT_DIR"/{.claude,notes,tasks,reference}

# Templates (only if not already present — don't overwrite user edits)
[ -f "$CONTEXT_DIR/CLAUDE.md" ] || cp "$SCRIPT_DIR/templates/CLAUDE.md" "$CONTEXT_DIR/CLAUDE.md"
[ -f "$CONTEXT_DIR/.claude/settings.json" ] || cp "$SCRIPT_DIR/templates/settings.json" "$CONTEXT_DIR/.claude/settings.json"

# Generate .mcp.json: merge MCP servers from config + add pa-bridge stdio entry
# jq merges the existing HTTP servers with the local pa-bridge command
if command -v jq &>/dev/null; then
    jq --arg bridge "$PA_BRIDGE" \
       '.mcpServers["pa-bridge"] = {"command": "bun", "args": [$bridge]}' \
       "$MCP_JSON" > "$CONTEXT_DIR/.mcp.json"
    echo "  Generated .mcp.json with pa-bridge"
else
    # Fallback: symlink without pa-bridge (manual setup needed)
    echo "  Warning: jq not found, creating symlink instead (pa-bridge not included)"
    ln -sf "$MCP_JSON" "$CONTEXT_DIR/.mcp.json"
fi

# Install pa-bridge dependencies
if [ -f "$CHANNEL_DIR/package.json" ] && command -v bun &>/dev/null; then
    echo "  Installing pa-bridge dependencies..."
    (cd "$CHANNEL_DIR" && bun install --silent 2>/dev/null)
fi

touch "$CONTEXT_DIR/notes/.gitkeep"
touch "$CONTEXT_DIR/tasks/.gitkeep"
touch "$CONTEXT_DIR/reference/.gitkeep"

echo "Done."
echo "  Context:  $CONTEXT_DIR"
echo "  Bridge:   $PA_BRIDGE"
echo "  Test:     cd $CONTEXT_DIR && claude --dangerously-load-development-channels server:pa-bridge -p 'Hello'"
