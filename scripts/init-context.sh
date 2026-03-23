#!/bin/bash
# Creates project context directory for Claude Code bridge mode.
# Usage: ./scripts/init-context.sh /path/to/assistant_context

set -euo pipefail

CONTEXT_DIR="${1:?Usage: $0 /path/to/context-dir}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MCP_JSON="$SCRIPT_DIR/config/mcp.json"

if [ ! -f "$MCP_JSON" ]; then
    echo "Error: $MCP_JSON not found. Run from the personal_assistant repo." >&2
    exit 1
fi

echo "Creating assistant context at: $CONTEXT_DIR"

mkdir -p "$CONTEXT_DIR"/{.claude,notes,tasks,reference}

# Templates (only if not already present — don't overwrite user edits)
[ -f "$CONTEXT_DIR/CLAUDE.md" ] || cp "$SCRIPT_DIR/templates/CLAUDE.md" "$CONTEXT_DIR/CLAUDE.md"
[ -f "$CONTEXT_DIR/.claude/settings.json" ] || cp "$SCRIPT_DIR/templates/settings.json" "$CONTEXT_DIR/.claude/settings.json"

# Symlink to mcp.json (machine-specific, recreate on each run)
ln -sf "$MCP_JSON" "$CONTEXT_DIR/.mcp.json"

touch "$CONTEXT_DIR/notes/.gitkeep"
touch "$CONTEXT_DIR/tasks/.gitkeep"
touch "$CONTEXT_DIR/reference/.gitkeep"

echo "Done."
echo "  Context: $CONTEXT_DIR"
echo "  MCP:     $CONTEXT_DIR/.mcp.json -> $MCP_JSON"
echo "  Test:    cd $CONTEXT_DIR && claude -p 'Hello'"
