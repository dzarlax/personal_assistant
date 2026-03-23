#!/bin/bash
# Full setup for Personal Assistant bot.
# Usage: ./scripts/setup.sh [--with-claude /path/to/assistant_context]
#
# Without flags: sets up the bot (copies example configs, creates data dir).
# With --with-claude: also sets up Claude Bridge (context dir, bridge binary, token).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Personal Assistant Setup ==="
echo ""

# --- Base setup ---

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — fill in your API keys and tokens"
else
    echo ".env already exists, skipping"
fi

if [ ! -f config/mcp.json ]; then
    if [ -f config/mcp.json.example ]; then
        cp config/mcp.json.example config/mcp.json
        echo "Created config/mcp.json from example"
    fi
else
    echo "config/mcp.json already exists, skipping"
fi

if [ ! -f config/system_prompt.md ]; then
    if [ -f config/system_prompt.md.example ]; then
        cp config/system_prompt.md.example config/system_prompt.md
        echo "Created config/system_prompt.md from example"
    fi
else
    echo "config/system_prompt.md already exists, skipping"
fi

mkdir -p data

echo ""
echo "Base setup done."

# --- Claude Bridge (optional) ---

CONTEXT_DIR=""
for arg in "$@"; do
    if [ "$arg" = "--with-claude" ]; then
        shift
        CONTEXT_DIR="${1:-}"
        shift || true
        break
    fi
done

if [ -z "$CONTEXT_DIR" ]; then
    echo ""
    echo "To add Claude Bridge, re-run with:"
    echo "  ./scripts/setup.sh --with-claude /path/to/assistant_context"
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env with your API keys"
    echo "  2. Edit config/config.yaml if needed"
    echo "  3. Run: make docker-up"
    exit 0
fi

echo ""
echo "=== Claude Bridge Setup ==="
echo ""

# Check prerequisites
if ! command -v claude &>/dev/null; then
    echo "Error: Claude Code CLI not found. Install with:"
    echo "  npm install -g @anthropic-ai/claude-code"
    echo "  claude  # login with your Anthropic account"
    exit 1
fi

# Create context directory
"$SCRIPT_DIR/scripts/init-context.sh" "$CONTEXT_DIR"

# Generate shared secret
TOKEN=$(openssl rand -hex 16)

# Build bridge
echo ""
echo "Building claude-bridge..."
(cd "$SCRIPT_DIR/bridge" && go build -o claude-bridge .)
echo "Built: bridge/claude-bridge"

# Generate bridge config
cat > "$SCRIPT_DIR/bridge/bridge.yaml" << EOF
listen: "127.0.0.1:9900"
project_dir: "$CONTEXT_DIR"
cli_path: "claude"
default_timeout: 120
max_concurrent: 2
auth_token: "$TOKEN"
EOF
echo "Created bridge/bridge.yaml"

# Add token to .env if not already there
if ! grep -q "CLAUDE_BRIDGE_TOKEN" .env 2>/dev/null; then
    echo "" >> .env
    echo "# Claude Bridge shared secret" >> .env
    echo "CLAUDE_BRIDGE_TOKEN=$TOKEN" >> .env
    echo "Added CLAUDE_BRIDGE_TOKEN to .env"
else
    echo "CLAUDE_BRIDGE_TOKEN already in .env, skipping (update manually if needed)"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Start the bridge:"
echo "  ./bridge/claude-bridge bridge/bridge.yaml"
echo ""
echo "Start the bot:"
echo "  make docker-up"
echo ""
echo "Or run both:"
echo "  ./bridge/claude-bridge bridge/bridge.yaml &"
echo "  make docker-up"
