#!/bin/bash
# Full setup for Personal Assistant bot.
# Usage: ./scripts/setup.sh [--with-claude /path/to/assistant_context]
#
# Without flags: sets up the bot (copies example configs, creates data dir).
# With --with-claude: also sets up Claude Bridge (context dir, binary, token).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

REPO="dzarlax/personal_assistant"

echo "=== Personal Assistant Setup ==="
echo ""

# --- Base setup ---

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env — fill in your API keys and tokens"
else
    echo ".env exists, skipping"
fi

if [ ! -f config/mcp.json ] && [ -f config/mcp.json.example ]; then
    cp config/mcp.json.example config/mcp.json
    echo "Created config/mcp.json from example"
fi

if [ ! -f config/system_prompt.md ] && [ -f config/system_prompt.md.example ]; then
    cp config/system_prompt.md.example config/system_prompt.md
    echo "Created config/system_prompt.md from example"
fi

mkdir -p data
echo "Base setup done."

# --- Parse --with-claude flag ---

CONTEXT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --with-claude) CONTEXT_DIR="${2:?--with-claude requires a path}"; shift 2 ;;
        *) shift ;;
    esac
done

if [ -z "$CONTEXT_DIR" ]; then
    echo ""
    echo "Next steps:"
    echo "  1. Edit .env with your API keys"
    echo "  2. Run: make docker-up"
    echo ""
    echo "To add Claude Bridge: ./scripts/setup.sh --with-claude /path/to/context"
    exit 0
fi

# --- Claude Bridge ---

echo ""
echo "=== Claude Bridge Setup ==="

if ! command -v claude &>/dev/null; then
    echo "Error: Claude Code CLI not found."
    echo "  npm install -g @anthropic-ai/claude-code && claude"
    exit 1
fi

# Context directory
"$SCRIPT_DIR/scripts/init-context.sh" "$CONTEXT_DIR"

# Build or download bridge binary
BRIDGE_BIN="$SCRIPT_DIR/bridge/claude-bridge"
if [ ! -f "$BRIDGE_BIN" ]; then
    if command -v go &>/dev/null; then
        echo "Building claude-bridge..."
        (cd "$SCRIPT_DIR/bridge" && go build -o claude-bridge .)
    else
        echo "Downloading claude-bridge..."
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m); [ "$ARCH" = "x86_64" ] && ARCH="amd64"
        curl -sL -o "$BRIDGE_BIN" \
            "https://github.com/$REPO/releases/latest/download/claude-bridge-${OS}-${ARCH}" \
            && chmod +x "$BRIDGE_BIN" \
            || { echo "Download failed. Install Go and re-run, or build manually: cd bridge && go build -o claude-bridge ."; exit 1; }
    fi
fi
echo "Bridge binary: $BRIDGE_BIN"

# Generate shared secret + add to .env
TOKEN=$(openssl rand -hex 16)
if ! grep -q "CLAUDE_BRIDGE_TOKEN" .env 2>/dev/null; then
    printf "\n# Claude Bridge\nCLAUDE_BRIDGE_TOKEN=%s\nCLAUDE_BRIDGE_PROJECT_DIR=%s\n" "$TOKEN" "$CONTEXT_DIR" >> .env
    echo "Added CLAUDE_BRIDGE_TOKEN and PROJECT_DIR to .env"
else
    echo "CLAUDE_BRIDGE_TOKEN already in .env"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Start bridge:  source .env && ./bridge/claude-bridge"
echo "Start bot:     make docker-up"
