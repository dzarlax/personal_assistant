#!/bin/bash
# Update claude-bridge binary from GitHub Releases and restart the systemd service.
set -e

REPO="dzarlax/personal_assistant"
BIN="/root/personal_assistant/bridge/claude-bridge"
TMP="/tmp/claude-bridge-new"

echo "Downloading latest claude-bridge..."
curl -sfL -o "$TMP" \
  "https://github.com/$REPO/releases/download/latest/claude-bridge-linux-amd64"

chmod +x "$TMP"

# Verify it is a valid ELF binary
file "$TMP" | grep -q ELF || { echo "ERROR: invalid binary"; rm -f "$TMP"; exit 1; }

mv "$TMP" "$BIN"
systemctl restart claude-bridge
echo "Updated and restarted."
systemctl status claude-bridge --no-pager | head -8
