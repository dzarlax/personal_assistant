#!/bin/bash
# Clean up old Claude CLI sessions older than N days (default: 7).
# Sessions accumulate in ~/.claude/sessions/ from bridge calls.
#
# Usage: ./scripts/cleanup-sessions.sh [days]
DAYS="${1:-7}"
SESSIONS_DIR="${HOME}/.claude/sessions"

if [ ! -d "$SESSIONS_DIR" ]; then
  echo "No sessions directory found at $SESSIONS_DIR"
  exit 0
fi

COUNT=$(find "$SESSIONS_DIR" -type f -mtime +"$DAYS" | wc -l)
if [ "$COUNT" -eq 0 ]; then
  echo "No sessions older than $DAYS days."
  exit 0
fi

find "$SESSIONS_DIR" -type f -mtime +"$DAYS" -delete
echo "Deleted $COUNT sessions older than $DAYS days."
