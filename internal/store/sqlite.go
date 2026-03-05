package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"telegram-agent/internal/llm"
)

const sqliteSchema = `
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    role         TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    parts        TEXT,
    tool_calls   TEXT,
    tool_call_id TEXT,
    is_summary   INTEGER NOT NULL DEFAULT 0,
    is_compacted INTEGER NOT NULL DEFAULT 0,
    is_reset     INTEGER NOT NULL DEFAULT 0,
    created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_id, id);
`

const (
	sqliteMaxHistory   = 30
	sessionIdleTimeout = 4 * time.Hour
)

type SQLite struct {
	db *sql.DB
}

func NewSQLite(path string) (*SQLite, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite doesn't support concurrent writes
	if _, err := db.Exec(sqliteSchema); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}
	// Migration: add parts column for existing databases
	db.Exec(`ALTER TABLE messages ADD COLUMN parts TEXT`) //nolint:errcheck
	return &SQLite{db: db}, nil
}

func (s *SQLite) GetHistory(chatID int64) []llm.Message {
	lastReset := s.lastResetID(chatID)
	rows, err := s.db.Query(`
		SELECT role, content, parts, tool_calls, tool_call_id
		FROM messages
		WHERE chat_id = ? AND id > ? AND is_compacted = 0 AND is_reset = 0
		ORDER BY id DESC LIMIT ?`,
		chatID, lastReset, sqliteMaxHistory)
	if err != nil {
		return nil
	}
	defer rows.Close()
	return reverseMessages(scanMessages(rows))
}

func (s *SQLite) AddMessage(chatID int64, msg llm.Message) {
	// Auto session break: if user sends a message after a long pause, start fresh
	if msg.Role == "user" {
		if last := s.lastMessageTime(chatID); !last.IsZero() && time.Since(last) > sessionIdleTimeout {
			slog.Info("auto session break", "chat_id", chatID, "idle", time.Since(last).Round(time.Minute))
			s.insertSessionBreak(chatID, "AUTO_SESSION_BREAK")
		}
	}

	tcJSON, tcID := encodeToolFields(msg)
	var partsJSON sql.NullString
	if len(msg.Parts) > 0 {
		if b, err := json.Marshal(msg.Parts); err == nil {
			partsJSON = sql.NullString{String: string(b), Valid: true}
		}
	}
	_, err := s.db.Exec(`
		INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id)
		VALUES (?, ?, ?, ?, ?, ?)`,
		chatID, msg.Role, msg.Content, partsJSON, tcJSON, tcID)
	if err != nil {
		slog.Error("sqlite AddMessage failed", "role", msg.Role, "err", err)
	}
}

func (s *SQLite) lastMessageTime(chatID int64) time.Time {
	var ts sql.NullString
	s.db.QueryRow(`
		SELECT created_at FROM messages
		WHERE chat_id = ? AND is_reset = 0
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&ts)
	if !ts.Valid {
		return time.Time{}
	}
	t, _ := time.Parse("2006-01-02 15:04:05", ts.String)
	return t
}

func (s *SQLite) ClearHistory(chatID int64) {
	// Full reset — no summary carried over
	s.db.Exec(`INSERT INTO messages (chat_id, role, content, is_reset) VALUES (?, 'system', 'CONTEXT_RESET', 1)`, chatID) //nolint:errcheck
}

// insertSessionBreak starts a new session and carries over the last summary.
func (s *SQLite) insertSessionBreak(chatID int64, reason string) {
	s.db.Exec(`INSERT INTO messages (chat_id, role, content, is_reset) VALUES (?, 'system', ?, 1)`, chatID, reason) //nolint:errcheck

	var summary sql.NullString
	s.db.QueryRow(`
		SELECT content FROM messages
		WHERE chat_id = ? AND is_summary = 1
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&summary)
	if summary.Valid && summary.String != "" {
		s.db.Exec(`INSERT INTO messages (chat_id, role, content, is_summary) VALUES (?, 'assistant', ?, 1)`, chatID, summary.String) //nolint:errcheck
	}
}

func (s *SQLite) GetAllActive(chatID int64) ([]MessageRow, error) {
	lastReset := s.lastResetID(chatID)
	rows, err := s.db.Query(`
		SELECT id, role, content, parts, tool_calls, tool_call_id
		FROM messages
		WHERE chat_id = ? AND id > ? AND is_compacted = 0 AND is_reset = 0
		ORDER BY id ASC`,
		chatID, lastReset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []MessageRow
	for rows.Next() {
		var row MessageRow
		var partsJSON, tcJSON, tcID sql.NullString
		if err := rows.Scan(&row.ID, &row.Message.Role, &row.Message.Content, &partsJSON, &tcJSON, &tcID); err != nil {
			continue
		}
		if partsJSON.Valid && partsJSON.String != "" {
			json.Unmarshal([]byte(partsJSON.String), &row.Message.Parts) //nolint:errcheck
		}
		if tcJSON.Valid && tcJSON.String != "" {
			json.Unmarshal([]byte(tcJSON.String), &row.Message.ToolCalls) //nolint:errcheck
		}
		row.Message.ToolCallID = tcID.String
		result = append(result, row)
	}
	return result, rows.Err()
}

func (s *SQLite) AddSummary(chatID int64, content string) {
	s.db.Exec(`
		INSERT INTO messages (chat_id, role, content, is_summary)
		VALUES (?, 'assistant', ?, 1)`,
		chatID, content)
}

func (s *SQLite) MarkCompacted(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	placeholders := strings.Repeat("?,", len(ids))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		args[i] = id
	}
	_, err := s.db.Exec(
		"UPDATE messages SET is_compacted = 1 WHERE id IN ("+placeholders+")",
		args...,
	)
	return err
}

func (s *SQLite) ActiveCharCount(chatID int64) int {
	lastReset := s.lastResetID(chatID)
	var total sql.NullInt64
	s.db.QueryRow(`
		SELECT SUM(LENGTH(content))
		FROM messages
		WHERE chat_id = ? AND id > ? AND is_compacted = 0 AND is_reset = 0`,
		chatID, lastReset).Scan(&total)
	return int(total.Int64)
}

func (s *SQLite) lastResetID(chatID int64) int64 {
	var id sql.NullInt64
	s.db.QueryRow(`
		SELECT id FROM messages
		WHERE chat_id = ? AND is_reset = 1
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&id)
	return id.Int64
}

func encodeToolFields(msg llm.Message) (sql.NullString, sql.NullString) {
	var tcJSON sql.NullString
	if len(msg.ToolCalls) > 0 {
		if b, err := json.Marshal(msg.ToolCalls); err == nil {
			tcJSON = sql.NullString{String: string(b), Valid: true}
		}
	}
	var tcID sql.NullString
	if msg.ToolCallID != "" {
		tcID = sql.NullString{String: msg.ToolCallID, Valid: true}
	}
	return tcJSON, tcID
}

func scanMessages(rows *sql.Rows) []llm.Message {
	var msgs []llm.Message
	for rows.Next() {
		var m llm.Message
		var partsJSON, tcJSON, tcID sql.NullString
		if err := rows.Scan(&m.Role, &m.Content, &partsJSON, &tcJSON, &tcID); err != nil {
			continue
		}
		if partsJSON.Valid && partsJSON.String != "" {
			json.Unmarshal([]byte(partsJSON.String), &m.Parts) //nolint:errcheck
		}
		if tcJSON.Valid && tcJSON.String != "" {
			json.Unmarshal([]byte(tcJSON.String), &m.ToolCalls) //nolint:errcheck
		}
		m.ToolCallID = tcID.String
		msgs = append(msgs, m)
	}
	return msgs
}

func reverseMessages(msgs []llm.Message) []llm.Message {
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs
}
