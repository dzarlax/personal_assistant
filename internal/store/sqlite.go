package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sort"
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

CREATE TABLE IF NOT EXISTS model_capabilities (
    provider         TEXT    NOT NULL,
    model_id         TEXT    NOT NULL,
    vision           INTEGER NOT NULL DEFAULT 0,
    tools            INTEGER NOT NULL DEFAULT 0,
    reasoning        INTEGER NOT NULL DEFAULT 0,
    prompt_price     REAL    NOT NULL DEFAULT 0,
    completion_price REAL    NOT NULL DEFAULT 0,
    context_length   INTEGER NOT NULL DEFAULT 0,
    score            REAL    NOT NULL DEFAULT 0,
    fetched_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (provider, model_id)
);

CREATE TABLE IF NOT EXISTS kv_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS usage_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    provider             TEXT    NOT NULL,
    model_id             TEXT    NOT NULL,
    role                 TEXT    NOT NULL,
    chat_id              INTEGER NOT NULL DEFAULT 0,
    prompt_tokens        INTEGER NOT NULL DEFAULT 0,
    completion_tokens    INTEGER NOT NULL DEFAULT 0,
    cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens     INTEGER NOT NULL DEFAULT 0,
    latency_ms           INTEGER NOT NULL DEFAULT 0,
    success              INTEGER NOT NULL DEFAULT 1,
    error_class          TEXT    NOT NULL DEFAULT '',
    request_id           TEXT    NOT NULL DEFAULT '',
    tool_call_count      INTEGER NOT NULL DEFAULT 0,
    user_message_id      INTEGER,
    assistant_message_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_usage_ts_model ON usage_log(ts, model_id);
CREATE INDEX IF NOT EXISTS idx_usage_chat_ts  ON usage_log(chat_id, ts);
CREATE INDEX IF NOT EXISTS idx_usage_user_msg ON usage_log(user_message_id);
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
	// Migrations for existing databases
	db.Exec(`ALTER TABLE messages ADD COLUMN parts TEXT`)                              //nolint:errcheck
	db.Exec(`ALTER TABLE messages ADD COLUMN embedding BLOB`)                          //nolint:errcheck
	db.Exec(`ALTER TABLE model_capabilities ADD COLUMN score REAL NOT NULL DEFAULT 0`) //nolint:errcheck
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
	s.maybeSessionBreak(chatID, msg.Role)
	tcJSON, tcID := encodeToolFields(msg)
	partsJSON := encodePartsJSON(msg.Parts)
	_, err := s.db.Exec(`
		INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id)
		VALUES (?, ?, ?, ?, ?, ?)`,
		chatID, msg.Role, msg.Content, partsJSON, tcJSON, tcID)
	if err != nil {
		slog.Error("sqlite AddMessage failed", "chat_id", chatID, "role", msg.Role, "content_len", len(msg.Content), "err", err)
	}
}

// AddMessageWithEmbedding stores msg together with its pre-computed embedding.
// Only user messages need embeddings; other roles are stored with embedding=NULL.
func (s *SQLite) AddMessageWithEmbedding(chatID int64, msg llm.Message, emb []float32) {
	s.maybeSessionBreak(chatID, msg.Role)
	tcJSON, tcID := encodeToolFields(msg)
	partsJSON := encodePartsJSON(msg.Parts)
	var embBlob []byte
	if len(emb) > 0 {
		embBlob = floatsToBlob(emb)
	}
	_, err := s.db.Exec(`
		INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id, embedding)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		chatID, msg.Role, msg.Content, partsJSON, tcJSON, tcID, embBlob)
	if err != nil {
		slog.Error("sqlite AddMessageWithEmbedding failed", "chat_id", chatID, "role", msg.Role, "content_len", len(msg.Content), "err", err)
	}
}

// maybeSessionBreak inserts a reset marker when a user message arrives after a long idle period.
func (s *SQLite) maybeSessionBreak(chatID int64, role string) {
	if role != "user" {
		return
	}
	if last := s.lastMessageTime(chatID); !last.IsZero() && time.Since(last) > sessionIdleTimeout {
		slog.Info("auto session break", "chat_id", chatID, "idle", time.Since(last).Round(time.Minute))
		s.insertSessionBreak(chatID, "AUTO_SESSION_BREAK")
	}
}

// GetSemanticHistory returns the last recentN messages unconditionally, plus the
// top topK older conversational turns ranked by cosine similarity to queryEmb.
// Results are in chronological order.
func (s *SQLite) GetSemanticHistory(chatID int64, queryEmb []float32, recentN, topK int) []llm.Message {
	lastReset := s.lastResetID(chatID)
	rows, err := s.db.Query(`
		SELECT id, role, content, parts, tool_calls, tool_call_id, embedding
		FROM messages
		WHERE chat_id = ? AND id > ? AND is_compacted = 0 AND is_reset = 0
		ORDER BY id ASC`,
		chatID, lastReset)
	if err != nil {
		return nil
	}
	defer rows.Close()

	type msgRow struct {
		id  int64
		msg llm.Message
		emb []float32
	}
	var all []msgRow
	for rows.Next() {
		var r msgRow
		var partsJSON, tcJSON, tcID sql.NullString
		var embBlob []byte
		if err := rows.Scan(&r.id, &r.msg.Role, &r.msg.Content, &partsJSON, &tcJSON, &tcID, &embBlob); err != nil {
			continue
		}
		if partsJSON.Valid && partsJSON.String != "" {
			json.Unmarshal([]byte(partsJSON.String), &r.msg.Parts) //nolint:errcheck
		}
		if tcJSON.Valid && tcJSON.String != "" {
			json.Unmarshal([]byte(tcJSON.String), &r.msg.ToolCalls) //nolint:errcheck
		}
		r.msg.ToolCallID = tcID.String
		if len(embBlob) > 0 {
			r.emb = blobToFloats(embBlob)
		}
		all = append(all, r)
	}

	if len(all) <= recentN {
		msgs := make([]llm.Message, len(all))
		for i, r := range all {
			msgs[i] = r.msg
		}
		return msgs
	}

	recentStart := len(all) - recentN
	older := all[:recentStart]
	recent := all[recentStart:]

	// Group older messages into conversational turns.
	// A turn begins at each user message and ends before the next one.
	type turn struct {
		rows  []msgRow
		score float64
	}
	var turns []turn
	var cur []msgRow
	for _, r := range older {
		if r.msg.Role == "user" && len(cur) > 0 {
			turns = append(turns, turn{rows: cur})
			cur = nil
		}
		cur = append(cur, r)
	}
	if len(cur) > 0 {
		turns = append(turns, turn{rows: cur})
	}

	// Score each turn by the cosine similarity of its user message embedding.
	for i := range turns {
		for _, r := range turns[i].rows {
			if r.msg.Role == "user" && len(r.emb) > 0 {
				turns[i].score = cosineSimilarityF32(queryEmb, r.emb)
				break
			}
		}
	}

	// Select top-K turns by score.
	sort.Slice(turns, func(i, j int) bool { return turns[i].score > turns[j].score })
	if len(turns) > topK {
		turns = turns[:topK]
	}
	// Restore chronological order.
	sort.Slice(turns, func(i, j int) bool { return turns[i].rows[0].id < turns[j].rows[0].id })

	var result []llm.Message
	for _, t := range turns {
		for _, r := range t.rows {
			result = append(result, r.msg)
		}
	}
	for _, r := range recent {
		result = append(result, r.msg)
	}
	return result
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
		SELECT id, role, content, parts, tool_calls, tool_call_id, embedding
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
		var embBlob []byte
		if err := rows.Scan(&row.ID, &row.Message.Role, &row.Message.Content, &partsJSON, &tcJSON, &tcID, &embBlob); err != nil {
			continue
		}
		if partsJSON.Valid && partsJSON.String != "" {
			json.Unmarshal([]byte(partsJSON.String), &row.Message.Parts) //nolint:errcheck
		}
		if tcJSON.Valid && tcJSON.String != "" {
			json.Unmarshal([]byte(tcJSON.String), &row.Message.ToolCalls) //nolint:errcheck
		}
		row.Message.ToolCallID = tcID.String
		if len(embBlob) > 0 {
			row.Embedding = blobToFloats(embBlob)
		}
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

func (s *SQLite) GetStats(chatID int64) ChatStats {
	lastReset := s.lastResetID(chatID)

	var stats ChatStats

	// Active message count and char size in current session
	s.db.QueryRow(`
		SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0)
		FROM messages
		WHERE chat_id = ? AND id > ? AND is_compacted = 0 AND is_reset = 0`,
		chatID, lastReset).Scan(&stats.ActiveMessages, &stats.ActiveChars)

	// Last compaction timestamp
	var lastCompact sql.NullString
	s.db.QueryRow(`
		SELECT MAX(created_at) FROM messages
		WHERE chat_id = ? AND is_compacted = 1`,
		chatID).Scan(&lastCompact)
	if lastCompact.Valid && lastCompact.String != "" {
		stats.LastCompactAt, _ = time.Parse("2006-01-02 15:04:05", lastCompact.String)
	}

	// Last message timestamp
	var lastMsg sql.NullString
	s.db.QueryRow(`
		SELECT MAX(created_at) FROM messages
		WHERE chat_id = ? AND is_reset = 0 AND role != 'system'`,
		chatID).Scan(&lastMsg)
	if lastMsg.Valid && lastMsg.String != "" {
		stats.LastMessageAt, _ = time.Parse("2006-01-02 15:04:05", lastMsg.String)
	}

	return stats
}

// SearchAllSessions searches the entire message history (across all sessions) for
// turns semantically similar to queryEmb. Turns from the current session are excluded.
// Each result is a (date, userText, botText) snippet truncated for system-prompt injection.
func (s *SQLite) SearchAllSessions(chatID int64, queryEmb []float32, topK int, minScore float64) []HistorySnippet {
	lastReset := s.lastResetID(chatID)

	rows, err := s.db.Query(`
		SELECT id, role, content, created_at, embedding
		FROM messages
		WHERE chat_id = ? AND id <= ? AND is_compacted = 0 AND is_reset = 0
		      AND role IN ('user','assistant') AND is_summary = 0
		ORDER BY id ASC`,
		chatID, lastReset)
	if err != nil {
		return nil
	}
	defer rows.Close()

	type rawRow struct {
		id        int64
		role      string
		content   string
		createdAt string
		emb       []float32
	}
	var all []rawRow
	for rows.Next() {
		var r rawRow
		var embBlob []byte
		if err := rows.Scan(&r.id, &r.role, &r.content, &r.createdAt, &embBlob); err != nil {
			continue
		}
		if len(embBlob) > 0 {
			r.emb = blobToFloats(embBlob)
		}
		all = append(all, r)
	}
	if len(all) == 0 {
		return nil
	}

	// Group into turns: user message followed by the next assistant response.
	type turn struct {
		date      time.Time
		userText  string
		botText   string
		userEmb   []float32
		score     float64
	}
	var turns []turn
	for i := 0; i < len(all); i++ {
		if all[i].role != "user" {
			continue
		}
		t := turn{
			userText: all[i].content,
			userEmb:  all[i].emb,
		}
		t.date, _ = time.Parse("2006-01-02 15:04:05", all[i].createdAt)
		// Find the next assistant message in this turn.
		for j := i + 1; j < len(all) && all[j].role != "user"; j++ {
			if all[j].role == "assistant" {
				t.botText = all[j].content
				break
			}
		}
		turns = append(turns, t)
	}

	// Score each turn.
	for i := range turns {
		if len(turns[i].userEmb) > 0 {
			turns[i].score = cosineSimilarityF32(queryEmb, turns[i].userEmb)
		}
	}

	// Filter by minScore, sort by score descending, take topK.
	var candidates []turn
	for _, t := range turns {
		if t.score >= minScore {
			candidates = append(candidates, t)
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	// Sort results chronologically before returning.
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].date.Before(candidates[j].date) })

	snippets := make([]HistorySnippet, len(candidates))
	for i, t := range candidates {
		snippets[i] = HistorySnippet{
			Date:     t.date,
			UserText: t.userText,
			BotText:  t.botText,
		}
	}
	return snippets
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

func encodePartsJSON(parts []llm.ContentPart) sql.NullString {
	if len(parts) == 0 {
		return sql.NullString{}
	}
	b, err := json.Marshal(parts)
	if err != nil {
		return sql.NullString{}
	}
	return sql.NullString{String: string(b), Valid: true}
}

func floatsToBlob(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func blobToFloats(b []byte) []float32 {
	n := len(b) / 4
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func cosineSimilarityF32(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// --- CapabilityStore implementation ---

// GetCapabilities returns cached capabilities for (provider, modelID) or (_, false, nil) when absent.
func (s *SQLite) GetCapabilities(ctx context.Context, provider, modelID string) (llm.Capabilities, bool, error) {
	row := s.db.QueryRowContext(ctx, `
		SELECT vision, tools, reasoning, prompt_price, completion_price, context_length, score
		FROM model_capabilities WHERE provider = ? AND model_id = ?`,
		provider, modelID)
	var c llm.Capabilities
	var vision, tools, reasoning int
	err := row.Scan(&vision, &tools, &reasoning, &c.PromptPrice, &c.CompletionPrice, &c.ContextLength, &c.Score)
	if err == sql.ErrNoRows {
		return llm.Capabilities{}, false, nil
	}
	if err != nil {
		return llm.Capabilities{}, false, fmt.Errorf("get capabilities: %w", err)
	}
	c.Vision = vision == 1
	c.Tools = tools == 1
	c.Reasoning = reasoning == 1
	return c, true, nil
}

// PutCapabilities upserts capabilities for (provider, modelID) and updates fetched_at.
func (s *SQLite) PutCapabilities(ctx context.Context, provider, modelID string, c llm.Capabilities) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO model_capabilities
			(provider, model_id, vision, tools, reasoning, prompt_price, completion_price, context_length, score, fetched_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(provider, model_id) DO UPDATE SET
			vision = excluded.vision,
			tools = excluded.tools,
			reasoning = excluded.reasoning,
			prompt_price = excluded.prompt_price,
			completion_price = excluded.completion_price,
			context_length = excluded.context_length,
			score = excluded.score,
			fetched_at = CURRENT_TIMESTAMP`,
		provider, modelID, boolToInt(c.Vision), boolToInt(c.Tools), boolToInt(c.Reasoning),
		c.PromptPrice, c.CompletionPrice, c.ContextLength, c.Score)
	if err != nil {
		return fmt.Errorf("put capabilities: %w", err)
	}
	return nil
}

// GetAllCapabilities returns all known capabilities for the given provider.
func (s *SQLite) GetAllCapabilities(ctx context.Context, provider string) (map[string]llm.Capabilities, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT model_id, vision, tools, reasoning, prompt_price, completion_price, context_length, score
		FROM model_capabilities WHERE provider = ?`, provider)
	if err != nil {
		return nil, fmt.Errorf("list capabilities: %w", err)
	}
	defer rows.Close()
	out := make(map[string]llm.Capabilities)
	for rows.Next() {
		var id string
		var c llm.Capabilities
		var vision, tools, reasoning int
		if err := rows.Scan(&id, &vision, &tools, &reasoning, &c.PromptPrice, &c.CompletionPrice, &c.ContextLength, &c.Score); err != nil {
			return nil, err
		}
		c.Vision = vision == 1
		c.Tools = tools == 1
		c.Reasoning = reasoning == 1
		out[id] = c
	}
	return out, rows.Err()
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// --- SettingsStore implementation ---

func (s *SQLite) GetSetting(ctx context.Context, key string) (string, bool, error) {
	var v string
	err := s.db.QueryRowContext(ctx,
		`SELECT value FROM kv_settings WHERE key = ?`, key).Scan(&v)
	if err == sql.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("get setting: %w", err)
	}
	return v, true, nil
}

func (s *SQLite) PutSetting(ctx context.Context, key, value string) error {
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO kv_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP`,
		key, value)
	if err != nil {
		return fmt.Errorf("put setting: %w", err)
	}
	return nil
}

// --- UsageStore implementation ---

func (s *SQLite) PutUsage(ctx context.Context, u llm.UsageLog) (int64, error) {
	res, err := s.db.ExecContext(ctx, `
		INSERT INTO usage_log (
			provider, model_id, role, chat_id,
			prompt_tokens, completion_tokens, cached_prompt_tokens, reasoning_tokens,
			latency_ms, success, error_class, request_id, tool_call_count,
			user_message_id, assistant_message_id
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		u.Provider, u.ModelID, u.Role, u.ChatID,
		u.PromptTokens, u.CompletionTokens, u.CachedPromptTokens, u.ReasoningTokens,
		u.LatencyMs, boolToInt(u.Success), u.ErrorClass, u.RequestID, u.ToolCallCount,
		nullableInt64(u.UserMessageID), nullableInt64(u.AssistantMessageID))
	if err != nil {
		return 0, fmt.Errorf("put usage: %w", err)
	}
	return res.LastInsertId()
}

func (s *SQLite) UpdateAssistantMessageID(ctx context.Context, usageID, msgID int64) error {
	_, err := s.db.ExecContext(ctx,
		`UPDATE usage_log SET assistant_message_id = ? WHERE id = ?`, msgID, usageID)
	if err != nil {
		return fmt.Errorf("update usage assistant_message_id: %w", err)
	}
	return nil
}

func nullableInt64(p *int64) any {
	if p == nil {
		return nil
	}
	return *p
}
