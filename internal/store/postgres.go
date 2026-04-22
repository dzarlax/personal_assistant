package store

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"telegram-agent/internal/llm"
)

const (
	pgMaxHistory = 30
)

// Postgres implements SemanticStore backed by PostgreSQL via pgx.
type Postgres struct {
	pool *pgxpool.Pool
}

// NewPostgres creates a new PostgreSQL-backed store.
// connString example: "postgres://assistant_user:pass@localhost:5432/aistack?search_path=assistant"
func NewPostgres(ctx context.Context, connString string) (*Postgres, error) {
	config, err := pgxpool.ParseConfig(connString)
	if err != nil {
		return nil, fmt.Errorf("parse pg config: %w", err)
	}
	config.MaxConns = 10
	config.MinConns = 2

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("connect pg: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping pg: %w", err)
	}

	p := &Postgres{pool: pool}
	if err := p.migrate(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pg migrate: %w", err)
	}
	return p, nil
}

// migrate creates tables this binary owns (idempotent).
// The main messages table is assumed to exist — it's maintained externally.
func (p *Postgres) migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS model_capabilities (
			provider         TEXT        NOT NULL,
			model_id         TEXT        NOT NULL,
			vision           BOOLEAN     NOT NULL DEFAULT FALSE,
			tools            BOOLEAN     NOT NULL DEFAULT FALSE,
			reasoning        BOOLEAN     NOT NULL DEFAULT FALSE,
			prompt_price     DOUBLE PRECISION NOT NULL DEFAULT 0,
			completion_price DOUBLE PRECISION NOT NULL DEFAULT 0,
			context_length   INTEGER     NOT NULL DEFAULT 0,
			score            DOUBLE PRECISION NOT NULL DEFAULT 0,
			fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY (provider, model_id)
		)`,
		`ALTER TABLE model_capabilities ADD COLUMN IF NOT EXISTS score DOUBLE PRECISION NOT NULL DEFAULT 0`,
		`CREATE TABLE IF NOT EXISTS kv_settings (
			key        TEXT PRIMARY KEY,
			value      TEXT NOT NULL,
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`,
		`CREATE TABLE IF NOT EXISTS usage_log (
			id                   BIGSERIAL PRIMARY KEY,
			ts                   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			provider             TEXT        NOT NULL,
			model_id             TEXT        NOT NULL,
			role                 TEXT        NOT NULL,
			chat_id              BIGINT      NOT NULL DEFAULT 0,
			prompt_tokens        INTEGER     NOT NULL DEFAULT 0,
			completion_tokens    INTEGER     NOT NULL DEFAULT 0,
			cached_prompt_tokens INTEGER     NOT NULL DEFAULT 0,
			reasoning_tokens     INTEGER     NOT NULL DEFAULT 0,
			cost_usd             DOUBLE PRECISION NOT NULL DEFAULT 0,
			latency_ms           INTEGER     NOT NULL DEFAULT 0,
			success              BOOLEAN     NOT NULL DEFAULT TRUE,
			error_class          TEXT        NOT NULL DEFAULT '',
			request_id           TEXT        NOT NULL DEFAULT '',
			tool_call_count      INTEGER     NOT NULL DEFAULT 0,
			user_message_id      BIGINT,
			assistant_message_id BIGINT
		)`,
		`ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0`,
		`ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS turn_latency_ms INTEGER NOT NULL DEFAULT 0`,
		`CREATE INDEX IF NOT EXISTS idx_usage_ts_model ON usage_log(ts, model_id)`,
		`CREATE INDEX IF NOT EXISTS idx_usage_chat_ts  ON usage_log(chat_id, ts)`,
		`CREATE INDEX IF NOT EXISTS idx_usage_user_msg ON usage_log(user_message_id)`,
		`CREATE TABLE IF NOT EXISTS tool_call_log (
			id          BIGSERIAL PRIMARY KEY,
			ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			chat_id     BIGINT      NOT NULL DEFAULT 0,
			user_msg_id BIGINT,
			tool_name   TEXT        NOT NULL,
			latency_ms  INTEGER     NOT NULL DEFAULT 0,
			success     BOOLEAN     NOT NULL DEFAULT TRUE,
			error_text  TEXT        NOT NULL DEFAULT ''
		)`,
		`CREATE INDEX IF NOT EXISTS idx_tool_call_chat_ts ON tool_call_log(chat_id, ts)`,
	}
	for _, s := range stmts {
		if _, err := p.pool.Exec(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// --- SettingsStore implementation ---

func (p *Postgres) GetSetting(ctx context.Context, key string) (string, bool, error) {
	var v string
	err := p.pool.QueryRow(ctx, `SELECT value FROM kv_settings WHERE key = $1`, key).Scan(&v)
	if err == pgx.ErrNoRows {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("get setting: %w", err)
	}
	return v, true, nil
}

func (p *Postgres) PutSetting(ctx context.Context, key, value string) error {
	_, err := p.pool.Exec(ctx, `
		INSERT INTO kv_settings (key, value, updated_at) VALUES ($1, $2, NOW())
		ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()`,
		key, value)
	if err != nil {
		return fmt.Errorf("put setting: %w", err)
	}
	return nil
}

// --- CapabilityStore implementation ---

func (p *Postgres) GetCapabilities(ctx context.Context, provider, modelID string) (llm.Capabilities, bool, error) {
	var c llm.Capabilities
	err := p.pool.QueryRow(ctx, `
		SELECT vision, tools, reasoning, prompt_price, completion_price, context_length, score
		FROM model_capabilities WHERE provider = $1 AND model_id = $2`,
		provider, modelID).Scan(&c.Vision, &c.Tools, &c.Reasoning,
		&c.PromptPrice, &c.CompletionPrice, &c.ContextLength, &c.Score)
	if err == pgx.ErrNoRows {
		return llm.Capabilities{}, false, nil
	}
	if err != nil {
		return llm.Capabilities{}, false, fmt.Errorf("get capabilities: %w", err)
	}
	return c, true, nil
}

func (p *Postgres) PutCapabilities(ctx context.Context, provider, modelID string, c llm.Capabilities) error {
	_, err := p.pool.Exec(ctx, `
		INSERT INTO model_capabilities
			(provider, model_id, vision, tools, reasoning, prompt_price, completion_price, context_length, score, fetched_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
		ON CONFLICT (provider, model_id) DO UPDATE SET
			vision           = EXCLUDED.vision,
			tools            = EXCLUDED.tools,
			reasoning        = EXCLUDED.reasoning,
			prompt_price     = EXCLUDED.prompt_price,
			completion_price = EXCLUDED.completion_price,
			context_length   = EXCLUDED.context_length,
			score            = EXCLUDED.score,
			fetched_at       = NOW()`,
		provider, modelID, c.Vision, c.Tools, c.Reasoning,
		c.PromptPrice, c.CompletionPrice, c.ContextLength, c.Score)
	if err != nil {
		return fmt.Errorf("put capabilities: %w", err)
	}
	return nil
}

func (p *Postgres) GetAllCapabilities(ctx context.Context, provider string) (map[string]llm.Capabilities, error) {
	rows, err := p.pool.Query(ctx, `
		SELECT model_id, vision, tools, reasoning, prompt_price, completion_price, context_length, score
		FROM model_capabilities WHERE provider = $1`, provider)
	if err != nil {
		return nil, fmt.Errorf("list capabilities: %w", err)
	}
	defer rows.Close()
	out := make(map[string]llm.Capabilities)
	for rows.Next() {
		var id string
		var c llm.Capabilities
		if err := rows.Scan(&id, &c.Vision, &c.Tools, &c.Reasoning,
			&c.PromptPrice, &c.CompletionPrice, &c.ContextLength, &c.Score); err != nil {
			return nil, err
		}
		out[id] = c
	}
	return out, rows.Err()
}

// Close releases the connection pool.
func (p *Postgres) Close() {
	p.pool.Close()
}

func (p *Postgres) GetHistory(chatID int64) []llm.Message {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)

	rows, err := p.pool.Query(ctx, `
		SELECT role, content, parts, tool_calls, tool_call_id
		FROM messages
		WHERE chat_id = $1 AND id > $2 AND is_compacted = FALSE AND is_reset = FALSE
		ORDER BY id DESC LIMIT $3`,
		chatID, lastReset, pgMaxHistory)
	if err != nil {
		return nil
	}
	defer rows.Close()
	return reverseMessages(scanMessagesPg(rows))
}

func (p *Postgres) AddMessage(chatID int64, msg llm.Message) int64 {
	ctx := context.Background()
	p.maybeSessionBreak(ctx, chatID, msg.Role)
	tcJSON, tcID := encodeToolFields(msg)
	partsJSON := encodePartsJSON(msg.Parts)

	var id int64
	err := p.pool.QueryRow(ctx, `
		INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id`,
		chatID, msg.Role, msg.Content, partsJSON, tcJSON, tcID).Scan(&id)
	if err != nil {
		slog.Error("pg AddMessage failed", "chat_id", chatID, "role", msg.Role, "content_len", len(msg.Content), "err", err)
		return 0
	}
	return id
}

func (p *Postgres) AddMessageWithEmbedding(chatID int64, msg llm.Message, emb []float32) int64 {
	ctx := context.Background()
	p.maybeSessionBreak(ctx, chatID, msg.Role)
	tcJSON, tcID := encodeToolFields(msg)
	partsJSON := encodePartsJSON(msg.Parts)

	var embBlob []byte
	if len(emb) > 0 {
		embBlob = floatsToBlob(emb)
	}

	var id int64
	err := p.pool.QueryRow(ctx, `
		INSERT INTO messages (chat_id, role, content, parts, tool_calls, tool_call_id, embedding)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id`,
		chatID, msg.Role, msg.Content, partsJSON, tcJSON, tcID, embBlob).Scan(&id)
	if err != nil {
		slog.Error("pg AddMessageWithEmbedding failed", "chat_id", chatID, "role", msg.Role, "content_len", len(msg.Content), "err", err)
		return 0
	}
	return id
}

func (p *Postgres) maybeSessionBreak(ctx context.Context, chatID int64, role string) {
	if role != "user" {
		return
	}
	if last := p.lastMessageTime(ctx, chatID); !last.IsZero() && time.Since(last) > sessionIdleTimeout {
		slog.Info("auto session break", "chat_id", chatID, "idle", time.Since(last).Round(time.Minute))
		p.insertSessionBreak(ctx, chatID, "AUTO_SESSION_BREAK")
	}
}

func (p *Postgres) GetSemanticHistory(chatID int64, queryEmb []float32, recentN, topK int) []llm.Message {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)

	rows, err := p.pool.Query(ctx, `
		SELECT id, role, content, parts, tool_calls, tool_call_id, embedding
		FROM messages
		WHERE chat_id = $1 AND id > $2 AND is_compacted = FALSE AND is_reset = FALSE
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
		var partsJSON, tcJSON, tcID *string
		var embBlob []byte
		if err := rows.Scan(&r.id, &r.msg.Role, &r.msg.Content, &partsJSON, &tcJSON, &tcID, &embBlob); err != nil {
			continue
		}
		if partsJSON != nil && *partsJSON != "" {
			json.Unmarshal([]byte(*partsJSON), &r.msg.Parts) //nolint:errcheck
		}
		if tcJSON != nil && *tcJSON != "" {
			json.Unmarshal([]byte(*tcJSON), &r.msg.ToolCalls) //nolint:errcheck
		}
		if tcID != nil {
			r.msg.ToolCallID = *tcID
		}
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

	for i := range turns {
		for _, r := range turns[i].rows {
			if r.msg.Role == "user" && len(r.emb) > 0 {
				turns[i].score = cosineSimilarityF32(queryEmb, r.emb)
				break
			}
		}
	}

	sort.Slice(turns, func(i, j int) bool { return turns[i].score > turns[j].score })
	if len(turns) > topK {
		turns = turns[:topK]
	}
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

func (p *Postgres) lastMessageTime(ctx context.Context, chatID int64) time.Time {
	var ts *time.Time
	p.pool.QueryRow(ctx, `
		SELECT created_at FROM messages
		WHERE chat_id = $1 AND is_reset = FALSE
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&ts)
	if ts == nil {
		return time.Time{}
	}
	return *ts
}

func (p *Postgres) ClearHistory(chatID int64) {
	ctx := context.Background()
	p.pool.Exec(ctx, `INSERT INTO messages (chat_id, role, content, is_reset) VALUES ($1, 'system', 'CONTEXT_RESET', TRUE)`, chatID) //nolint:errcheck
}

// DisplayHistory returns the last `limit` non-compacted rows including
// is_reset session-break markers. Used by the admin web UI to show the
// full recent conversation with dividers between sessions.
func (p *Postgres) DisplayHistory(chatID int64, limit, offset int) []HistoryItem {
	if limit <= 0 {
		limit = 200
	}
	if offset < 0 {
		offset = 0
	}
	ctx := context.Background()
	rows, err := p.pool.Query(ctx, `
		SELECT role, content, parts, is_reset, created_at
		FROM messages
		WHERE chat_id = $1 AND is_compacted = FALSE
		ORDER BY id DESC LIMIT $2 OFFSET $3`,
		chatID, limit, offset)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var out []HistoryItem
	for rows.Next() {
		var role, content string
		var partsJSON *string
		var isReset bool
		var createdAt time.Time
		if err := rows.Scan(&role, &content, &partsJSON, &isReset, &createdAt); err != nil {
			continue
		}
		item := HistoryItem{Content: content, CreatedAt: createdAt}
		if isReset {
			item.Role = "break"
			out = append(out, item)
			continue
		}
		item.Role = role
		if partsJSON != nil && *partsJSON != "" {
			var parts []llm.ContentPart
			if err := json.Unmarshal([]byte(*partsJSON), &parts); err == nil {
				var textBuf strings.Builder
				for _, pt := range parts {
					switch pt.Type {
					case "text":
						if textBuf.Len() > 0 {
							textBuf.WriteByte('\n')
						}
						textBuf.WriteString(pt.Text)
					case "image_url":
						if pt.ImageURL != nil && pt.ImageURL.URL != "" {
							item.ImageURLs = append(item.ImageURLs, pt.ImageURL.URL)
						}
					}
				}
				if content == "" {
					item.Content = textBuf.String()
				}
			}
		}
		out = append(out, item)
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// LastUserMessage returns the most recent user-role row within the current
// session (id > last reset). Used by admin chat for Regenerate / Edit.
func (p *Postgres) LastUserMessage(chatID int64) (int64, string, bool) {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)
	var id int64
	var content string
	err := p.pool.QueryRow(ctx, `
		SELECT id, content FROM messages
		WHERE chat_id = $1 AND id > $2 AND role = 'user' AND is_compacted = FALSE
		ORDER BY id DESC LIMIT 1`,
		chatID, lastReset).Scan(&id, &content)
	if err != nil {
		return 0, "", false
	}
	return id, content, true
}

// TruncateAfter deletes every row with id >= fromID within chatID.
func (p *Postgres) TruncateAfter(chatID int64, fromID int64) error {
	ctx := context.Background()
	_, err := p.pool.Exec(ctx, `DELETE FROM messages WHERE chat_id = $1 AND id >= $2`, chatID, fromID)
	return err
}

func (p *Postgres) insertSessionBreak(ctx context.Context, chatID int64, reason string) {
	p.pool.Exec(ctx, `INSERT INTO messages (chat_id, role, content, is_reset) VALUES ($1, 'system', $2, TRUE)`, chatID, reason) //nolint:errcheck

	var summary *string
	p.pool.QueryRow(ctx, `
		SELECT content FROM messages
		WHERE chat_id = $1 AND is_summary = TRUE
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&summary)
	if summary != nil && *summary != "" {
		p.pool.Exec(ctx, `INSERT INTO messages (chat_id, role, content, is_summary) VALUES ($1, 'assistant', $2, TRUE)`, chatID, *summary) //nolint:errcheck
	}
}

func (p *Postgres) GetAllActive(chatID int64) ([]MessageRow, error) {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)

	rows, err := p.pool.Query(ctx, `
		SELECT id, role, content, parts, tool_calls, tool_call_id, embedding
		FROM messages
		WHERE chat_id = $1 AND id > $2 AND is_compacted = FALSE AND is_reset = FALSE
		ORDER BY id ASC`,
		chatID, lastReset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []MessageRow
	for rows.Next() {
		var row MessageRow
		var partsJSON, tcJSON, tcID *string
		var embBlob []byte
		if err := rows.Scan(&row.ID, &row.Message.Role, &row.Message.Content, &partsJSON, &tcJSON, &tcID, &embBlob); err != nil {
			continue
		}
		if partsJSON != nil && *partsJSON != "" {
			json.Unmarshal([]byte(*partsJSON), &row.Message.Parts) //nolint:errcheck
		}
		if tcJSON != nil && *tcJSON != "" {
			json.Unmarshal([]byte(*tcJSON), &row.Message.ToolCalls) //nolint:errcheck
		}
		if tcID != nil {
			row.Message.ToolCallID = *tcID
		}
		if len(embBlob) > 0 {
			row.Embedding = blobToFloats(embBlob)
		}
		result = append(result, row)
	}
	return result, rows.Err()
}

func (p *Postgres) AddSummary(chatID int64, content string) {
	ctx := context.Background()
	p.pool.Exec(ctx, `
		INSERT INTO messages (chat_id, role, content, is_summary)
		VALUES ($1, 'assistant', $2, TRUE)`,
		chatID, content) //nolint:errcheck
}

func (p *Postgres) MarkCompacted(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	// Build $1, $2, $3... placeholders
	placeholders := make([]string, len(ids))
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
		args[i] = id
	}
	_, err := p.pool.Exec(context.Background(),
		"UPDATE messages SET is_compacted = TRUE WHERE id IN ("+strings.Join(placeholders, ",")+`)`,
		args...,
	)
	return err
}

func (p *Postgres) GetStats(chatID int64) ChatStats {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)

	var stats ChatStats

	p.pool.QueryRow(ctx, `
		SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0)
		FROM messages
		WHERE chat_id = $1 AND id > $2 AND is_compacted = FALSE AND is_reset = FALSE`,
		chatID, lastReset).Scan(&stats.ActiveMessages, &stats.ActiveChars) //nolint:errcheck

	var lastCompact *time.Time
	p.pool.QueryRow(ctx, `
		SELECT MAX(created_at) FROM messages
		WHERE chat_id = $1 AND is_compacted = TRUE`,
		chatID).Scan(&lastCompact) //nolint:errcheck
	if lastCompact != nil {
		stats.LastCompactAt = *lastCompact
	}

	var lastMsg *time.Time
	p.pool.QueryRow(ctx, `
		SELECT MAX(created_at) FROM messages
		WHERE chat_id = $1 AND is_reset = FALSE AND role != 'system'`,
		chatID).Scan(&lastMsg) //nolint:errcheck
	if lastMsg != nil {
		stats.LastMessageAt = *lastMsg
	}

	return stats
}

func (p *Postgres) SearchAllSessions(chatID int64, queryEmb []float32, topK int, minScore float64) []HistorySnippet {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)

	rows, err := p.pool.Query(ctx, `
		SELECT id, role, content, created_at, embedding
		FROM messages
		WHERE chat_id = $1 AND id <= $2 AND is_compacted = FALSE AND is_reset = FALSE
		      AND role IN ('user','assistant') AND is_summary = FALSE
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
		createdAt time.Time
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

	type turn struct {
		date    time.Time
		userText string
		botText  string
		userEmb  []float32
		score    float64
	}
	var turns []turn
	for i := 0; i < len(all); i++ {
		if all[i].role != "user" {
			continue
		}
		t := turn{
			userText: all[i].content,
			userEmb:  all[i].emb,
			date:     all[i].createdAt,
		}
		for j := i + 1; j < len(all) && all[j].role != "user"; j++ {
			if all[j].role == "assistant" {
				t.botText = all[j].content
				break
			}
		}
		turns = append(turns, t)
	}

	for i := range turns {
		if len(turns[i].userEmb) > 0 {
			turns[i].score = cosineSimilarityF32(queryEmb, turns[i].userEmb)
		}
	}

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

func (p *Postgres) ActiveCharCount(chatID int64) int {
	ctx := context.Background()
	lastReset := p.lastResetID(ctx, chatID)
	var total *int64
	p.pool.QueryRow(ctx, `
		SELECT SUM(LENGTH(content))
		FROM messages
		WHERE chat_id = $1 AND id > $2 AND is_compacted = FALSE AND is_reset = FALSE`,
		chatID, lastReset).Scan(&total)
	if total == nil {
		return 0
	}
	return int(*total)
}

func (p *Postgres) lastResetID(ctx context.Context, chatID int64) int64 {
	var id *int64
	p.pool.QueryRow(ctx, `
		SELECT id FROM messages
		WHERE chat_id = $1 AND is_reset = TRUE
		ORDER BY id DESC LIMIT 1`,
		chatID).Scan(&id)
	if id == nil {
		return 0
	}
	return *id
}

// scanMessagesPg scans pgx rows into messages.
func scanMessagesPg(rows pgx.Rows) []llm.Message {
	var msgs []llm.Message
	for rows.Next() {
		var m llm.Message
		var partsJSON, tcJSON, tcID *string
		if err := rows.Scan(&m.Role, &m.Content, &partsJSON, &tcJSON, &tcID); err != nil {
			continue
		}
		if partsJSON != nil && *partsJSON != "" {
			json.Unmarshal([]byte(*partsJSON), &m.Parts) //nolint:errcheck
		}
		if tcJSON != nil && *tcJSON != "" {
			json.Unmarshal([]byte(*tcJSON), &m.ToolCalls) //nolint:errcheck
		}
		if tcID != nil {
			m.ToolCallID = *tcID
		}
		msgs = append(msgs, m)
	}
	return msgs
}

// --- UsageStore implementation ---

func (p *Postgres) PutUsage(ctx context.Context, u llm.UsageLog) (int64, error) {
	var id int64
	err := p.pool.QueryRow(ctx, `
		INSERT INTO usage_log (
			provider, model_id, role, chat_id,
			prompt_tokens, completion_tokens, cached_prompt_tokens, reasoning_tokens,
			cost_usd, latency_ms, turn_latency_ms, success, error_class, request_id, tool_call_count,
			user_message_id, assistant_message_id
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
		RETURNING id`,
		u.Provider, u.ModelID, u.Role, u.ChatID,
		u.PromptTokens, u.CompletionTokens, u.CachedPromptTokens, u.ReasoningTokens,
		u.Cost, u.LatencyMs, u.TurnLatencyMs, u.Success, u.ErrorClass, u.RequestID, u.ToolCallCount,
		u.UserMessageID, u.AssistantMessageID,
	).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("put usage: %w", err)
	}
	return id, nil
}

func (p *Postgres) UpdateAssistantMessageID(ctx context.Context, usageID, msgID int64) error {
	_, err := p.pool.Exec(ctx,
		`UPDATE usage_log SET assistant_message_id = $1 WHERE id = $2`, msgID, usageID)
	if err != nil {
		return fmt.Errorf("update usage assistant_message_id: %w", err)
	}
	return nil
}

func (p *Postgres) UpdateTurnLatencyMs(ctx context.Context, usageID int64, ms int) error {
	_, err := p.pool.Exec(ctx,
		`UPDATE usage_log SET turn_latency_ms = $1 WHERE id = $2`, ms, usageID)
	if err != nil {
		return fmt.Errorf("update usage turn_latency_ms: %w", err)
	}
	return nil
}

func (p *Postgres) PutToolCall(ctx context.Context, t llm.ToolCallLog) error {
	_, err := p.pool.Exec(ctx, `
		INSERT INTO tool_call_log (chat_id, user_msg_id, tool_name, latency_ms, success, error_text)
		VALUES ($1, $2, $3, $4, $5, $6)`,
		t.ChatID, t.UserMsgID, t.ToolName, t.LatencyMs, t.Success, t.ErrorText)
	if err != nil {
		return fmt.Errorf("put tool call: %w", err)
	}
	return nil
}

func (p *Postgres) UsageTotals(ctx context.Context, since time.Time) (llm.UsageTotals, error) {
	var t llm.UsageTotals
	err := p.pool.QueryRow(ctx, `
		SELECT COALESCE(COUNT(*), 0),
		       COALESCE(SUM(prompt_tokens), 0),
		       COALESCE(SUM(completion_tokens), 0),
		       COALESCE(SUM(cached_prompt_tokens), 0),
		       COALESCE(SUM(reasoning_tokens), 0),
		       COALESCE(SUM(cost_usd), 0),
		       COALESCE(SUM(CASE WHEN success THEN 0 ELSE 1 END), 0)
		FROM usage_log WHERE ts >= $1`, since).
		Scan(&t.Calls, &t.PromptTokens, &t.CompletionTokens, &t.CachedPromptTokens,
			&t.ReasoningTokens, &t.CostUSD, &t.ErrorCount)
	if err != nil {
		return llm.UsageTotals{}, fmt.Errorf("usage totals: %w", err)
	}
	return t, nil
}

func (p *Postgres) UsageByDay(ctx context.Context, since time.Time) ([]llm.UsageDayBucket, error) {
	rows, err := p.pool.Query(ctx, `
		SELECT date_trunc('day', ts) AS day,
		       COUNT(*),
		       COALESCE(SUM(prompt_tokens), 0),
		       COALESCE(SUM(completion_tokens), 0),
		       COALESCE(SUM(cost_usd), 0)
		FROM usage_log WHERE ts >= $1
		GROUP BY day ORDER BY day`, since)
	if err != nil {
		return nil, fmt.Errorf("usage by day: %w", err)
	}
	defer rows.Close()
	var out []llm.UsageDayBucket
	for rows.Next() {
		var b llm.UsageDayBucket
		if err := rows.Scan(&b.Day, &b.Calls, &b.PromptTokens, &b.CompletionTokens, &b.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

func (p *Postgres) UsageByModel(ctx context.Context, since time.Time, limit int) ([]llm.UsageModelRow, error) {
	rows, err := p.pool.Query(ctx, `
		SELECT provider, model_id,
		       COUNT(*),
		       COALESCE(SUM(prompt_tokens), 0),
		       COALESCE(SUM(completion_tokens), 0),
		       COALESCE(SUM(cost_usd), 0) AS cost
		FROM usage_log WHERE ts >= $1
		GROUP BY provider, model_id
		ORDER BY cost DESC, 3 DESC
		LIMIT $2`, since, limit)
	if err != nil {
		return nil, fmt.Errorf("usage by model: %w", err)
	}
	defer rows.Close()
	var out []llm.UsageModelRow
	for rows.Next() {
		var r llm.UsageModelRow
		if err := rows.Scan(&r.Provider, &r.ModelID, &r.Calls, &r.PromptTokens, &r.CompletionTokens, &r.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

func (p *Postgres) UsageByRole(ctx context.Context, since time.Time) ([]llm.UsageRoleRow, error) {
	rows, err := p.pool.Query(ctx, `
		SELECT role, COUNT(*), COALESCE(SUM(cost_usd), 0)
		FROM usage_log WHERE ts >= $1
		GROUP BY role ORDER BY COUNT(*) DESC`, since)
	if err != nil {
		return nil, fmt.Errorf("usage by role: %w", err)
	}
	defer rows.Close()
	var out []llm.UsageRoleRow
	for rows.Next() {
		var r llm.UsageRoleRow
		if err := rows.Scan(&r.Role, &r.Calls, &r.CostUSD); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}

func (p *Postgres) ExpensiveTurns(ctx context.Context, since time.Time, limit int) ([]llm.ExpensiveTurn, error) {
	rows, err := p.pool.Query(ctx, `
		SELECT u.ts, u.chat_id, u.role, u.model_id, u.cost_usd,
		       (u.prompt_tokens + u.completion_tokens) AS tokens,
		       COALESCE(mq.content, ''), COALESCE(ma.content, '')
		FROM usage_log u
		LEFT JOIN messages mq ON mq.id = u.user_message_id
		LEFT JOIN messages ma ON ma.id = u.assistant_message_id
		WHERE u.ts >= $1 AND u.cost_usd > 0
		ORDER BY u.cost_usd DESC
		LIMIT $2`, since, limit)
	if err != nil {
		return nil, fmt.Errorf("expensive turns: %w", err)
	}
	defer rows.Close()
	var out []llm.ExpensiveTurn
	for rows.Next() {
		var t llm.ExpensiveTurn
		if err := rows.Scan(&t.Ts, &t.ChatID, &t.Role, &t.ModelID, &t.CostUSD, &t.Tokens, &t.Question, &t.Answer); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, nil
}
