package store

import (
	"time"

	"telegram-agent/internal/llm"
)

// ChatStats holds per-chat usage statistics.
type ChatStats struct {
	ActiveMessages int
	ActiveChars    int
	LastCompactAt  time.Time // zero if never compacted
	LastMessageAt  time.Time // zero if no messages
}

// Store is the interface for conversation history storage.
// AddMessage returns the new row's id (0 if the backend has no IDs, e.g. memory).
type Store interface {
	GetHistory(chatID int64) []llm.Message
	AddMessage(chatID int64, msg llm.Message) int64
	ClearHistory(chatID int64)
}

// MessageRow wraps a message with its database ID (for compaction).
type MessageRow struct {
	ID        int64
	Message   llm.Message
	Embedding []float32 // nil if not stored; populated by GetAllActive when available
}

// CompactableStore extends Store with compaction support (implemented by SQLite).
type CompactableStore interface {
	Store
	GetAllActive(chatID int64) ([]MessageRow, error)
	AddSummary(chatID int64, content string)
	MarkCompacted(ids []int64) error
	ActiveCharCount(chatID int64) int
	GetStats(chatID int64) ChatStats
}

// HistorySnippet is a short excerpt from a past conversation turn, used for
// cross-session context injection into the system prompt.
type HistorySnippet struct {
	Date      time.Time
	UserText  string
	BotText   string
}

// HistoryItem is one row returned by DisplayableStore.DisplayHistory — a
// message, an image attachment descriptor, or a session-break marker. The
// admin web UI uses this to render the conversation with <hr/>-style
// dividers between sessions (from /clear or the 4-hour idle break).
type HistoryItem struct {
	Role      string    // "user" | "assistant" | "tool" | "break"
	Content   string    // message text OR break reason
	ImageURLs []string  // image_url parts (data URIs or http links)
	CreatedAt time.Time // for display timestamps + divider labels
}

// DisplayableStore extends Store with a UI-oriented history fetch that
// includes session-break markers and surfaces image attachments.
// Implemented by SQLite and Postgres; the in-memory store skips it.
type DisplayableStore interface {
	Store
	DisplayHistory(chatID int64, limit, offset int) []HistoryItem
}

// TruncatableStore supports dropping the tail of a conversation. Used by the
// admin web chat to implement Regenerate / Edit. LastUserMessage returns the
// most recent user-role row within the current session (id > last reset).
// TruncateAfter deletes every row with id >= fromID within the same chat.
type TruncatableStore interface {
	Store
	LastUserMessage(chatID int64) (id int64, text string, ok bool)
	TruncateAfter(chatID int64, fromID int64) error
}

// SemanticStore extends CompactableStore with vector-similarity history retrieval.
// AddMessageWithEmbedding stores a message alongside its pre-computed embedding so
// that GetSemanticHistory can rank older turns by relevance to the current query.
type SemanticStore interface {
	CompactableStore
	// AddMessageWithEmbedding stores msg and its embedding in one write.
	// Returns the new row's id (0 if the backend has no IDs).
	AddMessageWithEmbedding(chatID int64, msg llm.Message, emb []float32) int64
	// GetSemanticHistory returns the last recentN messages unconditionally, plus up
	// to topK older conversational turns ranked by cosine similarity to queryEmb.
	// Results are always returned in chronological order.
	GetSemanticHistory(chatID int64, queryEmb []float32, recentN, topK int) []llm.Message
	// SearchAllSessions searches across ALL sessions (ignoring reset markers) and
	// returns up to topK turns from past sessions that are semantically similar to
	// queryEmb and have cosine similarity above minScore.
	// Turns from the current session are excluded.
	SearchAllSessions(chatID int64, queryEmb []float32, topK int, minScore float64) []HistorySnippet
}
