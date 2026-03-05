package store

import "telegram-agent/internal/llm"

// Store is the interface for conversation history storage.
type Store interface {
	GetHistory(chatID int64) []llm.Message
	AddMessage(chatID int64, msg llm.Message)
	ClearHistory(chatID int64)
}

// MessageRow wraps a message with its database ID (for compaction).
type MessageRow struct {
	ID      int64
	Message llm.Message
}

// CompactableStore extends Store with compaction support (implemented by SQLite).
type CompactableStore interface {
	Store
	GetAllActive(chatID int64) ([]MessageRow, error)
	AddSummary(chatID int64, content string)
	MarkCompacted(ids []int64) error
	ActiveCharCount(chatID int64) int
}
