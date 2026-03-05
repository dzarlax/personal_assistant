package agent

import (
	"context"
	"fmt"
	"log/slog"

	"telegram-agent/internal/llm"
	"telegram-agent/internal/store"
)

const (
	compactionKeepLast      = 10
	compactionCharThreshold = 60000 // ~20K tokens, practical threshold for daily use
)

const compactionSystemPrompt = `Сожми историю разговора в краткое резюме на том же языке что и разговор.
Сохрани: ключевые факты о пользователе, принятые решения, незавершённые задачи, важный контекст.
Пиши только суть — без предисловий и лишних слов.`

// Compacter summarizes old conversation history.
type Compacter struct {
	provider llm.Provider
}

func NewCompacter(provider llm.Provider) *Compacter {
	return &Compacter{provider: provider}
}

// NeedsCompaction returns true if the conversation history should be compacted.
func NeedsCompaction(s store.Store, chatID int64) bool {
	cs, ok := s.(store.CompactableStore)
	if !ok {
		return false
	}
	return cs.ActiveCharCount(chatID) > compactionCharThreshold
}

// Compact summarizes old messages and marks them as archived.
func (c *Compacter) Compact(ctx context.Context, chatID int64, s store.Store) error {
	cs, ok := s.(store.CompactableStore)
	if !ok {
		return fmt.Errorf("store does not support compaction")
	}

	rows, err := cs.GetAllActive(chatID)
	if err != nil {
		return fmt.Errorf("get active messages: %w", err)
	}

	slog.Info("compact: active messages", "count", len(rows), "char_count", cs.ActiveCharCount(chatID))

	boundary := findBoundary(rows, compactionKeepLast)
	slog.Info("compact: boundary", "boundary", boundary, "keep_last", compactionKeepLast)
	if boundary == 0 {
		return nil // nothing to compact
	}

	toCompact := rows[:boundary]

	// Build message history for the summary request
	history := make([]llm.Message, 0, len(toCompact))
	for _, row := range toCompact {
		history = append(history, row.Message)
	}

	resp, err := c.provider.Chat(ctx, history, compactionSystemPrompt, nil)
	if err != nil {
		return fmt.Errorf("summarize: %w", err)
	}

	// Insert summary before marking old messages as compacted
	cs.AddSummary(chatID, "[Резюме предыдущего разговора]\n\n"+resp.Content)

	ids := make([]int64, len(toCompact))
	for i, row := range toCompact {
		ids[i] = row.ID
	}
	return cs.MarkCompacted(ids)
}

// findBoundary finds the split point: messages before this index get compacted.
// Snaps to a user message boundary to avoid splitting tool call sequences.
func findBoundary(rows []store.MessageRow, keepLast int) int {
	if len(rows) <= keepLast {
		return 0
	}
	boundary := len(rows) - keepLast
	// Snap back to the nearest user message so we don't split mid-sequence
	for boundary > 0 && rows[boundary].Message.Role != "user" {
		boundary--
	}
	return boundary
}
