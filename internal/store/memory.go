package store

import (
	"sync"

	"telegram-agent/internal/llm"
)

const maxHistory = 50

type Memory struct {
	mu       sync.RWMutex
	sessions map[int64][]llm.Message
}

func NewMemory() *Memory {
	return &Memory{
		sessions: make(map[int64][]llm.Message),
	}
}

func (m *Memory) GetHistory(chatID int64) []llm.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	msgs := m.sessions[chatID]
	if len(msgs) == 0 {
		return nil
	}

	result := make([]llm.Message, len(msgs))
	copy(result, msgs)
	return result
}

func (m *Memory) AddMessage(chatID int64, msg llm.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.sessions[chatID] = append(m.sessions[chatID], msg)

	if len(m.sessions[chatID]) > maxHistory {
		m.sessions[chatID] = m.sessions[chatID][len(m.sessions[chatID])-maxHistory:]
	}
}

func (m *Memory) ClearHistory(chatID int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, chatID)
}
