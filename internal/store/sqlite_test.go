package store

import (
	"context"
	"testing"

	"telegram-agent/internal/llm"
)

// newTestStore создаёт SQLite в памяти для тестов — быстро, без файлов на диске.
func newTestStore(t *testing.T) *SQLite {
	t.Helper()
	s, err := NewSQLite(":memory:")
	if err != nil {
		t.Fatalf("failed to create in-memory SQLite: %v", err)
	}
	t.Cleanup(func() { s.db.Close() })
	return s
}

// TestSQLite_AddAndGetHistory: добавляем сообщения, читаем историю.
func TestSQLite_AddAndGetHistory(t *testing.T) {
	s := newTestStore(t)
	chatID := int64(42)

	s.AddMessage(chatID, llm.Message{Role: "user", Content: "hello"})
	s.AddMessage(chatID, llm.Message{Role: "assistant", Content: "world"})

	history := s.GetHistory(chatID)
	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}
	if history[0].Role != "user" || history[0].Content != "hello" {
		t.Errorf("unexpected first message: %+v", history[0])
	}
	if history[1].Role != "assistant" || history[1].Content != "world" {
		t.Errorf("unexpected second message: %+v", history[1])
	}
}

// TestSQLite_EmptyHistory: новый чат → пустая история.
func TestSQLite_EmptyHistory(t *testing.T) {
	s := newTestStore(t)
	history := s.GetHistory(999)
	if len(history) != 0 {
		t.Errorf("expected empty history for new chat, got %d messages", len(history))
	}
}

// TestSQLite_ChatIsolation: сообщения разных чатов не смешиваются.
func TestSQLite_ChatIsolation(t *testing.T) {
	s := newTestStore(t)

	s.AddMessage(1, llm.Message{Role: "user", Content: "chat1"})
	s.AddMessage(2, llm.Message{Role: "user", Content: "chat2"})

	h1 := s.GetHistory(1)
	h2 := s.GetHistory(2)

	if len(h1) != 1 || h1[0].Content != "chat1" {
		t.Errorf("chat1 history wrong: %+v", h1)
	}
	if len(h2) != 1 || h2[0].Content != "chat2" {
		t.Errorf("chat2 history wrong: %+v", h2)
	}
}

// TestSQLite_ClearHistory: после ClearHistory — история пустая.
func TestSQLite_ClearHistory(t *testing.T) {
	s := newTestStore(t)
	chatID := int64(1)

	s.AddMessage(chatID, llm.Message{Role: "user", Content: "before clear"})
	s.ClearHistory(chatID)

	history := s.GetHistory(chatID)
	if len(history) != 0 {
		t.Errorf("expected empty history after clear, got %d messages", len(history))
	}
}

// TestSQLite_ClearHistoryDoesNotAffectOtherChats: /clear в одном чате не трогает другой.
func TestSQLite_ClearHistoryDoesNotAffectOtherChats(t *testing.T) {
	s := newTestStore(t)

	s.AddMessage(1, llm.Message{Role: "user", Content: "stay"})
	s.AddMessage(2, llm.Message{Role: "user", Content: "cleared"})

	s.ClearHistory(2)

	if h := s.GetHistory(1); len(h) != 1 {
		t.Errorf("chat1 should still have 1 message, got %d", len(h))
	}
	if h := s.GetHistory(2); len(h) != 0 {
		t.Errorf("chat2 should be empty after clear, got %d", len(h))
	}
}

// TestSQLite_MessageWithToolCalls: tool call сообщения сохраняются и читаются корректно.
func TestSQLite_MessageWithToolCalls(t *testing.T) {
	s := newTestStore(t)
	chatID := int64(1)

	s.AddMessage(chatID, llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{
			{ID: "call_1", Name: "get_weather", Arguments: `{"city":"Moscow"}`},
		},
	})
	s.AddMessage(chatID, llm.Message{
		Role:       "tool",
		Content:    "Sunny, 20°C",
		ToolCallID: "call_1",
	})

	history := s.GetHistory(chatID)
	if len(history) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(history))
	}

	assistantMsg := history[0]
	if len(assistantMsg.ToolCalls) != 1 {
		t.Errorf("expected 1 tool call, got %d", len(assistantMsg.ToolCalls))
	}
	if assistantMsg.ToolCalls[0].Name != "get_weather" {
		t.Errorf("unexpected tool name: %q", assistantMsg.ToolCalls[0].Name)
	}

	toolMsg := history[1]
	if toolMsg.ToolCallID != "call_1" {
		t.Errorf("unexpected tool_call_id: %q", toolMsg.ToolCallID)
	}
}

// TestSQLite_MultimodalPartsRoundtrip: multimodal сообщение (Parts) сохраняется и восстанавливается.
func TestSQLite_MultimodalPartsRoundtrip(t *testing.T) {
	s := newTestStore(t)
	chatID := int64(1)

	original := llm.Message{
		Role: "user",
		Parts: []llm.ContentPart{
			{Type: "text", Text: "what is this?"},
			{Type: "image_url", ImageURL: &llm.ImageURL{URL: "data:image/jpeg;base64,abc123"}},
		},
	}
	s.AddMessage(chatID, original)

	history := s.GetHistory(chatID)
	if len(history) != 1 {
		t.Fatalf("expected 1 message, got %d", len(history))
	}

	msg := history[0]
	if len(msg.Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(msg.Parts))
	}
	if msg.Parts[0].Type != "text" || msg.Parts[0].Text != "what is this?" {
		t.Errorf("unexpected text part: %+v", msg.Parts[0])
	}
	if msg.Parts[1].Type != "image_url" || msg.Parts[1].ImageURL.URL != "data:image/jpeg;base64,abc123" {
		t.Errorf("unexpected image part: %+v", msg.Parts[1])
	}
}

// TestSQLite_HistoryLimitedTo30: история ограничена последними 30 сообщениями.
func TestSQLite_HistoryLimitedTo30(t *testing.T) {
	s := newTestStore(t)
	chatID := int64(1)

	for i := 0; i < 50; i++ {
		s.AddMessage(chatID, llm.Message{Role: "user", Content: "msg"})
	}

	history := s.GetHistory(chatID)
	if len(history) != sqliteMaxHistory {
		t.Errorf("expected %d messages (limit), got %d", sqliteMaxHistory, len(history))
	}
}

// TestSQLite_Capabilities: put → get round-trip and list by provider.
func TestSQLite_Capabilities(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	in := llm.Capabilities{
		Vision:          true,
		Tools:           true,
		Reasoning:       false,
		PromptPrice:     3.0,
		CompletionPrice: 15.0,
		ContextLength:   200000,
	}
	if err := s.PutCapabilities(ctx, "openrouter", "anthropic/claude-sonnet-4.5", in); err != nil {
		t.Fatalf("put: %v", err)
	}

	got, ok, err := s.GetCapabilities(ctx, "openrouter", "anthropic/claude-sonnet-4.5")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if !ok {
		t.Fatal("capabilities not found after put")
	}
	if got != in {
		t.Errorf("roundtrip mismatch:\n want %+v\n got  %+v", in, got)
	}

	// Missing entry → (_, false, nil)
	_, ok, err = s.GetCapabilities(ctx, "openrouter", "nonexistent/model")
	if err != nil {
		t.Fatalf("get missing: %v", err)
	}
	if ok {
		t.Error("expected ok=false for missing entry")
	}

	// Second entry — verify GetAll returns both and filters by provider.
	other := llm.Capabilities{Tools: true, PromptPrice: 0.27, CompletionPrice: 1.10, ContextLength: 65536}
	if err := s.PutCapabilities(ctx, "openrouter", "deepseek/deepseek-chat-v3.1", other); err != nil {
		t.Fatalf("put 2: %v", err)
	}
	// Entry under a different provider that GetAll must NOT return.
	if err := s.PutCapabilities(ctx, "other-provider", "foo/bar", llm.Capabilities{}); err != nil {
		t.Fatalf("put other: %v", err)
	}

	all, err := s.GetAllCapabilities(ctx, "openrouter")
	if err != nil {
		t.Fatalf("getall: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("want 2 openrouter entries, got %d", len(all))
	}
	if _, ok := all["anthropic/claude-sonnet-4.5"]; !ok {
		t.Error("missing claude entry")
	}
	if _, ok := all["deepseek/deepseek-chat-v3.1"]; !ok {
		t.Error("missing deepseek entry")
	}

	// Upsert semantics — same key, new values overwrite.
	updated := in
	updated.PromptPrice = 5.0
	if err := s.PutCapabilities(ctx, "openrouter", "anthropic/claude-sonnet-4.5", updated); err != nil {
		t.Fatalf("put update: %v", err)
	}
	got, _, _ = s.GetCapabilities(ctx, "openrouter", "anthropic/claude-sonnet-4.5")
	if got.PromptPrice != 5.0 {
		t.Errorf("upsert did not overwrite: got %v", got.PromptPrice)
	}
}

// TestSQLite_Settings: kv round-trip + upsert semantics.
func TestSQLite_Settings(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	// Missing key → (_, false, nil)
	_, ok, err := s.GetSetting(ctx, "routing.overrides")
	if err != nil {
		t.Fatalf("get missing: %v", err)
	}
	if ok {
		t.Error("expected ok=false for missing key")
	}

	payload := `{"default":"workhorse","openrouter_models":{"workhorse":"deepseek/deepseek-chat-v3.1"}}`
	if err := s.PutSetting(ctx, "routing.overrides", payload); err != nil {
		t.Fatalf("put: %v", err)
	}
	got, ok, err := s.GetSetting(ctx, "routing.overrides")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if !ok {
		t.Fatal("setting not found after put")
	}
	if got != payload {
		t.Errorf("roundtrip mismatch: got %q want %q", got, payload)
	}

	// Upsert
	newPayload := `{"default":"cheap-or"}`
	if err := s.PutSetting(ctx, "routing.overrides", newPayload); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	got, _, _ = s.GetSetting(ctx, "routing.overrides")
	if got != newPayload {
		t.Errorf("upsert did not overwrite: got %q", got)
	}
}
