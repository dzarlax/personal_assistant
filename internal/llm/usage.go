package llm

import (
	"context"
	"time"
)

// UsageLog is a billing-grained record of a single LLM call. One record per
// call, independent of the messages table — an agentic loop with N tool
// iterations produces N records, all sharing the same UserMessageID.
//
// Schema lives in SQLite + Postgres migrations under usage_log.
type UsageLog struct {
	ID                 int64
	Ts                 time.Time
	Provider           string // "openrouter" / "gemini" / "claude_bridge" / "ollama" / "local"
	ModelID            string // provider-specific model identifier at call time
	Role               string // routing role chosen ("simple" / "default" / "complex" / "multimodal" / "compaction" / "classifier" / "fallback" / "override")
	ChatID             int64
	PromptTokens       int
	CompletionTokens   int
	CachedPromptTokens int // subset of PromptTokens that hit provider's prompt cache (Anthropic/OpenAI)
	ReasoningTokens    int // for thinking-enabled models: tokens spent on internal reasoning (billed as completion but semantically distinct)
	Cost               float64 // USD, authoritative per-request cost reported by the provider (OpenRouter's usage.cost). 0 when the provider doesn't surface this.
	LatencyMs          int
	Success            bool
	ErrorClass         string // "" / "rate_limit" / "5xx" / "network" / "timeout" / "other"
	RequestID          string // provider's request id (e.g. OpenRouter gen-xxxxx) — useful for cross-referencing with provider dashboards
	ToolCallCount      int    // number of tool_calls in the response
	UserMessageID      *int64 // FK to messages.id; NULL for background tasks (compaction, scheduled jobs)
	AssistantMessageID *int64 // FK to messages.id; NULL except on the final successful call of a turn
}

// UsageStore persists UsageLog records and exposes aggregation queries used
// by the admin UI (implemented in Stage A.3).
type UsageStore interface {
	PutUsage(ctx context.Context, u UsageLog) (int64, error)
	UpdateAssistantMessageID(ctx context.Context, usageID, msgID int64) error
}

// Usage is a per-response token breakdown attached to Response. Providers
// populate it when their API reports counts; zero-value otherwise. Router
// copies these fields into UsageLog.
type Usage struct {
	PromptTokens       int
	CompletionTokens   int
	CachedPromptTokens int
	ReasoningTokens    int
	Cost               float64 // USD (OpenRouter `usage.cost`); 0 when provider doesn't report it
	RequestID          string
}

// TurnMeta carries the conversation-layer identity (chat, user message id,
// role) across layers so Router.Chat can tag UsageLog records without every
// caller having to thread the data through function signatures.
//
// When absent, usage is still logged but with ChatID=0 / UserMessageID=nil.
type TurnMeta struct {
	ChatID        int64
	UserMessageID int64 // 0 = not applicable (compaction, scheduled jobs)
	// RoleHint overrides the role that Router would normally record. Used for
	// classifier calls (bypass Router.pick, role="classifier") and compaction
	// (role="compaction"). Empty = let Router decide.
	RoleHint string
}

type turnMetaKey struct{}

// WithTurnMeta returns a context carrying the supplied TurnMeta. Router's
// usage logger reads this via TurnMetaFrom.
func WithTurnMeta(ctx context.Context, m TurnMeta) context.Context {
	return context.WithValue(ctx, turnMetaKey{}, m)
}

// TurnMetaFrom retrieves the TurnMeta put into ctx by WithTurnMeta.
func TurnMetaFrom(ctx context.Context) (TurnMeta, bool) {
	m, ok := ctx.Value(turnMetaKey{}).(TurnMeta)
	return m, ok
}

// ClassifyErrorClass maps an error into a short bucket suitable for usage_log.
// Unknown errors land in "other" so dashboards aren't polluted by one-off
// messages. Returns "" when err is nil.
func ClassifyErrorClass(err error) string {
	if err == nil {
		return ""
	}
	var apiErr *APIError
	if ok := asAPIError(err, &apiErr); ok {
		switch {
		case apiErr.StatusCode == 429:
			return "rate_limit"
		case apiErr.StatusCode >= 500:
			return "5xx"
		case apiErr.StatusCode >= 400:
			return "4xx"
		}
	}
	msg := err.Error()
	switch {
	case contains(msg, "context deadline exceeded"):
		return "timeout"
	case contains(msg, "connection refused"), contains(msg, "no such host"), contains(msg, "i/o timeout"), contains(msg, "connection reset"):
		return "network"
	}
	return "other"
}

// small helpers kept here to avoid an extra import of errors/strings in cold
// paths of tests.
func asAPIError(err error, target **APIError) bool {
	for e := err; e != nil; {
		if a, ok := e.(*APIError); ok {
			*target = a
			return true
		}
		type wrapped interface{ Unwrap() error }
		if w, ok := e.(wrapped); ok {
			e = w.Unwrap()
			continue
		}
		return false
	}
	return false
}

func contains(s, sub string) bool {
	return len(sub) > 0 && len(s) >= len(sub) && indexOf(s, sub) >= 0
}

func indexOf(s, sub string) int {
	n, m := len(s), len(sub)
	for i := 0; i+m <= n; i++ {
		if s[i:i+m] == sub {
			return i
		}
	}
	return -1
}
