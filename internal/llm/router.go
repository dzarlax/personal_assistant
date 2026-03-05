package llm

import (
	"context"
	"errors"
	"strings"
	"sync"

	"github.com/sashabaranov/go-openai"
)

var reasonerKeywords = []string{
	"подумай пошагово", "рассуди", "докажи", "разбери по шагам",
	"step by step", "think step",
}

// Router selects the appropriate LLM provider based on context.
// Primary → Fallback on 5xx/network error.
// Reasoner → selected by keyword or explicit /model command.
// Multimodal → selected when message contains image parts.
type Router struct {
	primary    Provider
	fallback   Provider // used if primary is unavailable
	reasoner   Provider // used for complex reasoning
	multimodal Provider // used when message contains images

	mu       sync.RWMutex
	override string // "reasoner" or "" (primary)

	OnFallback func(from, to string) // optional: called when fallback is triggered
}

func NewRouter(primary, fallback, reasoner, multimodal Provider) *Router {
	return &Router{
		primary:    primary,
		fallback:   fallback,
		reasoner:   reasoner,
		multimodal: multimodal,
	}
}

func (r *Router) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	provider := r.pick(messages)

	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	if err != nil && r.fallback != nil && provider != r.fallback && isUnavailable(err) {
		if r.OnFallback != nil {
			r.OnFallback(provider.Name(), r.fallback.Name())
		}
		resp, err = r.fallback.Chat(ctx, messages, systemPrompt, tools)
	}
	return resp, err
}

func (r *Router) Name() string {
	return r.pick(nil).Name()
}

func (r *Router) SetOverride(model string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.override = model
}

func (r *Router) GetOverride() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.override
}

func (r *Router) pick(messages []Message) Provider {
	r.mu.RLock()
	override := r.override
	r.mu.RUnlock()

	// Multimodal takes priority — DeepSeek doesn't support vision
	if r.multimodal != nil && hasMultimodalContent(messages) {
		return r.multimodal
	}

	if override == "reasoner" && r.reasoner != nil {
		return r.reasoner
	}
	// Check last user message for reasoner keywords (single-request escalation)
	if r.reasoner != nil {
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				if containsReasonerKeyword(messages[i].Content) {
					return r.reasoner
				}
				break
			}
		}
	}
	return r.primary
}

func hasMultimodalContent(messages []Message) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return len(messages[i].Parts) > 0
		}
	}
	return false
}

func containsReasonerKeyword(text string) bool {
	lower := strings.ToLower(text)
	for _, kw := range reasonerKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func isUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return apiErr.HTTPStatusCode >= 500 || apiErr.HTTPStatusCode == 429
	}
	return true // network error — try fallback
}
