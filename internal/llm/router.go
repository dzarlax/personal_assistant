package llm

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"

	"github.com/sashabaranov/go-openai"
)

const classifierPrompt = `Does this message require deep step-by-step reasoning, mathematical proof, or complex multi-step analysis? Reply with only 'yes' or 'no'.`

// Router selects the appropriate LLM provider based on context.
// Primary → Fallback on 5xx/network error.
// Reasoner → selected by LLM classifier or explicit /model command.
// Multimodal → selected when message contains image parts.
type Router struct {
	primary    Provider
	fallback   Provider // used if primary is unavailable
	reasoner   Provider // used for complex reasoning
	multimodal Provider // used when message contains images

	classifierMinLen int // min message length to run classifier; 0 = disabled

	mu       sync.RWMutex
	override string // "reasoner" or "" (primary)

	OnFallback func(from, to string) // optional: called when fallback is triggered

	logger *slog.Logger
}

func NewRouter(primary, fallback, reasoner, multimodal Provider, classifierMinLen int) *Router {
	return &Router{
		primary:          primary,
		fallback:         fallback,
		reasoner:         reasoner,
		multimodal:       multimodal,
		classifierMinLen: classifierMinLen,
		logger:           slog.Default(),
	}
}

func (r *Router) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	provider := r.pick(ctx, messages)

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
	return r.pick(context.Background(), nil).Name()
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

func (r *Router) pick(ctx context.Context, messages []Message) Provider {
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

	// Classifier-based routing to reasoner
	if r.reasoner != nil && r.classifierMinLen > 0 {
		if text := lastUserText(messages); len([]rune(text)) >= r.classifierMinLen {
			if r.classify(ctx, text) {
				return r.reasoner
			}
		}
	}

	return r.primary
}

// classify calls the primary LLM with a minimal prompt to determine
// if the message requires deep reasoning. Returns false on any error.
func (r *Router) classify(ctx context.Context, text string) bool {
	msgs := []Message{{Role: "user", Content: text}}
	resp, err := r.primary.Chat(ctx, msgs, classifierPrompt, nil)
	if err != nil {
		r.logger.Warn("classifier call failed, using primary", "err", err)
		return false
	}
	result := strings.HasPrefix(strings.ToLower(strings.TrimSpace(resp.Content)), "yes")
	r.logger.Debug("classifier result", "needs_reasoning", result)
	return result
}

func lastUserText(messages []Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			if messages[i].Content != "" {
				return messages[i].Content
			}
			var sb strings.Builder
			for _, p := range messages[i].Parts {
				if p.Type == "text" {
					sb.WriteString(p.Text)
				}
			}
			return sb.String()
		}
	}
	return ""
}

func hasMultimodalContent(messages []Message) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return len(messages[i].Parts) > 0
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
