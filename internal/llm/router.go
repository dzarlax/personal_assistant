package llm

import (
	"context"
	"errors"
	"log/slog"
	"sort"
	"strings"
	"sync"

)

const classifierPrompt = `Does this message require deep step-by-step reasoning, mathematical proof, or complex multi-step analysis? Reply with only 'yes' or 'no'.`

// RouterConfig holds the keys into the providers map for special roles.
type RouterConfig struct {
	Primary          string
	Fallback         string
	Multimodal       string
	Reasoner         string
	Classifier       string // provider used for reasoning classification; falls back to Primary if empty
	ClassifierMinLen int    // min rune length to run classifier; 0 = disabled
}

// Router selects the appropriate LLM provider based on context.
// All providers are stored by name; special roles reference names from RouterConfig.
type Router struct {
	providers map[string]Provider
	cfg       RouterConfig

	mu       sync.RWMutex
	override string // set via SetOverride; any key in providers, or "" for auto

	OnFallback func(from, to string)
	logger     *slog.Logger
}

func NewRouter(providers map[string]Provider, cfg RouterConfig) *Router {
	return &Router{
		providers: providers,
		cfg:       cfg,
		logger:    slog.Default(),
	}
}

func (r *Router) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	provider := r.pick(ctx, messages)

	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	if err != nil && isUnavailable(err) {
		if fallback := r.get(r.cfg.Fallback); fallback != nil && fallback != provider {
			if r.OnFallback != nil {
				r.OnFallback(provider.Name(), fallback.Name())
			}
			resp, err = fallback.Chat(ctx, messages, systemPrompt, tools)
		}
	}
	return resp, err
}

// Name returns the name of the currently active provider (respecting override).
func (r *Router) Name() string {
	return r.pick(context.Background(), nil).Name()
}

// SetOverride sets a named model override. Pass "" to clear. Returns error if name is unknown.
func (r *Router) SetOverride(name string) error {
	if name != "" {
		if _, ok := r.providers[name]; !ok {
			return errors.New("unknown model: " + name)
		}
	}
	r.mu.Lock()
	r.override = name
	r.mu.Unlock()
	return nil
}

// GetOverride returns the current override name (empty = auto).
func (r *Router) GetOverride() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.override
}

// ProviderNames returns sorted list of all available provider names.
func (r *Router) ProviderNames() []string {
	names := make([]string, 0, len(r.providers))
	for k := range r.providers {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

func (r *Router) pick(ctx context.Context, messages []Message) Provider {
	// Multimodal takes priority — only vision-capable models support image parts
	if p := r.get(r.cfg.Multimodal); p != nil && hasMultimodalContent(messages) {
		return p
	}

	r.mu.RLock()
	override := r.override
	r.mu.RUnlock()

	if override != "" {
		if p := r.get(override); p != nil {
			return p
		}
	}

	// Classifier-based routing to reasoner
	if r.cfg.ClassifierMinLen > 0 {
		if p := r.get(r.cfg.Reasoner); p != nil {
			if text := lastUserText(messages); len([]rune(text)) >= r.cfg.ClassifierMinLen {
				if r.classify(ctx, text) {
					return p
				}
			}
		}
	}

	if p := r.get(r.cfg.Primary); p != nil {
		return p
	}
	// Should never happen if config is valid
	for _, p := range r.providers {
		return p
	}
	return nil
}

func (r *Router) get(key string) Provider {
	if key == "" {
		return nil
	}
	return r.providers[key]
}

// classify calls the classifier provider (or primary) to determine if the message
// requires deep reasoning. Returns false on any error.
func (r *Router) classify(ctx context.Context, text string) bool {
	classifierKey := r.cfg.Classifier
	if classifierKey == "" {
		classifierKey = r.cfg.Primary
	}
	provider := r.get(classifierKey)
	if provider == nil {
		return false
	}
	msgs := []Message{{Role: "user", Content: text}}
	resp, err := provider.Chat(ctx, msgs, classifierPrompt, nil)
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
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode >= 500 || apiErr.StatusCode == 429
	}
	return true // network error — try fallback
}
