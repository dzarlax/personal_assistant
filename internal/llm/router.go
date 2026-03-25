package llm

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
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

// routingOverrides is the persisted subset of RouterConfig (JSON file).
type routingOverrides struct {
	Primary          string `json:"primary,omitempty"`
	Fallback         string `json:"fallback,omitempty"`
	Multimodal       string `json:"multimodal,omitempty"`
	Reasoner         string `json:"reasoner,omitempty"`
	Classifier       string `json:"classifier,omitempty"`
	ClassifierMinLen *int   `json:"classifier_min_len,omitempty"`
}

// Router selects the appropriate LLM provider based on context.
// All providers are stored by name; special roles reference names from RouterConfig.
type Router struct {
	providers map[string]Provider
	cfg       RouterConfig

	mu          sync.RWMutex
	override    string // set via SetOverride; any key in providers, or "" for auto
	persistPath string // path to save/load routing overrides; empty = no persistence

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

// SetPersistPath sets the file path for persisting routing overrides.
func (r *Router) SetPersistPath(path string) {
	r.mu.Lock()
	r.persistPath = path
	r.mu.Unlock()
}

// LoadPersistedOverrides reads routing overrides from the persist path and applies them.
func (r *Router) LoadPersistedOverrides() error {
	r.mu.RLock()
	path := r.persistPath
	r.mu.RUnlock()
	if path == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var ov routingOverrides
	if err := json.Unmarshal(data, &ov); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if ov.Primary != "" {
		r.cfg.Primary = ov.Primary
	}
	if ov.Fallback != "" {
		r.cfg.Fallback = ov.Fallback
	}
	if ov.Multimodal != "" {
		r.cfg.Multimodal = ov.Multimodal
	}
	if ov.Reasoner != "" {
		r.cfg.Reasoner = ov.Reasoner
	}
	if ov.Classifier != "" {
		r.cfg.Classifier = ov.Classifier
	}
	if ov.ClassifierMinLen != nil {
		r.cfg.ClassifierMinLen = *ov.ClassifierMinLen
	}
	return nil
}

// saveOverrides writes current cfg to the persist path. Must be called with mu held.
func (r *Router) saveOverrides() {
	if r.persistPath == "" {
		return
	}
	minLen := r.cfg.ClassifierMinLen
	ov := routingOverrides{
		Primary:          r.cfg.Primary,
		Fallback:         r.cfg.Fallback,
		Multimodal:       r.cfg.Multimodal,
		Reasoner:         r.cfg.Reasoner,
		Classifier:       r.cfg.Classifier,
		ClassifierMinLen: &minLen,
	}
	data, err := json.Marshal(ov)
	if err != nil {
		return
	}
	if err := os.WriteFile(r.persistPath, data, 0644); err != nil {
		r.logger.Error("failed to save routing overrides", "path", r.persistPath, "err", err)
	} else {
		r.logger.Info("routing overrides saved", "path", r.persistPath)
	}
}

// GetConfig returns a snapshot of the current routing configuration.
func (r *Router) GetConfig() RouterConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.cfg
}

// SetRole updates a routing role to point at the given model name.
// Valid roles: primary, fallback, reasoner, classifier, multimodal.
func (r *Router) SetRole(role, model string) error {
	if _, ok := r.providers[model]; !ok {
		return errors.New("unknown model: " + model)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	switch role {
	case "primary":
		r.cfg.Primary = model
	case "fallback":
		r.cfg.Fallback = model
	case "reasoner":
		r.cfg.Reasoner = model
	case "classifier":
		r.cfg.Classifier = model
	case "multimodal":
		r.cfg.Multimodal = model
	default:
		return errors.New("unknown role: " + role)
	}
	r.saveOverrides()
	return nil
}

// SetClassifierMinLen sets the minimum message length to trigger the classifier.
// Set to 0 to disable classifier routing.
func (r *Router) SetClassifierMinLen(n int) {
	r.mu.Lock()
	r.cfg.ClassifierMinLen = n
	r.saveOverrides()
	r.mu.Unlock()
}

func (r *Router) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	provider := r.pick(ctx, messages)

	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	if err != nil && isUnavailable(err) {
		// Build fallback chain: override → default → fallback.
		// Skip providers already tried or equal to current.
		r.mu.RLock()
		chain := []string{r.cfg.Primary, r.cfg.Fallback}
		r.mu.RUnlock()

		for _, key := range chain {
			next := r.get(key)
			if next == nil || next == provider {
				continue
			}
			slog.Info("routing", "reason", "fallback", "from", provider.Name(), "to", next.Name(), "err", err.Error())
			if r.OnFallback != nil {
				r.OnFallback(provider.Name(), next.Name())
			}
			resp, err = next.Chat(ctx, messages, systemPrompt, tools)
			if err == nil || !isUnavailable(err) {
				break
			}
			provider = next // track last tried for next iteration
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

// SessionResetter is an optional interface for providers that support session management.
type SessionResetter interface {
	ResetSession()
}

// ResetProviderSession resets session state on the named provider (if it supports sessions).
func (r *Router) ResetProviderSession(name string) {
	if p, ok := r.providers[name]; ok {
		if sr, ok := p.(SessionResetter); ok {
			sr.ResetSession()
		}
	}
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
	r.mu.RLock()
	cfg := r.cfg
	override := r.override
	r.mu.RUnlock()

	// Multimodal takes priority — only vision-capable models support image parts
	if p := r.get(cfg.Multimodal); p != nil && hasMultimodalContent(messages) {
		slog.Info("routing", "reason", "multimodal", "provider", p.Name())
		return p
	}

	if override != "" {
		if p := r.get(override); p != nil {
			slog.Info("routing", "reason", "override", "provider", p.Name())
			return p
		}
	}

	// Classifier-based routing to reasoner
	if cfg.ClassifierMinLen > 0 {
		if p := r.get(cfg.Reasoner); p != nil {
			if text := lastUserText(messages); len([]rune(text)) >= cfg.ClassifierMinLen {
				if r.classify(ctx, text) {
					slog.Info("routing", "reason", "classifier→reasoner", "provider", p.Name())
					return p
				}
			}
		}
	}

	if p := r.get(cfg.Primary); p != nil {
		slog.Info("routing", "reason", "primary", "provider", p.Name())
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
	r.mu.RLock()
	classifierKey := r.cfg.Classifier
	primaryKey := r.cfg.Primary
	r.mu.RUnlock()

	if classifierKey == "" {
		classifierKey = primaryKey
	}
	provider := r.get(classifierKey)
	if provider == nil {
		return false
	}
	classifierCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	// Truncate to save tokens — classifier only needs the beginning to judge complexity.
	classifierText := text
	if len([]rune(classifierText)) > 500 {
		classifierText = string([]rune(classifierText)[:500])
	}
	msgs := []Message{{Role: "user", Content: classifierText}}
	resp, err := provider.Chat(classifierCtx, msgs, classifierPrompt, nil)
	if err != nil {
		r.logger.Warn("classifier error, falling back to primary", "classifier", classifierKey, "err", err)
		return false
	}
	result := strings.HasPrefix(strings.ToLower(strings.TrimSpace(resp.Content)), "yes")
	if result {
		r.logger.Info("classifier routed to reasoner", "text_len", len(text))
	} else {
		r.logger.Debug("classifier result: primary", "text_len", len(text))
	}
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
