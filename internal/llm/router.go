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

const defaultClassifierPrompt = `You are a classifier. Output ONLY a single digit 1, 2, or 3.

1 = simple (greeting, chitchat, factual lookup, translation)
2 = moderate (summarization, code, analysis, multi-step)
3 = hard (math proof, deep reasoning, complex debugging)

Examples:
User: hello → 1
User: what is 2+2 → 1
User: write a REST API in Go → 2
User: prove Fermat last theorem → 3`

// RouterConfig holds the keys into the providers map for special roles.
type RouterConfig struct {
	Local            string // level 1: simple tasks (local model)
	Primary          string // level 2: moderate tasks (cloud model)
	Reasoner         string // level 3: complex reasoning
	Fallback         string
	Multimodal       string
	Classifier        string // provider used for complexity classification
	ClassifierMinLen  int    // min rune length to run classifier; 0 = always; <0 = disabled
	ClassifierTimeout int    // seconds; default 15
	ClassifierPrompt  string // system prompt for classifier; uses default if empty
}

// routingOverrides is the persisted subset of RouterConfig (JSON file).
type routingOverrides struct {
	Local            string `json:"local,omitempty"`
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
	lastRouted    string // display name of last provider (for UI)
	lastRoutedKey string // map key of last provider (for tool continuation)

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
	if ov.Local != "" {
		r.cfg.Local = ov.Local
	}
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
		Local:            r.cfg.Local,
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
	case "local":
		r.cfg.Local = model
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
	r.mu.Lock()
	r.lastRouted = provider.Name()
	r.lastRoutedKey = r.keyFor(provider)
	r.mu.Unlock()

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

// ChatStream returns a streaming channel if the current provider supports it.
// Falls back to wrapping a synchronous Chat() in a single-chunk channel.
func (r *Router) ChatStream(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error) {
	provider := r.pick(ctx, messages)
	r.mu.Lock()
	r.lastRouted = provider.Name()
	r.lastRoutedKey = r.keyFor(provider)
	r.mu.Unlock()
	if sp, ok := provider.(StreamProvider); ok {
		ch, err := sp.ChatStream(ctx, messages, systemPrompt, tools)
		if err != nil && isUnavailable(err) {
			// Try fallback providers synchronously.
			return r.syncFallbackStream(ctx, provider, messages, systemPrompt, tools, err)
		}
		return ch, err
	}
	// Provider does not stream — wrap synchronous call.
	return r.syncStream(ctx, provider, messages, systemPrompt, tools)
}

// syncStream wraps a synchronous Chat() call in a channel.
func (r *Router) syncStream(ctx context.Context, provider Provider, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error) {
	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Delta: resp.Content, ToolCalls: resp.ToolCalls, Done: true}
	close(ch)
	return ch, nil
}

// syncFallbackStream tries the fallback chain synchronously and wraps the result.
func (r *Router) syncFallbackStream(ctx context.Context, failed Provider, messages []Message, systemPrompt string, tools []Tool, origErr error) (<-chan StreamChunk, error) {
	r.mu.RLock()
	chain := []string{r.cfg.Primary, r.cfg.Fallback}
	r.mu.RUnlock()

	for _, key := range chain {
		next := r.get(key)
		if next == nil || next == failed {
			continue
		}
		slog.Info("routing", "reason", "fallback", "from", failed.Name(), "to", next.Name(), "err", origErr.Error())
		if r.OnFallback != nil {
			r.OnFallback(failed.Name(), next.Name())
		}
		// Try streaming on fallback if supported.
		if sp, ok := next.(StreamProvider); ok {
			ch, err := sp.ChatStream(ctx, messages, systemPrompt, tools)
			if err == nil {
				return ch, nil
			}
		}
		// Otherwise sync.
		return r.syncStream(ctx, next, messages, systemPrompt, tools)
	}
	return nil, origErr
}

// SupportsStreaming returns true if the currently active provider implements StreamProvider.
func (r *Router) SupportsStreaming() bool {
	provider := r.pick(context.Background(), nil)
	_, ok := provider.(StreamProvider)
	return ok
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

// LastRouted returns the name of the last provider used for a request.
func (r *Router) LastRouted() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.lastRouted
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

// AddProvider registers a new provider at runtime (e.g. dynamic Ollama Cloud models).
func (r *Router) AddProvider(key string, p Provider) {
	r.mu.Lock()
	r.providers[key] = p
	r.mu.Unlock()
}

// OllamaCloudConfigProvider is implemented by Ollama providers to expose their base config.
type OllamaCloudConfigProvider interface {
	OllamaBaseConfig() (baseURL, apiKey string, maxTokens int)
}

// FindOllamaCloudConfig searches providers for the first one whose base URL points
// to Ollama Cloud (api.ollama.com). Returns the base config for creating dynamic providers.
func (r *Router) FindOllamaCloudConfig() (baseURL, apiKey string, maxTokens int, found bool) {
	for _, p := range r.providers {
		if ocp, ok := p.(OllamaCloudConfigProvider); ok {
			bu, ak, mt := ocp.OllamaBaseConfig()
			if strings.Contains(bu, "ollama.com") {
				return bu, ak, mt, true
			}
		}
	}
	return "", "", 0, false
}

func (r *Router) pick(ctx context.Context, messages []Message) Provider {
	r.mu.RLock()
	cfg := r.cfg
	override := r.override
	r.mu.RUnlock()

	multimodal := hasMultimodalContent(messages)

	if override != "" {
		if p := r.get(override); p != nil {
			if !multimodal || supportsVision(p) {
				slog.Info("routing", "reason", "override", "provider", p.Name())
				return p
			}
			// Override doesn't support vision — fall through to multimodal
		}
	}

	// Multimodal routing — prefer local/primary if they support vision
	if multimodal {
		if p := r.get(cfg.Local); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "local+vision", "provider", p.Name())
			return p
		}
		if p := r.get(cfg.Primary); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "primary+vision", "provider", p.Name())
			return p
		}
		if p := r.get(cfg.Multimodal); p != nil {
			slog.Info("routing", "reason", "multimodal", "provider", p.Name())
			return p
		}
	}

	// Tool continuation — keep using the same provider that started the tool loop.
	if hasToolMessages(messages) && r.lastRoutedKey != "" {
		if last := r.get(r.lastRoutedKey); last != nil {
			slog.Info("routing", "reason", "tool-continuation", "provider", last.Name())
			return last
		}
	}

	// Classifier-based three-level routing: 1=local, 2=primary, 3=reasoner.
	// ClassifierMinLen > 0: only classify messages longer than threshold.
	// ClassifierMinLen == 0: always classify (classifier is a free local model).
	// ClassifierMinLen < 0: disabled entirely.
	if cfg.ClassifierMinLen >= 0 && r.get(cfg.Classifier) != nil {
		text := lastUserText(messages)
		if text != "" && (cfg.ClassifierMinLen == 0 || len([]rune(text)) >= cfg.ClassifierMinLen) {
			level := r.classify(ctx, text)
			switch level {
			case 1:
				if p := r.get(cfg.Local); p != nil {
					slog.Info("routing", "reason", "classifier→local", "level", 1, "provider", p.Name())
					return p
				}
			case 3:
				if p := r.get(cfg.Reasoner); p != nil {
					slog.Info("routing", "reason", "classifier→reasoner", "level", 3, "provider", p.Name())
					return p
				}
			}
			// level 2 or fallback → primary
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

func (r *Router) keyFor(p Provider) string {
	for k, v := range r.providers {
		if v == p {
			return k
		}
	}
	return ""
}

// classify calls the classifier provider to rate message complexity.
// Returns 1 (simple/local), 2 (moderate/primary), or 3 (hard/reasoner).
// Defaults to 2 on any error.
func (r *Router) classify(ctx context.Context, text string) int {
	r.mu.RLock()
	classifierKey := r.cfg.Classifier
	primaryKey := r.cfg.Primary
	r.mu.RUnlock()

	if classifierKey == "" {
		classifierKey = primaryKey
	}
	provider := r.get(classifierKey)
	if provider == nil {
		return 2
	}
	timeout := time.Duration(r.cfg.ClassifierTimeout) * time.Second
	if timeout <= 0 {
		timeout = 15 * time.Second
	}
	classifierCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Truncate to save tokens — classifier only needs the beginning to judge complexity.
	classifierText := text
	if len([]rune(classifierText)) > 500 {
		classifierText = string([]rune(classifierText)[:500])
	}
	prompt := r.cfg.ClassifierPrompt
	if prompt == "" {
		prompt = defaultClassifierPrompt
	}
	msgs := []Message{{Role: "user", Content: classifierText}}
	resp, err := provider.Chat(classifierCtx, msgs, prompt, nil)
	if err != nil {
		r.logger.Warn("classifier error, falling back to primary", "classifier", classifierKey, "err", err)
		return 2
	}
	answer := strings.TrimSpace(resp.Content)
	// Extract first digit from response
	for _, c := range answer {
		switch c {
		case '1':
			r.logger.Info("classifier result", "level", 1, "text_len", len(text))
			return 1
		case '2':
			r.logger.Info("classifier result", "level", 2, "text_len", len(text))
			return 2
		case '3':
			r.logger.Info("classifier result", "level", 3, "text_len", len(text))
			return 3
		}
	}
	r.logger.Warn("classifier returned unexpected response, using primary", "response", answer)
	return 2
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

func hasToolMessages(messages []Message) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "tool" || len(messages[i].ToolCalls) > 0 {
			return true
		}
		if messages[i].Role == "user" {
			return false // only look at recent messages after last user msg
		}
	}
	return false
}

func supportsVision(p Provider) bool {
	if vp, ok := p.(VisionProvider); ok {
		return vp.SupportsVision()
	}
	return false
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
