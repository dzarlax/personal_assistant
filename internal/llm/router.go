package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
// Role names are chosen to match RoutingConfig in internal/config:
// simple (L1) → default (L2) → complex (L3).
type RouterConfig struct {
	Simple            string // level 1: simple/cheap tasks
	Default           string // level 2: moderate tasks (agentic loop)
	Complex           string // level 3: complex reasoning
	Fallback          string
	Multimodal        string
	Classifier        string // provider used for complexity classification
	ClassifierMinLen  int    // min rune length to run classifier; 0 = always; <0 = disabled
	ClassifierTimeout int    // seconds; default 15
	ClassifierPrompt  string // system prompt for classifier; uses default if empty
}

// routingOverrides is the persisted subset of RouterConfig (JSON blob in
// kv_settings.routing.overrides).
type routingOverrides struct {
	Simple           string `json:"simple,omitempty"`
	Default          string `json:"default,omitempty"`
	Complex          string `json:"complex,omitempty"`
	Fallback         string `json:"fallback,omitempty"`
	Multimodal       string `json:"multimodal,omitempty"`
	Classifier       string `json:"classifier,omitempty"`
	ClassifierMinLen *int   `json:"classifier_min_len,omitempty"`
	// Per-slot OpenRouter model overrides (slot name → model id). Lets the admin
	// UI swap the model backing e.g. `workhorse` to `anthropic/claude-sonnet-4.5`
	// without editing config.yaml. Applied by main.go after LoadPersistedOverrides.
	OpenRouterModels map[string]string `json:"openrouter_models,omitempty"`
}

// Router selects the appropriate LLM provider based on context.
// All providers are stored by name; special roles reference names from RouterConfig.
type Router struct {
	providers map[string]Provider
	cfg       RouterConfig

	mu          sync.RWMutex
	override    string // set via SetOverride; any key in providers, or "" for auto
	persistPath string // legacy: file for routing overrides. Read once on first start
	//                   for migration, then the binary writes to the settings store.
	settings      SettingsStore // primary persistence; DB-backed
	lastRouted    string        // display name of last provider (for UI)
	lastRoutedKey string        // map key of last provider (for tool continuation)
	// Set by LoadPersistedOverrides; main.go drains via TakePendingOpenRouterOverrides
	// to apply SetModel on each OR-backed provider (needs CapabilityStore).
	pendingOpenRouterOverrides map[string]string

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

// SetPersistPath sets a legacy file path. Used only on first start when the
// settings store is empty — the file contents are imported and the file is
// removed. After that all writes go through the settings store.
func (r *Router) SetPersistPath(path string) {
	r.mu.Lock()
	r.persistPath = path
	r.mu.Unlock()
}

// SetSettingsStore wires the DB-backed persistence. Call before LoadPersistedOverrides.
func (r *Router) SetSettingsStore(s SettingsStore) {
	r.mu.Lock()
	r.settings = s
	r.mu.Unlock()
}

// LoadPersistedOverrides loads from the settings store. If the store is empty
// and a legacy file exists at persistPath, it's imported and removed.
func (r *Router) LoadPersistedOverrides() error {
	r.mu.RLock()
	settings := r.settings
	path := r.persistPath
	r.mu.RUnlock()

	var data []byte
	// 1. Prefer the settings store.
	if settings != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		v, ok, err := settings.GetSetting(ctx, settingsKeyRoutingOverrides)
		cancel()
		if err != nil {
			return fmt.Errorf("load routing overrides: %w", err)
		}
		if ok {
			data = []byte(v)
		}
	}
	// 2. Legacy migration: if store empty and file exists, import it.
	if len(data) == 0 && path != "" {
		fileData, err := os.ReadFile(path)
		if err == nil {
			data = fileData
			// Write to store immediately so next start doesn't re-read the file.
			if settings != nil {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				if putErr := settings.PutSetting(ctx, settingsKeyRoutingOverrides, string(fileData)); putErr == nil {
					// Successfully migrated — remove the legacy file.
					_ = os.Remove(path)
					r.logger.Info("migrated routing.json to settings store", "path", path)
				}
				cancel()
			}
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	if len(data) == 0 {
		return nil
	}

	var ov routingOverrides
	if err := json.Unmarshal(data, &ov); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if ov.Simple != "" {
		r.cfg.Simple = ov.Simple
	}
	if ov.Default != "" {
		r.cfg.Default = ov.Default
	}
	if ov.Fallback != "" {
		r.cfg.Fallback = ov.Fallback
	}
	if ov.Multimodal != "" {
		r.cfg.Multimodal = ov.Multimodal
	}
	if ov.Complex != "" {
		r.cfg.Complex = ov.Complex
	}
	if ov.Classifier != "" {
		r.cfg.Classifier = ov.Classifier
	}
	if ov.ClassifierMinLen != nil {
		r.cfg.ClassifierMinLen = *ov.ClassifierMinLen
	}
	// Stash OpenRouter model overrides for main.go to apply (needs CapabilityStore).
	if len(ov.OpenRouterModels) > 0 {
		r.pendingOpenRouterOverrides = make(map[string]string, len(ov.OpenRouterModels))
		for k, v := range ov.OpenRouterModels {
			r.pendingOpenRouterOverrides[k] = v
		}
	}
	return nil
}

// TakePendingOpenRouterOverrides returns the loaded per-slot model overrides
// and clears the internal buffer. Call once after LoadPersistedOverrides.
func (r *Router) TakePendingOpenRouterOverrides() map[string]string {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.pendingOpenRouterOverrides
	r.pendingOpenRouterOverrides = nil
	return m
}

// currentOpenRouterModels collects the current model id of every provider that
// implements ConfigurableProvider (OR-backed in practice).
// Must be called with r.mu held.
func (r *Router) currentOpenRouterModels() map[string]string {
	out := map[string]string{}
	for name, p := range r.providers {
		if cp, ok := p.(ConfigurableProvider); ok {
			out[name] = cp.CurrentModel()
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// settingsKeyRoutingOverrides is the key under which the JSON blob of routing
// overrides is stored in the settings store.
const settingsKeyRoutingOverrides = "routing.overrides"

// saveOverrides writes current cfg via the settings store (preferred) or the
// legacy file path. Must be called with mu held.
func (r *Router) saveOverrides() {
	if r.settings == nil && r.persistPath == "" {
		return
	}
	minLen := r.cfg.ClassifierMinLen
	ov := routingOverrides{
		Simple:           r.cfg.Simple,
		Default:          r.cfg.Default,
		Complex:          r.cfg.Complex,
		Fallback:         r.cfg.Fallback,
		Multimodal:       r.cfg.Multimodal,
		Classifier:       r.cfg.Classifier,
		ClassifierMinLen: &minLen,
		OpenRouterModels: r.currentOpenRouterModels(),
	}
	data, err := json.Marshal(ov)
	if err != nil {
		return
	}
	if r.settings != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := r.settings.PutSetting(ctx, settingsKeyRoutingOverrides, string(data)); err != nil {
			r.logger.Error("failed to save routing overrides", "err", err)
			return
		}
		r.logger.Info("routing overrides saved")
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

// Provider returns the provider registered under name, or (nil, false). Used
// by external packages (admin API) to inspect runtime state.
func (r *Router) Provider(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[name]
	return p, ok
}

// SetProviderModel swaps the underlying model id of a provider slot and
// persists the choice. Returns an error if the slot does not exist or does
// not support runtime reconfiguration.
func (r *Router) SetProviderModel(slot, modelID string, caps Capabilities) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.providers[slot]
	if !ok {
		return errors.New("unknown slot: " + slot)
	}
	cp, ok := p.(ConfigurableProvider)
	if !ok {
		return errors.New("slot is not reconfigurable: " + slot)
	}
	cp.SetModel(modelID, caps)
	r.saveOverrides()
	return nil
}

// SetRole updates a routing role to point at the given model name.
// Valid roles: simple, default, complex, fallback, classifier, multimodal.
func (r *Router) SetRole(role, model string) error {
	if _, ok := r.providers[model]; !ok {
		return errors.New("unknown model: " + model)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	switch role {
	case "simple":
		r.cfg.Simple = model
	case "default":
		r.cfg.Default = model
	case "complex":
		r.cfg.Complex = model
	case "fallback":
		r.cfg.Fallback = model
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
		chain := []string{r.cfg.Default, r.cfg.Fallback}
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
	chain := []string{r.cfg.Default, r.cfg.Fallback}
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

	// Multimodal routing — prefer simple/default if they support vision
	if multimodal {
		if p := r.get(cfg.Simple); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "simple+vision", "provider", p.Name())
			return p
		}
		if p := r.get(cfg.Default); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "default+vision", "provider", p.Name())
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

	// Classifier-based three-level routing: 1=simple, 2=default, 3=complex.
	// ClassifierMinLen > 0: only classify messages longer than threshold.
	// ClassifierMinLen == 0: always classify (classifier is a free local model).
	// ClassifierMinLen < 0: disabled entirely.
	if cfg.ClassifierMinLen >= 0 && r.get(cfg.Classifier) != nil {
		text := lastUserText(messages)
		if text != "" && (cfg.ClassifierMinLen == 0 || len([]rune(text)) >= cfg.ClassifierMinLen) {
			level := r.classify(ctx, text)
			switch level {
			case 1:
				if p := r.get(cfg.Simple); p != nil {
					slog.Info("routing", "reason", "classifier→simple", "level", 1, "provider", p.Name())
					return p
				}
			case 3:
				if p := r.get(cfg.Complex); p != nil {
					slog.Info("routing", "reason", "classifier→complex", "level", 3, "provider", p.Name())
					return p
				}
			}
			// level 2 or fallback → default
		}
	}

	if p := r.get(cfg.Default); p != nil {
		slog.Info("routing", "reason", "default", "provider", p.Name())
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
// Returns 1 (simple), 2 (default), or 3 (complex).
// Defaults to 2 on any error.
func (r *Router) classify(ctx context.Context, text string) int {
	r.mu.RLock()
	classifierKey := r.cfg.Classifier
	defaultKey := r.cfg.Default
	r.mu.RUnlock()

	if classifierKey == "" {
		classifierKey = defaultKey
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
	classifyStart := time.Now()
	resp, err := provider.Chat(classifierCtx, msgs, prompt, nil)
	classifyDur := time.Since(classifyStart)
	if t := timingsFrom(ctx); t != nil {
		t.ClassifyDur = classifyDur
		t.ClassifyRan = true
	}
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
