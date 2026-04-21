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
	Compaction        string // provider used for history summarisation
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
	Compaction       string `json:"compaction,omitempty"`
	ClassifierMinLen *int   `json:"classifier_min_len,omitempty"`
	// Legacy shape: slot name → model id. Stored when every slot was
	// OpenRouter-backed. Read on startup and migrated to SlotOverrides.
	OpenRouterModels map[string]string `json:"openrouter_models,omitempty"`
	// Per-slot provider+model overrides. Applied on startup by main.go via
	// TakePendingSlotOverrides after LoadPersistedOverrides. Lets the admin
	// UI change a slot's backend type (e.g. `simple` from OR to Gemini)
	// AND its model id atomically.
	SlotOverrides map[string]slotState `json:"slot_overrides,omitempty"`
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
	usage         UsageStore    // optional; when set, Router records UsageLog after every call
	lastRouted    string        // display name of last provider (for UI)
	lastRoutedKey string        // map key of last provider (for tool continuation)
	// Set by LoadPersistedOverrides; main.go drains via TakePendingSlotOverrides
	// to apply SetProviderModel on each reconfigurable slot (needs CapabilityStore).
	pendingSlotOverrides map[string]slotState

	// Factories per provider type. Used by SetProviderModel to rebuild a slot's
	// underlying Provider when the admin UI swaps its backend type. Populated
	// after construction via RegisterBackendFactories.
	factories map[string]BackendFactory

	// Current provider type per slot. Populated from config + persisted on swap.
	// Read via SlotProvider(name); written under mu by SetProviderModel.
	slotTypes map[string]string

	OnFallback func(from, to string)
	logger     *slog.Logger
}

// slotState is the persisted shape of a single slot — provider type AND
// model id together. Replaces the legacy "OpenRouterModels" map which could
// only carry model id.
type slotState struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// SetUsageStore wires the UsageLog sink. Calls before Chat() is invoked take
// effect on the next request; Chat() reads the field under r.mu.
func (r *Router) SetUsageStore(u UsageStore) {
	r.mu.Lock()
	r.usage = u
	r.mu.Unlock()
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
	// Only honor persisted role→slot pointers when the slot still exists in
	// the provider map. A stale override (e.g. a slot renamed or dropped
	// from config) would silently break routing — better to fall back to
	// the config default and log the discarded value.
	slotExists := func(name string) bool {
		_, ok := r.providers[name]
		return ok
	}
	applyIfExists := func(role, current, persisted string) string {
		if persisted == "" {
			return current
		}
		if !slotExists(persisted) {
			r.logger.Warn("discarding stale routing override — slot missing",
				"role", role, "persisted", persisted, "keeping", current)
			return current
		}
		return persisted
	}
	r.cfg.Simple = applyIfExists("simple", r.cfg.Simple, ov.Simple)
	r.cfg.Default = applyIfExists("default", r.cfg.Default, ov.Default)
	r.cfg.Fallback = applyIfExists("fallback", r.cfg.Fallback, ov.Fallback)
	r.cfg.Multimodal = applyIfExists("multimodal", r.cfg.Multimodal, ov.Multimodal)
	r.cfg.Complex = applyIfExists("complex", r.cfg.Complex, ov.Complex)
	r.cfg.Classifier = applyIfExists("classifier", r.cfg.Classifier, ov.Classifier)
	r.cfg.Compaction = applyIfExists("compaction", r.cfg.Compaction, ov.Compaction)
	if ov.ClassifierMinLen != nil {
		r.cfg.ClassifierMinLen = *ov.ClassifierMinLen
	}
	// Stash per-slot overrides for main.go to apply (needs CapabilityStore).
	// Prefer the new SlotOverrides shape; fall back to the legacy
	// OpenRouterModels map (assumes provider "openrouter" for every entry).
	r.pendingSlotOverrides = make(map[string]slotState)
	for k, v := range ov.SlotOverrides {
		r.pendingSlotOverrides[k] = v
	}
	for k, v := range ov.OpenRouterModels {
		if _, seen := r.pendingSlotOverrides[k]; seen {
			continue // new shape wins
		}
		r.pendingSlotOverrides[k] = slotState{Provider: "openrouter", Model: v}
	}
	if len(r.pendingSlotOverrides) == 0 {
		r.pendingSlotOverrides = nil
	}
	return nil
}

// TakePendingSlotOverrides returns the loaded per-slot (provider, model)
// overrides and clears the internal buffer. Call once after LoadPersistedOverrides.
func (r *Router) TakePendingSlotOverrides() map[string]slotState {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := r.pendingSlotOverrides
	r.pendingSlotOverrides = nil
	return m
}

// SlotState is a public view of a slot's current (provider, model) pair, used
// by the admin UI.
type SlotState = slotState

// SlotProvider returns the provider type currently backing the given slot,
// or "" if unknown. Used by the admin UI to render the current selection.
func (r *Router) SlotProvider(slot string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.slotTypes[slot]
}

// RegisterBackendFactories attaches the per-type factory map (built by
// BuildBackendFactories from the startup config) to the router. Must be
// called before SetProviderModel is used with a different provider type.
// Also seeds slotTypes from the initial providers map so persistence knows
// each slot's starting backend.
func (r *Router) RegisterBackendFactories(factories map[string]BackendFactory, initialTypes map[string]string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.factories = factories
	if r.slotTypes == nil {
		r.slotTypes = map[string]string{}
	}
	for k, v := range initialTypes {
		r.slotTypes[k] = v
	}
}

// currentSlotStates collects (provider, model) for every slot that has an
// active entry in slotTypes. Uses CurrentModel when the provider implements
// ConfigurableProvider; else falls back to the model baked into Provider.Name().
// Must be called with r.mu held.
func (r *Router) currentSlotStates() map[string]slotState {
	out := map[string]slotState{}
	for name := range r.providers {
		pType := r.slotTypes[name]
		if pType == "" {
			continue
		}
		var model string
		if p := r.providers[name]; p != nil {
			if cp, ok := p.(ConfigurableProvider); ok {
				model = cp.CurrentModel()
			}
		}
		out[name] = slotState{Provider: pType, Model: model}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// settingsKeyRoutingOverrides is the key under which the JSON blob of routing
// overrides is stored in the settings store.
const settingsKeyRoutingOverrides = "routing.overrides"

// Scalar setting keys for values that were historically in config.yaml and
// are now DB-overridable via the admin UI. Reads use GetIntSetting/
// GetBoolSetting with config-provided defaults.
//
// Note: classifier_min_length is NOT here — it lives in the routing.overrides
// JSON blob (legacy persistence) and is set via Router.SetClassifierMinLen.
const (
	SettingKeyClassifierTimeout               = "cfg.classifier_timeout"
	SettingKeyToolFilterTopK                  = "cfg.tool_filter_top_k"
	SettingKeyToolFilterAlwaysIncludeKeywords = "cfg.tool_filter_always_include_keywords"
	SettingKeyWebSearchEnabled                = "cfg.web_search_enabled"
	SettingKeyWebSearchProvider               = "cfg.web_search_provider"
	SettingKeyWebFetchEnabled                 = "cfg.web_fetch_enabled"
	SettingKeyFilesystemEnabled               = "cfg.filesystem_enabled"
	SettingKeyTTSEnabled                      = "cfg.tts_enabled"
	SettingKeyTTSVoice                        = "cfg.tts_voice"
	SettingKeyVoiceAPIChatID                  = "cfg.voice_api_chat_id"
	SettingKeyTrustForwardAuth                = "cfg.admin_trust_forward_auth"
)

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
		Compaction:       r.cfg.Compaction,
		ClassifierMinLen: &minLen,
		SlotOverrides:    r.currentSlotStates(),
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

// SetProviderModel swaps the backend behind a slot and/or its model id.
//
// When providerType matches the slot's current type, the underlying provider
// instance is reused — only SetModel is called (cheap; does not churn HTTP
// connection pools). When providerType differs, the router calls the
// registered factory for that type to build a fresh Provider and replaces
// the slot's instance atomically under r.mu. In-flight requests keep the
// old instance; new requests use the new one.
//
// Returns an error if:
//   - the slot is unknown,
//   - providerType is empty and the slot's current provider is not
//     ConfigurableProvider (nothing to swap),
//   - providerType is set but no factory is registered for it,
//   - the factory returns an error constructing the new provider.
func (r *Router) SetProviderModel(slot, providerType, modelID string, caps Capabilities) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	p, ok := r.providers[slot]
	if !ok {
		return errors.New("unknown slot: " + slot)
	}

	currentType := r.slotTypes[slot]
	// Caller didn't specify a type — keep the current one. This is the
	// same-slot reassign path (admin UI "swap model but stay on OR").
	if providerType == "" {
		providerType = currentType
	}

	// Same type — just swap the model id on the existing instance.
	if providerType == currentType || currentType == "" {
		cp, ok := p.(ConfigurableProvider)
		if !ok {
			return errors.New("slot is not reconfigurable: " + slot)
		}
		cp.SetModel(modelID, caps)
		if r.slotTypes == nil {
			r.slotTypes = map[string]string{}
		}
		r.slotTypes[slot] = providerType
		r.saveOverrides()
		return nil
	}

	// Cross-type swap — rebuild via factory.
	factory, err := factoryForType(r.factories, providerType)
	if err != nil {
		return err
	}
	newProv, err := factory(modelID, caps)
	if err != nil {
		return fmt.Errorf("build %s provider: %w", providerType, err)
	}
	r.providers[slot] = newProv
	if r.slotTypes == nil {
		r.slotTypes = map[string]string{}
	}
	r.slotTypes[slot] = providerType
	r.saveOverrides()
	return nil
}

// SetRole updates a routing role to point at the given model name.
// Valid roles: simple, default, complex, fallback, classifier, multimodal, compaction.
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
	case "compaction":
		r.cfg.Compaction = model
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
	provider, role := r.pick(ctx, messages)
	r.mu.Lock()
	r.lastRouted = provider.Name()
	r.lastRoutedKey = r.keyFor(provider)
	r.mu.Unlock()

	start := time.Now()
	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	r.recordUsage(ctx, provider, role, resp, err, time.Since(start))

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
			fbStart := time.Now()
			resp, err = next.Chat(ctx, messages, systemPrompt, tools)
			r.recordUsage(ctx, next, "fallback", resp, err, time.Since(fbStart))
			if err == nil || !isUnavailable(err) {
				break
			}
			provider = next // track last tried for next iteration
		}
	}
	return resp, err
}

// RecordCall logs a UsageLog row for a provider call that was dispatched
// directly (bypassing Router.Chat) — e.g. compaction in agent.Compacter
// which already knows its target provider by role. Callers are responsible
// for measuring latency and passing the response/error unchanged.
func (r *Router) RecordCall(ctx context.Context, p Provider, role string, resp Response, callErr error, latency time.Duration) {
	r.recordUsage(ctx, p, role, resp, callErr, latency)
}

// recordUsage persists a UsageLog row for one LLM call. Runs best-effort —
// errors are logged but never fail the request. Extracts turn meta from ctx
// when present; falls back to 0s when absent.
func (r *Router) recordUsage(ctx context.Context, p Provider, role string, resp Response, callErr error, latency time.Duration) {
	r.mu.RLock()
	store := r.usage
	r.mu.RUnlock()
	if store == nil {
		return
	}
	meta, _ := TurnMetaFrom(ctx)
	if meta.RoleHint != "" {
		role = meta.RoleHint
	}
	provKind, modelID := splitProviderName(p.Name())
	log := UsageLog{
		Provider:           provKind,
		ModelID:            modelID,
		Role:               role,
		ChatID:             meta.ChatID,
		PromptTokens:       resp.Usage.PromptTokens,
		CompletionTokens:   resp.Usage.CompletionTokens,
		CachedPromptTokens: resp.Usage.CachedPromptTokens,
		ReasoningTokens:    resp.Usage.ReasoningTokens,
		Cost:               resp.Usage.Cost,
		LatencyMs:          int(latency / time.Millisecond),
		Success:            callErr == nil,
		ErrorClass:         ClassifyErrorClass(callErr),
		RequestID:          resp.Usage.RequestID,
		ToolCallCount:      len(resp.ToolCalls),
	}
	if meta.UserMessageID != 0 {
		id := meta.UserMessageID
		log.UserMessageID = &id
	}
	// Best-effort write with short timeout — usage logging must never block or
	// fail a user request.
	writeCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	id, err := store.PutUsage(writeCtx, log)
	if err != nil {
		r.logger.Warn("usage log write failed", "err", err, "provider", provKind, "model", modelID)
		return
	}
	// Expose the new row's id so agent.Process can backfill assistant_message_id.
	if meta.LastUsageID != nil {
		*meta.LastUsageID = id
	}
}

// splitProviderName breaks a "provider/model" Name() into its parts. For
// providers whose Name() does not contain a slash, provider = full name and
// model = "".
func splitProviderName(name string) (provider, model string) {
	if i := strings.IndexByte(name, '/'); i >= 0 {
		return name[:i], name[i+1:]
	}
	return name, ""
}

// ChatStream returns a streaming channel if the current provider supports it.
// Falls back to wrapping a synchronous Chat() in a single-chunk channel.
func (r *Router) ChatStream(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error) {
	provider, role := r.pick(ctx, messages)
	r.mu.Lock()
	r.lastRouted = provider.Name()
	r.lastRoutedKey = r.keyFor(provider)
	r.mu.Unlock()
	if sp, ok := provider.(StreamProvider); ok {
		ch, err := sp.ChatStream(ctx, messages, systemPrompt, tools)
		if err != nil {
			// Even a start-of-stream failure deserves a usage record with
			// error_class set so dashboards see it.
			r.recordUsage(ctx, provider, role, Response{}, err, 0)
			if isUnavailable(err) {
				return r.syncFallbackStream(ctx, provider, messages, systemPrompt, tools, err)
			}
			return nil, err
		}
		return r.instrumentedStream(ctx, provider, role, ch), nil
	}
	// Provider does not stream — wrap synchronous call.
	return r.syncStream(ctx, provider, role, messages, systemPrompt, tools)
}

// instrumentedStream tees the provider's stream to the caller and records a
// UsageLog entry when the terminal chunk arrives. Usage data comes from the
// provider (openai_compat parses it out of the SSE payload); when absent the
// record still lands with zero tokens so dashboards show the call happened.
func (r *Router) instrumentedStream(ctx context.Context, provider Provider, role string, upstream <-chan StreamChunk) <-chan StreamChunk {
	out := make(chan StreamChunk, 64)
	start := time.Now()
	go func() {
		defer close(out)
		var finalResp Response
		var finalErr error
		for chunk := range upstream {
			out <- chunk
			if chunk.Done {
				finalResp = Response{ToolCalls: chunk.ToolCalls, Usage: chunk.Usage}
				finalErr = chunk.Err
			}
		}
		r.recordUsage(ctx, provider, role, finalResp, finalErr, time.Since(start))
	}()
	return out
}

// syncStream wraps a synchronous Chat() call in a channel. Usage is recorded
// via the same path as Router.Chat so streaming callers that hit non-stream
// providers still produce UsageLog rows.
func (r *Router) syncStream(ctx context.Context, provider Provider, role string, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error) {
	start := time.Now()
	resp, err := provider.Chat(ctx, messages, systemPrompt, tools)
	r.recordUsage(ctx, provider, role, resp, err, time.Since(start))
	if err != nil {
		return nil, err
	}
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Delta: resp.Content, ToolCalls: resp.ToolCalls, Done: true, Usage: resp.Usage}
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
		// Try streaming on fallback if supported — still instrumented.
		if sp, ok := next.(StreamProvider); ok {
			ch, err := sp.ChatStream(ctx, messages, systemPrompt, tools)
			if err == nil {
				return r.instrumentedStream(ctx, next, "fallback", ch), nil
			}
			r.recordUsage(ctx, next, "fallback", Response{}, err, 0)
		}
		// Otherwise sync.
		return r.syncStream(ctx, next, "fallback", messages, systemPrompt, tools)
	}
	return nil, origErr
}

// SupportsStreaming returns true if the currently active provider implements StreamProvider.
func (r *Router) SupportsStreaming() bool {
	provider, _ := r.pick(context.Background(), nil)
	_, ok := provider.(StreamProvider)
	return ok
}

// Name returns the name of the currently active provider (respecting override).
func (r *Router) Name() string {
	p, _ := r.pick(context.Background(), nil)
	return p.Name()
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

// pick selects the provider for this request AND returns the logical routing
// role that drove the decision (simple/default/complex/multimodal/override/
// tool-continuation). Role is what ends up in usage_log — conceptually
// useful for analytics. Separate from the provider's slot key (e.g. the same
// "simple" role may be backed by different slot names across deployments).
func (r *Router) pick(ctx context.Context, messages []Message) (Provider, string) {
	r.mu.RLock()
	cfg := r.cfg
	override := r.override
	lastRoutedKey := r.lastRoutedKey
	r.mu.RUnlock()

	multimodal := hasMultimodalContent(messages)

	// Per-request forced role from ctx (e.g. admin UI "Complex" dropdown).
	// Takes precedence over the global override so different clients can pin
	// different roles without fighting each other.
	if forced := forcedRoleFrom(ctx); forced != "" {
		var key, roleName string
		switch forced {
		case "simple":
			key, roleName = cfg.Simple, "simple"
		case "default":
			key, roleName = cfg.Default, "default"
		case "complex":
			key, roleName = cfg.Complex, "complex"
		}
		if p := r.get(key); p != nil {
			if !multimodal || supportsVision(p) {
				slog.Info("routing", "reason", "forced", "role", forced, "provider", p.Name())
				return p, roleName
			}
		}
	}

	if override != "" {
		if p := r.get(override); p != nil {
			if !multimodal || supportsVision(p) {
				slog.Info("routing", "reason", "override", "provider", p.Name())
				return p, "override"
			}
			// Override doesn't support vision — fall through to multimodal
		}
	}

	// Multimodal routing — prefer simple/default if they support vision
	if multimodal {
		if p := r.get(cfg.Simple); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "simple+vision", "provider", p.Name())
			return p, "simple"
		}
		if p := r.get(cfg.Default); p != nil && supportsVision(p) {
			slog.Info("routing", "reason", "default+vision", "provider", p.Name())
			return p, "default"
		}
		if p := r.get(cfg.Multimodal); p != nil {
			slog.Info("routing", "reason", "multimodal", "provider", p.Name())
			return p, "multimodal"
		}
	}

	// Tool continuation — keep using the same provider that started the tool loop.
	if hasToolMessages(messages) && lastRoutedKey != "" {
		if last := r.get(lastRoutedKey); last != nil {
			slog.Info("routing", "reason", "tool-continuation", "provider", last.Name())
			return last, "tool-continuation"
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
					return p, "simple"
				}
			case 3:
				if p := r.get(cfg.Complex); p != nil {
					slog.Info("routing", "reason", "classifier→complex", "level", 3, "provider", p.Name())
					return p, "complex"
				}
			}
			// level 2 or fallback → default
		}
	}

	if p := r.get(cfg.Default); p != nil {
		slog.Info("routing", "reason", "default", "provider", p.Name())
		return p, "default"
	}
	// Should never happen if config is valid
	for _, p := range r.providers {
		return p, "default"
	}
	return nil, ""
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
	// Timeout: DB override wins, then cfg, then 15s default.
	r.mu.RLock()
	settings := r.settings
	cfgTimeout := r.cfg.ClassifierTimeout
	r.mu.RUnlock()
	timeoutSec := GetIntSetting(ctx, settings, SettingKeyClassifierTimeout, cfgTimeout)
	if timeoutSec <= 0 {
		timeoutSec = 15
	}
	classifierCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
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
	// DB override takes precedence over config/default.
	if settings != nil {
		if v, ok, _ := settings.GetSetting(classifierCtx, "prompts.classifier"); ok && v != "" {
			prompt = v
		}
	}
	msgs := []Message{{Role: "user", Content: classifierText}}
	classifyStart := time.Now()
	resp, err := provider.Chat(classifierCtx, msgs, prompt, nil)
	classifyDur := time.Since(classifyStart)
	if t := timingsFrom(ctx); t != nil {
		t.ClassifyDur = classifyDur
		t.ClassifyRan = true
	}
	// Record classifier usage separately so it shows up in dashboards distinct
	// from the main call. The classifier inherits the turn's ChatID but marks
	// itself as role="classifier" regardless of which provider backs it.
	r.recordUsage(
		WithTurnMeta(ctx, TurnMeta{
			ChatID:        turnChatID(ctx),
			UserMessageID: turnUserMsgID(ctx),
			RoleHint:      "classifier",
		}),
		provider, "classifier", resp, err, classifyDur,
	)
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
