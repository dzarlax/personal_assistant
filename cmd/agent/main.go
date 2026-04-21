package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"telegram-agent/internal/adminapi"
	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
	"telegram-agent/internal/mcp"
	"telegram-agent/internal/store"
	"telegram-agent/internal/telegram"
	"telegram-agent/internal/voiceapi"
)

// splitKeywords parses a comma-separated setting value into a trimmed,
// non-empty list. Used for tool_filter.always_include_keywords.
func splitKeywords(raw string) []string {
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg, err := config.Load("config/config.yaml")
	if err != nil {
		logger.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	sysPromptPath := "config/system_prompt.md"
	if cfg.Filesystem.Enabled && cfg.Filesystem.Root != "" {
		candidate := cfg.Filesystem.Root + "/CLAUDE.md"
		if _, statErr := os.Stat(candidate); statErr == nil {
			sysPromptPath = candidate
		}
	}
	sysPromptBytes, err := os.ReadFile(sysPromptPath)
	if err != nil {
		logger.Error("failed to load system prompt", "err", err, "path", sysPromptPath)
		os.Exit(1)
	}
	logger.Info("system prompt loaded", "path", sysPromptPath)

	// Build providers map — all named LLM providers available for routing and /model switching.
	providers := make(map[string]llm.Provider)

	addProvider := func(key string, p llm.Provider, e error) {
		if e != nil {
			logger.Warn("failed to init LLM provider", "key", key, "err", e)
			return
		}
		providers[key] = p
	}

	// All providers are optional — at least the one referenced by routing.default must be present.
	// Dispatch each models.* entry by its `provider` field. The special key
	// "embedding" is reserved for the MCP embedding provider (handled below).
	for name, mc := range cfg.Models {
		if name == "embedding" {
			continue
		}
		// Fill in well-known defaults (base URLs etc.) so config entries can
		// omit them. See llm.ApplyProviderDefaults for the policy.
		mc = llm.ApplyProviderDefaults(mc)
		switch mc.Provider {
		case "openrouter":
			if mc.APIKey == "" || mc.Model == "" {
				continue
			}
			p, e := llm.NewOpenRouter(mc)
			addProvider(name, p, e)
		case "gemini":
			if mc.APIKey == "" {
				continue
			}
			p, e := llm.NewGeminiNative(mc)
			addProvider(name, p, e)
		case "ollama":
			if mc.Model == "" {
				continue
			}
			p, e := llm.NewOllama(mc)
			addProvider(name, p, e)
		case "claude-bridge":
			// base_url now has a default (host.docker.internal:9900); api_key
			// (bridge token) is still mandatory.
			if mc.APIKey == "" {
				continue
			}
			p, e := llm.NewClaudeBridge(mc)
			addProvider(name, p, e)
		case "local":
			if mc.BaseURL == "" {
				continue
			}
			p, e := llm.NewLocal(mc)
			addProvider(name, p, e)
		case "":
			logger.Warn("model entry has no provider — skipped", "model", name)
		default:
			logger.Warn("unknown provider — skipped", "model", name, "provider", mc.Provider)
		}
	}

	// Ensure the default routing provider is available.
	if providers[cfg.Routing.Default] == nil {
		logger.Error("default routing provider not configured or failed to init", "provider", cfg.Routing.Default)
		os.Exit(1)
	}

	// Optional roles fall through to routing.default when unspecified.
	multimodalKey := cfg.Routing.Multimodal
	if multimodalKey == "" {
		multimodalKey = cfg.Routing.Default
	}
	complexKey := cfg.Routing.Complex
	if complexKey == "" {
		complexKey = cfg.Routing.Default
	}
	simpleKey := cfg.Routing.Simple
	if simpleKey == "" {
		simpleKey = cfg.Routing.Default
	}

	compactionKey := cfg.Routing.Compaction
	if compactionKey == "" {
		compactionKey = cfg.Routing.Default
	}

	router := llm.NewRouter(providers, llm.RouterConfig{
		Simple:            simpleKey,
		Default:           cfg.Routing.Default,
		Complex:           complexKey,
		Fallback:          cfg.Routing.Fallback,
		Multimodal:        multimodalKey,
		Classifier:        cfg.Routing.Classifier,
		Compaction:        compactionKey,
		ClassifierMinLen:  cfg.Routing.ClassifierMinLength,
		ClassifierTimeout: cfg.Routing.ClassifierTimeout,
		ClassifierPrompt:  cfg.Routing.ClassifierPrompt,
	})

	// Warn about routing roles that reference missing providers
	for role, model := range map[string]string{
		"simple":     simpleKey,
		"fallback":   cfg.Routing.Fallback,
		"multimodal": multimodalKey,
		"complex":    complexKey,
		"classifier": cfg.Routing.Classifier,
	} {
		if model != "" && providers[model] == nil {
			logger.Warn("routing role references unconfigured provider (role will be skipped)",
				"role", role, "model", model)
		}
	}

	// Init store: PostgreSQL (DATABASE_URL) → SQLite fallback → memory fallback
	var s store.Store
	if dbURL := os.Getenv("DATABASE_URL"); dbURL != "" {
		pg, err := store.NewPostgres(context.Background(), dbURL)
		if err != nil {
			logger.Error("failed to init PostgreSQL, falling back to SQLite", "err", err)
		} else {
			logger.Info("using PostgreSQL store")
			s = pg
			defer pg.Close()
		}
	}
	if s == nil {
		dataDir := "data"
		if err := os.MkdirAll(dataDir, 0755); err == nil {
			sqlite, err := store.NewSQLite(filepath.Join(dataDir, "conversations.db"))
			if err != nil {
				logger.Warn("failed to init SQLite, using memory store", "err", err)
				s = store.NewMemory()
			} else {
				logger.Info("using SQLite store")
				s = sqlite
			}
		} else {
			s = store.NewMemory()
		}
	}

	// Hydrate OpenRouter capabilities: one fetch of /api/v1/models, upsert to
	// CapabilityStore, then apply caps to each OR-backed provider so the router
	// can make vision/tool-aware decisions.
	hydrateOpenRouterCapabilities(cfg, providers, s, logger)
	// Same for Gemini: list /v1beta/models, merge in hardcoded prices.
	hydrateGeminiCapabilities(cfg, providers, s, logger)
	// Overlay Artificial Analysis Intelligence Index scores onto cached caps.
	hydrateArtificialAnalysisScores(cfg, s, logger)

	// Persist routing overrides across restarts via the DB settings store.
	// The legacy file path is kept only for one-time migration on first start.
	if ss, ok := s.(llm.SettingsStore); ok {
		router.SetSettingsStore(ss)
	}
	router.SetPersistPath("config/routing.json")
	if err := router.LoadPersistedOverrides(); err != nil {
		logger.Warn("failed to load routing overrides", "err", err)
	}
	// Wire usage log sink (best-effort: router tolerates nil store).
	if us, ok := s.(llm.UsageStore); ok {
		router.SetUsageStore(us)
		logger.Info("usage logging enabled")
	}
	// Register backend factories so the admin UI can rebuild a slot's provider
	// when its backend type changes (e.g. simple: OR → Gemini).
	initialTypes := map[string]string{}
	for name, mc := range cfg.Models {
		if mc.Provider != "" && mc.Provider != "hf-tei" && mc.Provider != "openai" {
			initialTypes[name] = mc.Provider
		}
	}
	router.RegisterBackendFactories(llm.BuildBackendFactories(cfg), initialTypes)

	// Apply persisted per-slot overrides (provider + model). Pull caps from
	// the store keyed by the slot's provider type.
	if overrides := router.TakePendingSlotOverrides(); len(overrides) > 0 {
		applySlotOverrides(router, s, overrides, logger)
	}

	// Compacter resolves its providers from the router on every call, so
	// runtime role changes (admin UI) take effect without a restart.
	compacter := agent.NewCompacter(router)

	// Init MCP client. DB (kv_settings) wins over the legacy mcp.json file —
	// the admin UI writes there, so any user-edited list is authoritative.
	// On first boot with an empty DB, the file is auto-migrated so the next
	// restart can run with the config.yaml volume unmounted.
	var mcpClient *mcp.Client
	var mcpServers map[string]config.MCPServerConfig
	var mcpSource string
	settingsStoreEarly, _ := s.(llm.SettingsStore)
	if settingsStoreEarly != nil {
		mcpCtx, mcpCancel := context.WithTimeout(context.Background(), 3*time.Second)
		if dbServers, found, dbErr := adminapi.LoadMCPServersFromSettings(mcpCtx, settingsStoreEarly); dbErr != nil {
			logger.Warn("failed to load MCP servers from DB; falling back to file", "err", dbErr)
		} else if found {
			mcpServers = dbServers
			mcpSource = "db"
		}
		mcpCancel()
	}
	if mcpServers == nil {
		fileServers, err := config.LoadMCPServers("config/mcp.json")
		if err != nil {
			logger.Warn("failed to load mcp.json", "err", err)
		}
		mcpServers = fileServers
		if fileServers != nil {
			mcpSource = "file"
			// Auto-migrate: write the file contents to kv_settings so future
			// boots don't depend on the file being present.
			if settingsStoreEarly != nil {
				migCtx, migCancel := context.WithTimeout(context.Background(), 3*time.Second)
				if data, mErr := json.Marshal(fileServers); mErr == nil {
					if pErr := settingsStoreEarly.PutSetting(migCtx, adminapi.SettingKeyMCPServers, string(data)); pErr != nil {
						logger.Warn("MCP auto-migration failed", "err", pErr)
					} else {
						logger.Info("MCP config auto-migrated file → DB", "servers", len(fileServers))
					}
				}
				migCancel()
			}
		}
	}
	logger.Info("MCP config loaded", "servers", len(mcpServers), "source", mcpSource)
	if len(mcpServers) > 0 {
		mcpClient = mcp.NewClient(mcpServers, logger)
		initCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		mcpClient.Initialize(initCtx)
		cancel()

		// top_k: DB override wins over config; disables filtering if 0.
		topCtx, topCancel := context.WithTimeout(context.Background(), 2*time.Second)
		ss, _ := s.(llm.SettingsStore)
		topK := llm.GetIntSetting(topCtx, ss, llm.SettingKeyToolFilterTopK, cfg.ToolFilter.TopK)
		topCancel()
		if emb, ok := cfg.Models["embedding"]; ok && topK > 0 && (emb.APIKey != "" || emb.BaseURL != "") {
			mcpClient.EnableEmbeddings(emb, topK)
			embedCtx, embedCancel := context.WithTimeout(context.Background(), 60*time.Second)
			mcpClient.EmbedTools(embedCtx)
			embedCancel()

			// always_include_keywords: substring matches against tool names bypass top-K.
			kwCtx, kwCancel := context.WithTimeout(context.Background(), 2*time.Second)
			kwRaw := llm.GetStringSetting(kwCtx, ss, llm.SettingKeyToolFilterAlwaysIncludeKeywords, strings.Join(cfg.ToolFilter.AlwaysIncludeKeywords, ","))
			kwCancel()
			kw := splitKeywords(kwRaw)
			mcpClient.SetAlwaysIncludeKeywords(kw)
			logger.Info("tool filter enabled", "top_k", topK, "always_include_keywords", kw)
		}
	}

	ag := agent.New(router, s, mcpClient, compacter, string(sysPromptBytes), logger)

	// Enable voice transcription using whichever model serves routing.multimodal
	// (that's the vision/audio-capable model by design — same applies to audio).
	if mm, ok := cfg.Models[cfg.Routing.Multimodal]; ok && mm.Provider == "gemini" && mm.APIKey != "" && mm.Model != "" {
		ag.EnableTranscription(agent.TranscribeConfig{
			Model:  mm.Model,
			APIKey: mm.APIKey,
		})
		logger.Info("voice transcription enabled", "model", mm.Model)
	}

	// Feature flags: DB overrides > config > hardcoded defaults. Scalars with
	// a dedicated kv_settings key can be flipped via the Settings tab without
	// editing config.yaml. Sensitive URLs / keys still come from config (env).
	settingsStore, _ := s.(llm.SettingsStore)
	settingsCtx, settingsCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer settingsCancel()

	webSearchOn := llm.GetBoolSetting(settingsCtx, settingsStore, llm.SettingKeyWebSearchEnabled, cfg.WebSearch.Enabled)
	webSearchProvider := llm.GetStringSetting(settingsCtx, settingsStore, llm.SettingKeyWebSearchProvider, cfg.WebSearch.Provider)
	if webSearchOn {
		if webSearchProvider == "" {
			webSearchProvider = "ollama"
		}
		ag.EnableWebSearch(agent.WebSearchConfig{
			Provider: webSearchProvider,
			BaseURL:  cfg.WebSearch.BaseURL,
			APIKey:   cfg.WebSearch.APIKey,
		})
		logger.Info("web search enabled", "provider", webSearchProvider)
	}

	if llm.GetBoolSetting(settingsCtx, settingsStore, llm.SettingKeyWebFetchEnabled, cfg.WebFetch.Enabled) {
		ag.EnableWebFetch(agent.WebFetchConfig{CDPURL: cfg.WebFetch.CDPURL})
		logger.Info("web fetch enabled", "cdp_fallback", cfg.WebFetch.CDPURL != "")
	}

	if llm.GetBoolSetting(settingsCtx, settingsStore, llm.SettingKeyFilesystemEnabled, cfg.Filesystem.Enabled) {
		root := cfg.Filesystem.Root
		if root == "" {
			root = "/assistant_context"
		}
		ag.EnableFilesystem(agent.FilesystemConfig{Root: root})
		logger.Info("filesystem tools enabled", "root", root)
	}

	if llm.GetBoolSetting(settingsCtx, settingsStore, llm.SettingKeyTTSEnabled, cfg.TTS.Enabled) {
		voice := llm.GetStringSetting(settingsCtx, settingsStore, llm.SettingKeyTTSVoice, cfg.TTS.Voice)
		if voice == "" {
			voice = "ru-RU-DmitryNeural"
		}
		ag.EnableTTS(agent.TTSConfig{
			Voice:  voice,
			Rate:   cfg.TTS.Rate,
			Pitch:  cfg.TTS.Pitch,
			Volume: cfg.TTS.Volume,
		})
		logger.Info("TTS enabled", "voice", voice)
	}

	handler, err := telegram.NewHandler(cfg.Telegram, ag, logger)
	if err != nil {
		logger.Error("failed to init Telegram handler", "err", err)
		os.Exit(1)
	}
	if cfg.AdminAPI.Enabled && cfg.AdminAPI.BaseURL != "" {
		handler.SetAdminBaseURL(cfg.AdminAPI.BaseURL)
	}
	// /mcp update reads the live config from kv_settings (source of truth),
	// falling back to whatever the legacy file holds when the DB is empty.
	handler.SetMCPLoader(func(ctx context.Context) (map[string]config.MCPServerConfig, error) {
		if servers, found, err := adminapi.LoadMCPServersFromSettings(ctx, settingsStore); err != nil {
			return nil, err
		} else if found {
			return servers, nil
		}
		return config.LoadMCPServers("config/mcp.json")
	})

	if cfg.VoiceAPI.Enabled {
		// chat_id: DB override > config. Non-destructive: if no DB value,
		// cfg.VoiceAPI.ChatID stays; if the DB holds 0, treat that as unset.
		voiceCfg := cfg.VoiceAPI
		if chatID := llm.GetIntSetting(settingsCtx, settingsStore, llm.SettingKeyVoiceAPIChatID, 0); chatID != 0 {
			voiceCfg.ChatID = int64(chatID)
		}
		voiceSrv := voiceapi.New(ag, voiceCfg, logger)
		go func() {
			if err := voiceSrv.ListenAndServe(); err != nil {
				logger.Error("voice API error", "err", err)
			}
		}()
		defer voiceSrv.Shutdown(context.Background())
	}

	if cfg.AdminAPI.Enabled {
		capStore, _ := s.(llm.CapabilityStore)
		usageStore, _ := s.(llm.UsageStore)
		// trust_forward_auth: DB override > config.
		adminCfg := cfg.AdminAPI
		adminCfg.TrustForwardAuth = llm.GetBoolSetting(settingsCtx, settingsStore, llm.SettingKeyTrustForwardAuth, adminCfg.TrustForwardAuth)
		adminSrv := adminapi.New(adminCfg, router, capStore, settingsStore, usageStore, cfg, logger)
		adminSrv.SetMCPReloader(ag)
		if err := adminSrv.Start(); err != nil {
			logger.Error("admin API failed to start", "err", err)
		} else {
			defer func() {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				_ = adminSrv.Shutdown(ctx)
			}()
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger.Info("agent started", "model", router.Name(), "providers", len(providers), "mcp_servers", len(mcpServers))
	handler.NotifyMissingRouting()
	handler.Start(ctx)

	logger.Info("draining pending batches")
	drainDone := make(chan struct{})
	go func() {
		handler.Drain()
		close(drainDone)
	}()
	select {
	case <-drainDone:
		logger.Info("drain completed")
	case <-time.After(30 * time.Second):
		logger.Warn("drain timed out after 30s, proceeding with shutdown")
	}

	if mcpClient != nil {
		mcpClient.Close()
	}
	logger.Info("agent stopped")
}

// applySlotOverrides applies persisted per-slot (provider, model) overrides.
// When the persisted provider type differs from the slot's initial type, the
// router rebuilds the provider via the registered factory. Capabilities come
// from the CapabilityStore partition matching the persisted provider type.
func applySlotOverrides(router *llm.Router, s store.Store, overrides map[string]llm.SlotState, logger *slog.Logger) {
	capStore, _ := s.(llm.CapabilityStore)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for slot, st := range overrides {
		var caps llm.Capabilities
		if capStore != nil && st.Provider != "" {
			if c, ok, err := capStore.GetCapabilities(ctx, st.Provider, st.Model); err == nil && ok {
				caps = c
			}
		}
		if err := router.SetProviderModel(slot, st.Provider, st.Model, caps); err != nil {
			logger.Warn("failed to apply slot override", "slot", slot, "provider", st.Provider, "model", st.Model, "err", err)
			continue
		}
		logger.Info("applied slot override", "slot", slot, "provider", st.Provider, "model", st.Model,
			"vision", caps.Vision, "tools", caps.Tools)
	}
}

// hydrateOpenRouterCapabilities runs once at startup. It fetches
// /api/v1/models using the first OpenRouter API key it finds, upserts every
// returned model into the CapabilityStore (if the store supports it), and
// applies the caps to each OpenRouter-backed provider by its current model id.
// On fetch failure, it falls back to whatever is cached in the store.
func hydrateOpenRouterCapabilities(cfg *config.Config, providers map[string]llm.Provider, s store.Store, logger *slog.Logger) {
	var apiKey string
	for _, mc := range cfg.Models {
		if mc.Provider == "openrouter" && mc.APIKey != "" {
			apiKey = mc.APIKey
			break
		}
	}
	if apiKey == "" {
		return
	}

	capStore, _ := s.(llm.CapabilityStore)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	caps, err := llm.FetchOpenRouterModels(ctx, apiKey)
	if err != nil {
		logger.Warn("openrouter /models fetch failed; using cached capabilities", "err", err)
		if capStore != nil {
			cached, cErr := capStore.GetAllCapabilities(ctx, "openrouter")
			if cErr != nil {
				logger.Warn("failed to load cached capabilities", "err", cErr)
				return
			}
			caps = cached
		} else {
			return
		}
	} else if capStore != nil {
		for id, c := range caps {
			if putErr := capStore.PutCapabilities(ctx, "openrouter", id, c); putErr != nil {
				logger.Warn("cache put failed", "model", id, "err", putErr)
			}
		}
		logger.Info("openrouter capabilities cached", "count", len(caps))
	}

	// Apply caps to each OR-backed provider.
	for name, mc := range cfg.Models {
		if mc.Provider != "openrouter" {
			continue
		}
		p, ok := providers[name]
		if !ok {
			continue
		}
		cp, ok := p.(llm.ConfigurableProvider)
		if !ok {
			continue
		}
		cur := cp.CurrentModel()
		if c, found := caps[cur]; found {
			cp.SetModel(cur, c)
		}
	}
}

// hydrateGeminiCapabilities runs once at startup. Uses the first Gemini API key
// it finds to list the public model catalog, merges in hardcoded prices, and
// upserts each entry into the CapabilityStore under provider key "gemini".
// Also applies caps to each Gemini-backed provider so the router can see
// vision/tool flags from the first request.
func hydrateGeminiCapabilities(cfg *config.Config, providers map[string]llm.Provider, s store.Store, logger *slog.Logger) {
	var apiKey string
	for _, mc := range cfg.Models {
		if mc.Provider == "gemini" && mc.APIKey != "" {
			apiKey = mc.APIKey
			break
		}
	}
	if apiKey == "" {
		return
	}

	capStore, _ := s.(llm.CapabilityStore)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	caps, err := llm.FetchGeminiModels(ctx, apiKey)
	if err != nil {
		logger.Warn("gemini /models fetch failed; using cached capabilities", "err", err)
		if capStore != nil {
			cached, cErr := capStore.GetAllCapabilities(ctx, "gemini")
			if cErr != nil {
				logger.Warn("failed to load cached gemini capabilities", "err", cErr)
				return
			}
			caps = cached
		} else {
			return
		}
	} else if capStore != nil {
		for id, c := range caps {
			if putErr := capStore.PutCapabilities(ctx, "gemini", id, c); putErr != nil {
				logger.Warn("gemini cache put failed", "model", id, "err", putErr)
			}
		}
		logger.Info("gemini capabilities cached", "count", len(caps))
	}

	// Apply caps to each Gemini-backed provider at its currently configured model id.
	for name, mc := range cfg.Models {
		if mc.Provider != "gemini" {
			continue
		}
		p, ok := providers[name]
		if !ok {
			continue
		}
		cp, ok := p.(llm.ConfigurableProvider)
		if !ok {
			continue
		}
		cur := cp.CurrentModel()
		if c, found := caps[cur]; found {
			cp.SetModel(cur, c)
		}
	}
}

// hydrateArtificialAnalysisScores loads AA model data (from cache or live API)
// and merges Intelligence Index scores into the CapabilityStore.
// Silently skips when not configured.
func hydrateArtificialAnalysisScores(cfg *config.Config, s store.Store, logger *slog.Logger) {
	apiKey := cfg.ArtificialAnalysisAPIKey
	if apiKey == "" {
		return
	}
	capStore, ok := s.(llm.CapabilityStore)
	if !ok {
		return
	}
	settings, ok := s.(llm.SettingsStore)
	if !ok {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Try cache first; fetch from API if absent or expired.
	var models map[string]llm.AAModelInfo
	cache, err := llm.LoadAACache(ctx, settings)
	if err != nil {
		logger.Warn("aa cache load failed", "err", err)
	}
	if cache != nil {
		models = cache.Models
		logger.Info("aa cache loaded", "models", len(models), "age", time.Since(cache.FetchedAt).Round(time.Minute))
	} else {
		models, err = llm.FetchArtificialAnalysisData(ctx, apiKey)
		if err != nil {
			logger.Warn("artificialanalysis fetch failed", "err", err)
			return
		}
		if storeErr := llm.StoreAACache(ctx, settings, models); storeErr != nil {
			logger.Warn("aa cache store failed", "err", storeErr)
		} else {
			logger.Info("aa cache stored", "models", len(models))
		}
	}

	caps, err := capStore.GetAllCapabilities(ctx, "openrouter")
	if err != nil {
		logger.Warn("failed to load capabilities for AA merge", "err", err)
		return
	}
	llm.MergeAAScores(caps, models)
	updated := 0
	for id, c := range caps {
		if c.Score == 0 {
			continue
		}
		if putErr := capStore.PutCapabilities(ctx, "openrouter", id, c); putErr != nil {
			logger.Warn("AA score put failed", "model", id, "err", putErr)
		} else {
			updated++
		}
	}
	logger.Info("artificialanalysis scores merged", "models", len(models), "updated", updated)
}
