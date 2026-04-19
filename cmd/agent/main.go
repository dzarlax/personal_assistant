package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
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
			if mc.BaseURL == "" {
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
	primary := providers[cfg.Routing.Default]

	// Default role keys fall through to routing.default when unspecified.
	multimodalKey := cfg.Routing.Multimodal
	if multimodalKey == "" {
		multimodalKey = cfg.Routing.Default
	}
	reasonerKey := cfg.Routing.Reasoner
	if reasonerKey == "" {
		reasonerKey = cfg.Routing.Default
	}
	localKey := cfg.Routing.Local
	if localKey == "" {
		localKey = cfg.Routing.Default
	}

	router := llm.NewRouter(providers, llm.RouterConfig{
		Local:            localKey,
		Primary:          cfg.Routing.Default,
		Fallback:         cfg.Routing.Fallback,
		Multimodal:       multimodalKey,
		Reasoner:         reasonerKey,
		Classifier:        cfg.Routing.Classifier,
		ClassifierMinLen:  cfg.Routing.ClassifierMinLength,
		ClassifierTimeout: cfg.Routing.ClassifierTimeout,
		ClassifierPrompt:  cfg.Routing.ClassifierPrompt,
	})

	// Warn about routing roles that reference missing providers
	for role, model := range map[string]string{
		"local":      localKey,
		"fallback":   cfg.Routing.Fallback,
		"multimodal": multimodalKey,
		"reasoner":   reasonerKey,
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

	// Persist routing overrides across restarts via the DB settings store.
	// The legacy file path is kept only for one-time migration on first start.
	if ss, ok := s.(llm.SettingsStore); ok {
		router.SetSettingsStore(ss)
	}
	router.SetPersistPath("config/routing.json")
	if err := router.LoadPersistedOverrides(); err != nil {
		logger.Warn("failed to load routing overrides", "err", err)
	}
	// Apply persisted per-slot OpenRouter model overrides (e.g. admin UI picked
	// a different model last session). Pull caps from the store.
	if orOverrides := router.TakePendingOpenRouterOverrides(); len(orOverrides) > 0 {
		applyOpenRouterOverrides(router, s, orOverrides, logger)
	}

	// Init compacter — use the effective primary after overrides are loaded.
	// Fallback handles content-filter rejections (e.g. DashScope DataInspectionFailed).
	effectivePrimaryKey := router.GetConfig().Primary
	effectivePrimary := providers[effectivePrimaryKey]
	if effectivePrimary == nil {
		effectivePrimary = primary
	}
	fallbackKey := router.GetConfig().Fallback
	compactFallback := providers[fallbackKey]
	compacter := agent.NewCompacter(effectivePrimary, compactFallback)

	// Init MCP client
	var mcpClient *mcp.Client
	mcpServers, err := config.LoadMCPServers("config/mcp.json")
	if err != nil {
		logger.Warn("failed to load mcp.json", "err", err)
	}
	if len(mcpServers) > 0 {
		mcpClient = mcp.NewClient(mcpServers, logger)
		initCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		mcpClient.Initialize(initCtx)
		cancel()

		if emb, ok := cfg.Models["embedding"]; ok && cfg.ToolFilter.TopK > 0 && (emb.APIKey != "" || emb.BaseURL != "") {
			mcpClient.EnableEmbeddings(emb, cfg.ToolFilter.TopK)
			embedCtx, embedCancel := context.WithTimeout(context.Background(), 60*time.Second)
			mcpClient.EmbedTools(embedCtx)
			embedCancel()
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

	if cfg.WebSearch.Enabled {
		provider := cfg.WebSearch.Provider
		if provider == "" {
			provider = "ollama"
		}
		ag.EnableWebSearch(agent.WebSearchConfig{
			Provider: provider,
			BaseURL:  cfg.WebSearch.BaseURL,
			APIKey:   cfg.WebSearch.APIKey,
		})
		logger.Info("web search enabled", "provider", provider)
	}

	if cfg.WebFetch.Enabled {
		ag.EnableWebFetch(agent.WebFetchConfig{CDPURL: cfg.WebFetch.CDPURL})
		logger.Info("web fetch enabled", "cdp_fallback", cfg.WebFetch.CDPURL != "")
	}

	if cfg.Filesystem.Enabled {
		root := cfg.Filesystem.Root
		if root == "" {
			root = "/assistant_context"
		}
		ag.EnableFilesystem(agent.FilesystemConfig{Root: root})
		logger.Info("filesystem tools enabled", "root", root)
	}

	if cfg.TTS.Enabled {
		voice := cfg.TTS.Voice
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

	if cfg.VoiceAPI.Enabled {
		voiceSrv := voiceapi.New(ag, cfg.VoiceAPI, logger)
		go func() {
			if err := voiceSrv.ListenAndServe(); err != nil {
				logger.Error("voice API error", "err", err)
			}
		}()
		defer voiceSrv.Shutdown(context.Background())
	}

	if cfg.AdminAPI.Enabled {
		capStore, _ := s.(llm.CapabilityStore)
		adminSrv := adminapi.New(cfg.AdminAPI, router, capStore, cfg, logger)
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

// applyOpenRouterOverrides applies persisted per-slot model overrides to
// OR-backed providers. Capabilities come from the CapabilityStore; slots with
// unknown caps are still applied with zero-value caps (vision-aware routing
// will then treat them as text-only, which is the safer default).
func applyOpenRouterOverrides(router *llm.Router, s store.Store, overrides map[string]string, logger *slog.Logger) {
	capStore, _ := s.(llm.CapabilityStore)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for slot, modelID := range overrides {
		var caps llm.Capabilities
		if capStore != nil {
			if c, ok, err := capStore.GetCapabilities(ctx, "openrouter", modelID); err == nil && ok {
				caps = c
			}
		}
		if err := router.SetProviderModel(slot, modelID, caps); err != nil {
			logger.Warn("failed to apply OR override", "slot", slot, "model", modelID, "err", err)
			continue
		}
		logger.Info("applied OR model override", "slot", slot, "model", modelID,
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
