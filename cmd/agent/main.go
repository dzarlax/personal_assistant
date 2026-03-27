package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
	"telegram-agent/internal/mcp"
	"telegram-agent/internal/store"
	"telegram-agent/internal/telegram"
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
	if cfg.Models.DeepSeek.APIKey != "" {
		p, e := llm.NewDeepSeek(cfg.Models.DeepSeek)
		addProvider("deepseek", p, e)
	}
	if cfg.Models.DeepSeekR1.APIKey != "" {
		p, e := llm.NewDeepSeek(cfg.Models.DeepSeekR1)
		addProvider("deepseek-r1", p, e)
	}
	if cfg.Models.GeminiFlashLite.APIKey != "" {
		p, e := llm.NewGeminiNative(cfg.Models.GeminiFlashLite)
		addProvider("gemini-flash-lite", p, e)
	}
	if cfg.Models.GeminiFlash.APIKey != "" {
		p, e := llm.NewGeminiNative(cfg.Models.GeminiFlash)
		addProvider("gemini-flash", p, e)
	}
	if cfg.Models.QwenFlash.APIKey != "" {
		p, e := llm.NewQwen(cfg.Models.QwenFlash)
		addProvider("qwen-flash", p, e)
	}
	if cfg.Models.QwenPlus.APIKey != "" {
		p, e := llm.NewQwen(cfg.Models.QwenPlus)
		addProvider("qwen3.5-plus", p, e)
	}
	if cfg.Models.QwenMax.APIKey != "" {
		p, e := llm.NewQwen(cfg.Models.QwenMax)
		addProvider("qwen-max", p, e)
	}
	if cfg.Models.Ollama.Model != "" {
		p, e := llm.NewOllama(cfg.Models.Ollama)
		addProvider("ollama", p, e)
	}
	if cfg.Models.Claude.BaseURL != "" {
		p, e := llm.NewClaudeBridge(cfg.Models.Claude)
		addProvider("claude", p, e)
	}

	// Ensure the default routing provider is available.
	if providers[cfg.Routing.Default] == nil {
		logger.Error("default routing provider not configured or failed to init", "provider", cfg.Routing.Default)
		os.Exit(1)
	}
	primary := providers[cfg.Routing.Default]

	// Default role keys if not specified
	multimodalKey := cfg.Routing.Multimodal
	if multimodalKey == "" {
		multimodalKey = "gemini-flash"
	}
	reasonerKey := cfg.Routing.Reasoner
	if reasonerKey == "" {
		reasonerKey = "deepseek-r1"
	}

	router := llm.NewRouter(providers, llm.RouterConfig{
		Primary:          cfg.Routing.Default,
		Fallback:         cfg.Routing.Fallback,
		Multimodal:       multimodalKey,
		Reasoner:         reasonerKey,
		Classifier:       cfg.Routing.Classifier,
		ClassifierMinLen: cfg.Routing.ClassifierMinLength,
	})

	// Warn about routing roles that reference missing providers
	for role, model := range map[string]string{
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

	// Persist routing overrides across restarts
	router.SetPersistPath("config/routing.json")
	if err := router.LoadPersistedOverrides(); err != nil {
		logger.Warn("failed to load routing overrides", "err", err)
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

		if cfg.ToolFilter.TopK > 0 && (cfg.Models.Embedding.APIKey != "" || cfg.Models.Embedding.BaseURL != "") {
			mcpClient.EnableEmbeddings(cfg.Models.Embedding, cfg.ToolFilter.TopK)
			embedCtx, embedCancel := context.WithTimeout(context.Background(), 60*time.Second)
			mcpClient.EmbedTools(embedCtx)
			embedCancel()
		}
	}

	ag := agent.New(router, s, mcpClient, compacter, string(sysPromptBytes), logger)

	// Enable voice transcription if a multimodal Gemini model is configured.
	if cfg.Models.GeminiFlash.APIKey != "" && cfg.Models.GeminiFlash.Model != "" {
		ag.EnableTranscription(agent.TranscribeConfig{
			Model:  cfg.Models.GeminiFlash.Model,
			APIKey: cfg.Models.GeminiFlash.APIKey,
		})
		logger.Info("voice transcription enabled", "model", cfg.Models.GeminiFlash.Model)
	}

	if cfg.WebSearch.Enabled {
		baseURL := cfg.WebSearch.BaseURL
		if baseURL == "" {
			baseURL = "https://ollama.com"
		}
		ag.EnableWebSearch(agent.WebSearchConfig{
			BaseURL: baseURL,
			APIKey:  cfg.WebSearch.APIKey,
		})
		logger.Info("web search enabled", "base_url", baseURL)
	}

	if cfg.Filesystem.Enabled {
		root := cfg.Filesystem.Root
		if root == "" {
			root = "/assistant_context"
		}
		ag.EnableFilesystem(agent.FilesystemConfig{Root: root})
		logger.Info("filesystem tools enabled", "root", root)
	}

	handler, err := telegram.NewHandler(cfg.Telegram, ag, logger)
	if err != nil {
		logger.Error("failed to init Telegram handler", "err", err)
		os.Exit(1)
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
