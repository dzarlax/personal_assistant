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

	sysPromptBytes, err := os.ReadFile("config/system_prompt.md")
	if err != nil {
		logger.Error("failed to load system prompt", "err", err)
		os.Exit(1)
	}

	// Init LLM providers
	primary, err := llm.NewDeepSeek(cfg.Models.Default)
	if err != nil {
		logger.Error("failed to init primary LLM", "err", err)
		os.Exit(1)
	}

	var fallback llm.Provider
	if cfg.Models.FlashLite.APIKey != "" {
		fallback, err = llm.NewGemini(cfg.Models.FlashLite)
		if err != nil {
			logger.Warn("failed to init fallback LLM", "err", err)
		}
	}

	var reasoner llm.Provider
	if cfg.Models.Reasoner.APIKey != "" {
		reasoner, err = llm.NewDeepSeek(cfg.Models.Reasoner)
		if err != nil {
			logger.Warn("failed to init reasoner LLM", "err", err)
		}
	}

	var multimodal llm.Provider
	if cfg.Models.Multimodal.APIKey != "" {
		multimodal, err = llm.NewGeminiMultimodal(cfg.Models.Multimodal)
		if err != nil {
			logger.Warn("failed to init multimodal LLM", "err", err)
		}
	}

	router := llm.NewRouter(primary, fallback, reasoner, multimodal, cfg.Routing.ClassifierMinLength)

	// Init store (SQLite if data dir exists, otherwise memory)
	var s store.Store
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

	// Init compacter (uses primary model for summarization)
	compacter := agent.NewCompacter(primary)

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

		if cfg.ToolFilter.TopK > 0 && cfg.Models.Embedding.APIKey != "" {
			mcpClient.EnableEmbeddings(cfg.Models.Embedding.APIKey, cfg.Models.Embedding.Model, cfg.ToolFilter.TopK)
			embedCtx, embedCancel := context.WithTimeout(context.Background(), 60*time.Second)
			mcpClient.EmbedTools(embedCtx)
			embedCancel()
		}
	}

	ag := agent.New(router, s, mcpClient, compacter, string(sysPromptBytes), logger)

	handler, err := telegram.NewHandler(cfg.Telegram, ag, logger)
	if err != nil {
		logger.Error("failed to init Telegram handler", "err", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	logger.Info("agent started", "model", router.Name(), "mcp_servers", len(mcpServers))
	handler.Start(ctx)
	logger.Info("agent stopped")
}
