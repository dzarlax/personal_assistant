// claude-bridge: HTTP service that wraps Claude Code CLI.
// Runs on the host, accepts requests from the PA bot in Docker,
// calls `claude -p` as a subprocess, returns the result.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

// --- Config ---

type Config struct {
	Listen        string `yaml:"listen"`
	ProjectDir    string `yaml:"project_dir"`
	CLIPath       string `yaml:"cli_path"`
	DefaultTimeout int   `yaml:"default_timeout"`
	MaxConcurrent int    `yaml:"max_concurrent"`
	AuthToken     string `yaml:"auth_token"`
}

func loadConfig(path string) (Config, error) {
	cfg := Config{
		Listen:        "127.0.0.1:9900",
		CLIPath:       "claude",
		DefaultTimeout: 120,
		MaxConcurrent: 2,
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	expanded := os.ExpandEnv(string(data))
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return cfg, fmt.Errorf("parse config: %w", err)
	}
	if cfg.ProjectDir == "" {
		return cfg, fmt.Errorf("project_dir is required")
	}
	if cfg.AuthToken == "" {
		return cfg, fmt.Errorf("auth_token is required")
	}
	// Expand ~ in project_dir
	if strings.HasPrefix(cfg.ProjectDir, "~/") {
		home, _ := os.UserHomeDir()
		cfg.ProjectDir = home + cfg.ProjectDir[1:]
	}
	return cfg, nil
}

// --- Request / Response ---

type AskRequest struct {
	Prompt     string `json:"prompt"`
	SessionID  string `json:"session_id,omitempty"`
	MaxTokens  int    `json:"max_tokens,omitempty"`
	TimeoutSec int    `json:"timeout_sec,omitempty"`
}

type AskResponse struct {
	Result     string `json:"result"`
	Model      string `json:"model,omitempty"`
	DurationMs int64  `json:"duration_ms"`
	IsError    bool   `json:"is_error"`
	Error      string `json:"error,omitempty"`
}

// --- CLI output parsing ---

type CLIResult struct {
	Result   string `json:"result"`
	Model    string `json:"model"`
	Duration float64 `json:"duration_ms"`
	// Claude CLI --output-format json fields
	CostUSD float64 `json:"cost_usd"`
}

// --- Bridge ---

type Bridge struct {
	cfg    Config
	sem    chan struct{}
	logger *slog.Logger
}

func NewBridge(cfg Config, logger *slog.Logger) *Bridge {
	return &Bridge{
		cfg:    cfg,
		sem:    make(chan struct{}, cfg.MaxConcurrent),
		logger: logger,
	}
}

func (b *Bridge) callCLI(ctx context.Context, prompt string, timeoutSec int) AskResponse {
	start := time.Now()

	// Acquire semaphore with timeout
	select {
	case b.sem <- struct{}{}:
		defer func() { <-b.sem }()
	case <-time.After(10 * time.Second):
		return AskResponse{
			DurationMs: time.Since(start).Milliseconds(),
			IsError:    true,
			Error:      "queue_timeout",
		}
	case <-ctx.Done():
		return AskResponse{
			DurationMs: time.Since(start).Milliseconds(),
			IsError:    true,
			Error:      "cancelled",
		}
	}

	timeout := time.Duration(timeoutSec) * time.Second
	cliCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := []string{
		"-p", prompt,
		"--output-format", "json",
	}

	cmd := exec.CommandContext(cliCtx, b.cfg.CLIPath, args...)
	cmd.Dir = b.cfg.ProjectDir // Claude Code picks up CLAUDE.md and .mcp.json from cwd

	b.logger.Info("calling CLI", "timeout", timeoutSec, "prompt_len", len(prompt))

	output, err := cmd.Output()
	duration := time.Since(start).Milliseconds()

	if err != nil {
		if cliCtx.Err() == context.DeadlineExceeded {
			return AskResponse{DurationMs: duration, IsError: true, Error: "timeout"}
		}
		// Try to get stderr for error details
		errMsg := "cli_error"
		if exitErr, ok := err.(*exec.ExitError); ok {
			stderr := strings.TrimSpace(string(exitErr.Stderr))
			if stderr != "" {
				errMsg = stderr
			}
			// Rate limit detection
			if strings.Contains(stderr, "rate limit") || strings.Contains(stderr, "429") {
				return AskResponse{DurationMs: duration, IsError: true, Error: "rate_limited"}
			}
		}
		b.logger.Error("CLI failed", "err", err, "duration_ms", duration)
		return AskResponse{DurationMs: duration, IsError: true, Error: errMsg}
	}

	// Try JSON parse first
	var cliResult CLIResult
	if err := json.Unmarshal(output, &cliResult); err == nil && cliResult.Result != "" {
		return AskResponse{
			Result:     cliResult.Result,
			Model:      cliResult.Model,
			DurationMs: duration,
		}
	}

	// Fallback: raw text output
	text := strings.TrimSpace(string(output))
	if text == "" {
		return AskResponse{DurationMs: duration, IsError: true, Error: "empty_response"}
	}
	return AskResponse{Result: text, DurationMs: duration}
}

func (b *Bridge) checkCLI() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, b.cfg.CLIPath, "--version")
	return cmd.Run() == nil
}

// --- HTTP Handlers ---

func (b *Bridge) authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		expected := "Bearer " + b.cfg.AuthToken
		if token != expected {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func (b *Bridge) handleAsk(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AskRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, AskResponse{IsError: true, Error: "invalid_json"})
		return
	}
	if req.Prompt == "" {
		writeJSON(w, http.StatusBadRequest, AskResponse{IsError: true, Error: "prompt_required"})
		return
	}

	timeout := req.TimeoutSec
	if timeout <= 0 {
		timeout = b.cfg.DefaultTimeout
	}

	resp := b.callCLI(r.Context(), req.Prompt, timeout)

	status := http.StatusOK
	if resp.IsError {
		switch resp.Error {
		case "timeout":
			status = http.StatusRequestTimeout
		case "rate_limited", "queue_timeout":
			status = http.StatusTooManyRequests
		default:
			status = http.StatusInternalServerError
		}
	}

	b.logger.Info("response",
		"duration_ms", resp.DurationMs,
		"is_error", resp.IsError,
		"error", resp.Error,
		"result_len", len(resp.Result),
	)

	writeJSON(w, status, resp)
}

func (b *Bridge) handleHealth(w http.ResponseWriter, r *http.Request) {
	if b.checkCLI() {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	} else {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "cli_unavailable"})
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// --- Main ---

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	cfgPath := "bridge.yaml"
	if len(os.Args) > 1 {
		cfgPath = os.Args[1]
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		logger.Error("failed to load config", "path", cfgPath, "err", err)
		os.Exit(1)
	}

	if !fileExists(cfg.ProjectDir) {
		logger.Error("project_dir does not exist", "path", cfg.ProjectDir)
		os.Exit(1)
	}

	bridge := NewBridge(cfg, logger)

	if !bridge.checkCLI() {
		logger.Warn("Claude CLI not found or not responding", "path", cfg.CLIPath)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ask", bridge.authMiddleware(bridge.handleAsk))
	mux.HandleFunc("/health", bridge.handleHealth)

	srv := &http.Server{
		Addr:         cfg.Listen,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: time.Duration(cfg.DefaultTimeout+30) * time.Second,
	}

	// Graceful shutdown
	var wg sync.WaitGroup
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-ctx.Done()
		logger.Info("shutting down")
		shutCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		srv.Shutdown(shutCtx)
	}()

	logger.Info("claude-bridge starting",
		"listen", cfg.Listen,
		"project_dir", cfg.ProjectDir,
		"max_concurrent", cfg.MaxConcurrent,
	)

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		logger.Error("server error", "err", err)
		os.Exit(1)
	}
	wg.Wait()
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
