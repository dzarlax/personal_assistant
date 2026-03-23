// claude-bridge: HTTP service that wraps Claude Code CLI.
// Runs on the host, accepts requests from the PA bot in Docker,
// calls `claude -p` as a subprocess, returns the result.
//
// Configuration via environment variables:
//
//	CLAUDE_BRIDGE_LISTEN       — listen address (default 127.0.0.1:9900)
//	CLAUDE_BRIDGE_TOKEN        — bearer auth token (required)
//	CLAUDE_BRIDGE_PROJECT_DIR  — project directory for claude CLI (required)
//	CLAUDE_BRIDGE_CLI          — path to claude binary (default "claude")
//	CLAUDE_BRIDGE_TIMEOUT      — default timeout in seconds (default 120)
//	CLAUDE_BRIDGE_CONCURRENCY  — max parallel CLI calls (default 2)
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
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

// --- Request / Response ---

type AskRequest struct {
	Prompt     string `json:"prompt"`
	SessionID  string `json:"session_id,omitempty"`
	TimeoutSec int    `json:"timeout_sec,omitempty"`
}

type AskResponse struct {
	Result     string `json:"result"`
	Model      string `json:"model,omitempty"`
	DurationMs int64  `json:"duration_ms"`
	IsError    bool   `json:"is_error"`
	Error      string `json:"error,omitempty"`
}

// --- Bridge ---

type Bridge struct {
	projectDir     string
	cliPath        string
	authToken      string
	defaultTimeout int
	sem            chan struct{}
	logger         *slog.Logger
}

func (b *Bridge) callCLI(ctx context.Context, prompt string, timeoutSec int) AskResponse {
	start := time.Now()

	// Acquire semaphore with timeout
	select {
	case b.sem <- struct{}{}:
		defer func() { <-b.sem }()
	case <-time.After(10 * time.Second):
		return AskResponse{DurationMs: time.Since(start).Milliseconds(), IsError: true, Error: "queue_timeout"}
	case <-ctx.Done():
		return AskResponse{DurationMs: time.Since(start).Milliseconds(), IsError: true, Error: "cancelled"}
	}

	cliCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(cliCtx, b.cliPath, "-p", prompt, "--output-format", "json")
	cmd.Dir = b.projectDir

	b.logger.Info("calling CLI", "timeout", timeoutSec, "prompt_len", len(prompt))

	output, err := cmd.Output()
	duration := time.Since(start).Milliseconds()

	if err != nil {
		if cliCtx.Err() == context.DeadlineExceeded {
			return AskResponse{DurationMs: duration, IsError: true, Error: "timeout"}
		}
		errMsg := "cli_error"
		if exitErr, ok := err.(*exec.ExitError); ok {
			stderr := strings.TrimSpace(string(exitErr.Stderr))
			if stderr != "" {
				errMsg = stderr
			}
			if strings.Contains(stderr, "rate limit") || strings.Contains(stderr, "429") {
				return AskResponse{DurationMs: duration, IsError: true, Error: "rate_limited"}
			}
		}
		b.logger.Error("CLI failed", "err", err, "duration_ms", duration)
		return AskResponse{DurationMs: duration, IsError: true, Error: errMsg}
	}

	// Try JSON parse
	var parsed struct {
		Result string `json:"result"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(output, &parsed); err == nil && parsed.Result != "" {
		return AskResponse{Result: parsed.Result, Model: parsed.Model, DurationMs: duration}
	}

	// Fallback: raw text
	text := strings.TrimSpace(string(output))
	if text == "" {
		return AskResponse{DurationMs: duration, IsError: true, Error: "empty_response"}
	}
	return AskResponse{Result: text, DurationMs: duration}
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
		timeout = b.defaultTimeout
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

	b.logger.Info("response", "duration_ms", resp.DurationMs, "is_error", resp.IsError, "result_len", len(resp.Result))
	writeJSON(w, status, resp)
}

func (b *Bridge) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if exec.CommandContext(ctx, b.cliPath, "--version").Run() == nil {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	} else {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "cli_unavailable"})
	}
}

func (b *Bridge) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+b.authToken {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	token := env("CLAUDE_BRIDGE_TOKEN", "")
	projectDir := env("CLAUDE_BRIDGE_PROJECT_DIR", "")

	if token == "" || projectDir == "" {
		fmt.Fprintln(os.Stderr, "Required env: CLAUDE_BRIDGE_TOKEN, CLAUDE_BRIDGE_PROJECT_DIR")
		os.Exit(1)
	}

	// Expand ~
	if strings.HasPrefix(projectDir, "~/") {
		home, _ := os.UserHomeDir()
		projectDir = home + projectDir[1:]
	}

	if _, err := os.Stat(projectDir); err != nil {
		logger.Error("project dir not found", "path", projectDir)
		os.Exit(1)
	}

	listen := env("CLAUDE_BRIDGE_LISTEN", "127.0.0.1:9900")
	defaultTimeout := envInt("CLAUDE_BRIDGE_TIMEOUT", 120)

	bridge := &Bridge{
		projectDir:     projectDir,
		cliPath:        env("CLAUDE_BRIDGE_CLI", "claude"),
		authToken:      token,
		defaultTimeout: defaultTimeout,
		sem:            make(chan struct{}, envInt("CLAUDE_BRIDGE_CONCURRENCY", 2)),
		logger:         logger,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ask", bridge.auth(bridge.handleAsk))
	mux.HandleFunc("/health", bridge.handleHealth)

	srv := &http.Server{
		Addr:         listen,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: time.Duration(defaultTimeout+30) * time.Second,
	}

	var wg sync.WaitGroup
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-ctx.Done()
		logger.Info("shutting down")
		shutCtx, c := context.WithTimeout(context.Background(), 30*time.Second)
		defer c()
		srv.Shutdown(shutCtx)
	}()

	logger.Info("claude-bridge starting", "listen", listen, "project_dir", projectDir)

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		logger.Error("server error", "err", err)
		os.Exit(1)
	}
	wg.Wait()
}
