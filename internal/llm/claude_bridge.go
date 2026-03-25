package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"telegram-agent/internal/config"
)

// claudeBridgeProvider sends prompts to claude-bridge HTTP service
// which wraps Claude Code CLI (`claude -p`).
type claudeBridgeProvider struct {
	baseURL   string
	authToken string
	timeout   time.Duration
	client    *http.Client

	mu        sync.Mutex
	sessionID string // set after first call; enables --resume for session continuity
}

func NewClaudeBridge(cfg config.ModelConfig) (*claudeBridgeProvider, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("claude-bridge: base_url is required")
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("claude-bridge: api_key (auth token) is required")
	}
	timeout := 120
	if cfg.MaxTokens > 0 {
		// Reuse max_tokens field as timeout hint (seconds).
		// Bridge has its own default; this overrides per-request.
		timeout = cfg.MaxTokens
	}
	return &claudeBridgeProvider{
		baseURL:   strings.TrimRight(cfg.BaseURL, "/"),
		authToken: cfg.APIKey,
		timeout:   time.Duration(timeout) * time.Second,
		client:    &http.Client{Timeout: time.Duration(timeout+30) * time.Second},
	}, nil
}

func (p *claudeBridgeProvider) Name() string { return "claude-bridge" }

// ResetSession clears the stored session ID so the next call starts a fresh Claude session.
func (p *claudeBridgeProvider) ResetSession() {
	p.mu.Lock()
	p.sessionID = ""
	p.mu.Unlock()
}

func (p *claudeBridgeProvider) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	p.mu.Lock()
	sid := p.sessionID
	p.mu.Unlock()

	var prompt string
	if sid != "" {
		// Continuing a session — only send the latest user message.
		// Claude CLI resumes the session and already has prior context.
		for i := len(messages) - 1; i >= 0; i-- {
			if messages[i].Role == "user" {
				prompt = messageText(messages[i])
				break
			}
		}
	} else {
		// First call — send full history so Claude has context.
		// Cap at 30K chars to avoid bloating the CLI prompt.
		prompt = buildPrompt(messages, systemPrompt)
		if len(prompt) > 30000 {
			prompt = prompt[len(prompt)-30000:]
		}
	}

	body := map[string]any{
		"prompt":      prompt,
		"timeout_sec": int(p.timeout.Seconds()),
	}
	if sid != "" {
		body["session_id"] = sid
	}
	reqBody, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/ask", bytes.NewReader(reqBody))
	if err != nil {
		return Response{}, fmt.Errorf("claude-bridge: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.authToken)

	resp, err := p.client.Do(req)
	if err != nil {
		return Response{}, &APIError{StatusCode: 503, Message: fmt.Sprintf("bridge unavailable: %v", err)}
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return Response{}, fmt.Errorf("claude-bridge: read response: %w", err)
	}

	var bridgeResp struct {
		Result    string `json:"result"`
		SessionID string `json:"session_id"`
		IsError   bool   `json:"is_error"`
		Error     string `json:"error"`
	}
	if err := json.Unmarshal(respBody, &bridgeResp); err != nil {
		return Response{}, fmt.Errorf("claude-bridge: parse response: %w", err)
	}

	if bridgeResp.IsError {
		// If resume failed, reset session so next call starts fresh.
		p.mu.Lock()
		p.sessionID = ""
		p.mu.Unlock()
		statusCode := resp.StatusCode
		if statusCode == 200 {
			statusCode = 500
		}
		return Response{}, &APIError{StatusCode: statusCode, Message: bridgeResp.Error}
	}

	// Store session ID for subsequent calls (enables --resume).
	if bridgeResp.SessionID != "" {
		p.mu.Lock()
		p.sessionID = bridgeResp.SessionID
		p.mu.Unlock()
	}

	return Response{Content: bridgeResp.Result}, nil
}

// buildPrompt formats conversation history as a single text prompt
// for claude -p which doesn't support multi-turn messages.
func buildPrompt(messages []Message, systemPrompt string) string {
	var b strings.Builder

	if systemPrompt != "" {
		b.WriteString(systemPrompt)
		b.WriteString("\n\n")
	}

	// Format recent messages as conversation
	for _, m := range messages {
		switch m.Role {
		case "user":
			b.WriteString("User: ")
			b.WriteString(messageText(m))
			b.WriteString("\n")
		case "assistant":
			if m.Content != "" {
				b.WriteString("Assistant: ")
				b.WriteString(m.Content)
				b.WriteString("\n")
			}
		}
		// Skip tool messages — Claude CLI handles tools itself
	}

	return b.String()
}

// messageText extracts text content from a message, handling multimodal parts.
func messageText(m Message) string {
	if len(m.Parts) == 0 {
		return m.Content
	}
	var texts []string
	for _, p := range m.Parts {
		switch p.Type {
		case "text":
			texts = append(texts, p.Text)
		case "image_url":
			texts = append(texts, "[image]")
		case "input_audio":
			texts = append(texts, "[audio]")
		case "inline_data":
			texts = append(texts, "[document]")
		}
	}
	return strings.Join(texts, " ")
}
