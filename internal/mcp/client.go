package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

const (
	maxResponseSize  = 10 * 1024 * 1024 // 10 MB — prevent OOM from malicious server
	maxToolResultLen = 100 * 1024        // 100 KB — cap what enters conversation history
	maxDescLen       = 4 * 1024          // 4 KB — tool description sent to LLM
	maxToolNameLen   = 128
	maxArgsSize      = 1 * 1024 * 1024 // 1 MB
)

var validToolName = regexp.MustCompile(`^[a-zA-Z0-9_\-/:. ]+$`)

// Client manages connections to multiple MCP servers.
type Client struct {
	servers         map[string]*server
	tools          []Tool
	toolServers    map[string]string // tool name → server name
	logger         *slog.Logger
	embeddingCfg   config.ModelConfig
	topK           int
	embeddingsReady bool
}

type server struct {
	name       string
	url        string
	headers    map[string]string
	allowTools map[string]bool // nil = all allowed
	denyTools  map[string]bool
	sessionID  string
	http       *http.Client
	counter    atomic.Int32
}

// Tool is an MCP tool with its origin server.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	ServerName  string
	Embedding   []float32
}

func NewClient(configs map[string]config.MCPServerConfig, logger *slog.Logger) *Client {
	c := &Client{
		servers:     make(map[string]*server),
		toolServers: make(map[string]string),
		logger:      logger,
	}
	for name, cfg := range configs {
		if err := validateServerURL(cfg.URL); err != nil {
			logger.Warn("mcp server URL rejected", "server", name, "url", cfg.URL, "err", err)
			continue
		}
		srv := &server{
			name:    name,
			url:     cfg.URL,
			headers: cfg.Headers,
			http:    &http.Client{Timeout: 30 * time.Second},
		}
		if len(cfg.AllowTools) > 0 {
			srv.allowTools = make(map[string]bool, len(cfg.AllowTools))
			for _, t := range cfg.AllowTools {
				srv.allowTools[t] = true
			}
		}
		if len(cfg.DenyTools) > 0 {
			srv.denyTools = make(map[string]bool, len(cfg.DenyTools))
			for _, t := range cfg.DenyTools {
				srv.denyTools[t] = true
			}
		}
		c.servers[name] = srv
	}
	return c
}

// Initialize connects to all servers, discovers tools. Unavailable servers are skipped.
func (c *Client) Initialize(ctx context.Context) {
	for name, srv := range c.servers {
		if err := srv.initialize(ctx); err != nil {
			c.logger.Warn("mcp server unavailable", "server", name, "err", err)
			continue
		}

		tools, err := srv.listTools(ctx)
		if err != nil {
			c.logger.Warn("mcp tools/list failed", "server", name, "err", err)
			continue
		}

		allowed := 0
		for _, t := range tools {
			if !srv.isToolAllowed(t.Name) {
				continue
			}
			t.ServerName = name
			// Always prefix tool names with server name to avoid conflicts.
			t.Name = name + "__" + t.Name
			c.tools = append(c.tools, t)
			c.toolServers[t.Name] = name
			allowed++
		}
		c.logger.Info("mcp server connected", "server", name, "tools", allowed, "total", len(tools))
	}
}

// EnableEmbeddings configures vector-based tool filtering.
// Must be called before EmbedTools.
func (c *Client) EnableEmbeddings(cfg config.ModelConfig, topK int) {
	c.embeddingCfg = cfg
	c.topK = topK
}

// EmbedText computes an embedding for the given text using the configured embedding model.
// Returns an error if embeddings are not configured.
func (c *Client) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if c.embeddingCfg.APIKey == "" && c.embeddingCfg.BaseURL == "" {
		return nil, fmt.Errorf("embeddings not configured")
	}
	return embed(ctx, c.embeddingCfg, text)
}

// EmbedTools computes and caches embeddings for all tools.
// Should be called once after Initialize.
func (c *Client) EmbedTools(ctx context.Context) {
	if c.embeddingCfg.APIKey == "" && c.embeddingCfg.BaseURL == "" {
		return
	}
	ok := 0
	for i := range c.tools {
		text := c.tools[i].Name + ": " + c.tools[i].Description
		emb, err := embed(ctx, c.embeddingCfg, text)
		if err != nil {
			c.logger.Warn("failed to embed tool", "tool", c.tools[i].Name, "err", err)
			continue
		}
		c.tools[i].Embedding = emb
		ok++
	}
	if ok == len(c.tools) {
		c.embeddingsReady = true
		c.logger.Info("tool embeddings ready", "tools", ok)
	} else {
		c.logger.Warn("some tool embeddings failed, disabling filtering", "ok", ok, "total", len(c.tools))
	}
}

// LLMToolsForQuery returns the top-K most relevant tools for the given query.
// Falls back to all tools if embeddings are not ready or topK >= total tools.
func (c *Client) LLMToolsForQuery(ctx context.Context, query string) []llm.Tool {
	if !c.embeddingsReady || c.topK <= 0 || c.topK >= len(c.tools) || query == "" {
		return c.LLMTools()
	}

	queryEmb, err := embed(ctx, c.embeddingCfg, query)
	if err != nil {
		c.logger.Warn("query embedding failed, using all tools", "err", err)
		return c.LLMTools()
	}

	type scored struct {
		tool  Tool
		score float64
	}
	candidates := make([]scored, len(c.tools))
	for i, t := range c.tools {
		candidates[i] = scored{t, cosineSimilarity(queryEmb, t.Embedding)}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	result := make([]llm.Tool, 0, c.topK)
	for i := 0; i < c.topK; i++ {
		t := candidates[i].tool
		result = append(result, llm.Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	c.logger.Debug("tool filter applied", "query_len", len(query), "selected", len(result), "total", len(c.tools))
	return result
}

// LLMTools returns tools in the format expected by the LLM provider.
func (c *Client) LLMTools() []llm.Tool {
	result := make([]llm.Tool, 0, len(c.tools))
	for _, t := range c.tools {
		result = append(result, llm.Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	return result
}

// Tools returns the raw tool list (name + server) for display purposes.
func (c *Client) Tools() []Tool {
	return c.tools
}

// Close releases resources held by the MCP client (HTTP connection pools).
func (c *Client) Close() {
	for _, srv := range c.servers {
		srv.http.CloseIdleConnections()
	}
}

// Reconnect closes existing connections, replaces the server list with new configs,
// re-initializes all servers, and re-embeds tools if embeddings were enabled.
// Returns the number of tools discovered after reconnect.
func (c *Client) Reconnect(ctx context.Context, configs map[string]config.MCPServerConfig) (int, error) {
	// Close old connections.
	c.Close()

	// Reset state.
	c.servers = make(map[string]*server)
	c.tools = nil
	c.toolServers = make(map[string]string)
	c.embeddingsReady = false

	// Build new servers (same logic as NewClient).
	for name, cfg := range configs {
		if err := validateServerURL(cfg.URL); err != nil {
			c.logger.Warn("mcp server URL rejected", "server", name, "url", cfg.URL, "err", err)
			continue
		}
		srv := &server{
			name:    name,
			url:     cfg.URL,
			headers: cfg.Headers,
			http:    &http.Client{Timeout: 30 * time.Second},
		}
		if len(cfg.AllowTools) > 0 {
			srv.allowTools = make(map[string]bool, len(cfg.AllowTools))
			for _, t := range cfg.AllowTools {
				srv.allowTools[t] = true
			}
		}
		if len(cfg.DenyTools) > 0 {
			srv.denyTools = make(map[string]bool, len(cfg.DenyTools))
			for _, t := range cfg.DenyTools {
				srv.denyTools[t] = true
			}
		}
		c.servers[name] = srv
	}

	// Initialize and discover tools.
	c.Initialize(ctx)

	// Re-embed tools if embeddings were configured.
	if c.embeddingCfg.APIKey != "" || c.embeddingCfg.BaseURL != "" {
		c.EmbedTools(ctx)
	}

	return len(c.tools), nil
}

// CallTool executes a tool on the appropriate MCP server.
// On network failure it attempts one reconnect before giving up.
func (c *Client) CallTool(ctx context.Context, name string, argsJSON string) (string, error) {
	serverName, ok := c.toolServers[name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	srv := c.servers[serverName]
	// Strip server prefix: "server__tool_name" → "tool_name"
	originalName := name
	if idx := strings.Index(name, "__"); idx >= 0 {
		originalName = name[idx+2:]
	}
	result, err := srv.callTool(ctx, originalName, json.RawMessage(argsJSON))
	if err == nil {
		return result, nil
	}
	// On network/HTTP error try to reconnect once
	if !isRPCError(err) {
		c.logger.Warn("mcp tool call failed, reconnecting", "server", serverName, "tool", name, "err", err)
		initCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		if reinitErr := srv.initialize(initCtx); reinitErr != nil {
			c.logger.Warn("mcp reconnect failed", "server", serverName, "err", reinitErr)
			return "", err // return original error
		}
		result, err = srv.callTool(ctx, originalName, json.RawMessage(argsJSON))
	}
	return result, err
}

// isRPCError returns true for JSON-RPC level errors (server is up but returned an error).
func isRPCError(err error) bool {
	return err != nil && len(err.Error()) > 9 && err.Error()[:9] == "rpc error"
}

func (s *server) isToolAllowed(name string) bool {
	if s.denyTools[name] {
		return false
	}
	if s.allowTools != nil {
		return s.allowTools[name]
	}
	return true
}

// --- server methods ---

func (s *server) initialize(ctx context.Context) error {
	id := int(s.counter.Add(1))
	params := map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "telegram-agent", "version": "1.0"},
	}

	result, err := s.post(ctx, id, "initialize", params)
	if err != nil {
		return fmt.Errorf("initialize: %w", err)
	}
	_ = result // server capabilities, not needed for MVP

	// Send initialized notification (no response expected)
	_ = s.notify(ctx, "notifications/initialized", nil)
	return nil
}

func (s *server) listTools(ctx context.Context) ([]Tool, error) {
	id := int(s.counter.Add(1))
	result, err := s.post(ctx, id, "tools/list", map[string]any{})
	if err != nil {
		return nil, fmt.Errorf("tools/list: %w", err)
	}

	var resp struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"inputSchema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		return nil, fmt.Errorf("parse tools: %w", err)
	}

	tools := make([]Tool, 0, len(resp.Tools))
	for _, t := range resp.Tools {
		if len(t.Name) == 0 || len(t.Name) > maxToolNameLen || !validToolName.MatchString(t.Name) {
			continue // skip tools with invalid names
		}
		desc := t.Description
		if len(desc) > maxDescLen {
			desc = desc[:maxDescLen]
		}
		tools = append(tools, Tool{
			Name:        t.Name,
			Description: desc,
			InputSchema: t.InputSchema,
		})
	}
	return tools, nil
}

func (s *server) callTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	if len(args) > maxArgsSize {
		return "", fmt.Errorf("tool args too large: %d bytes (max %d)", len(args), maxArgsSize)
	}
	id := int(s.counter.Add(1))
	params := map[string]any{
		"name":      name,
		"arguments": args,
	}

	result, err := s.post(ctx, id, "tools/call", params)
	if err != nil {
		return "", fmt.Errorf("tools/call %s: %w", name, err)
	}

	var resp struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(result, &resp); err != nil {
		// If we can't parse, return raw result
		return string(result), nil
	}

	var parts []string
	for _, c := range resp.Content {
		if c.Type == "text" && c.Text != "" {
			parts = append(parts, c.Text)
		}
	}
	text := strings.Join(parts, "\n")
	if resp.IsError {
		return "", fmt.Errorf("tool error: %s", text)
	}
	if len(text) > maxToolResultLen {
		slog.Warn("tool result truncated", "tool", name, "original_len", len(text), "max", maxToolResultLen)
		text = text[:maxToolResultLen] + "\n...[truncated]"
	}
	return text, nil
}

// post sends a JSON-RPC request and returns the result.
func (s *server) post(ctx context.Context, id int, method string, params any) (json.RawMessage, error) {
	reqBody := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
		"params":  params,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	if s.sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", s.sessionID)
	}
	s.addAuth(httpReq)

	httpResp, err := s.http.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	if sid := httpResp.Header.Get("Mcp-Session-Id"); sid != "" {
		s.sessionID = sid
	}

	if httpResp.StatusCode >= 400 {
		return nil, fmt.Errorf("http %d", httpResp.StatusCode)
	}

	ct := httpResp.Header.Get("Content-Type")
	var raw json.RawMessage
	if strings.Contains(ct, "text/event-stream") {
		raw, err = readSSEResult(httpResp.Body)
	} else {
		raw, err = readJSONResult(httpResp.Body)
	}
	if err != nil {
		return nil, err
	}
	return raw, nil
}

// notify sends a JSON-RPC notification (no response expected).
func (s *server) notify(ctx context.Context, method string, params any) error {
	reqBody := map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
	}
	if params != nil {
		reqBody["params"] = params
	}

	body, _ := json.Marshal(reqBody)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if s.sessionID != "" {
		httpReq.Header.Set("Mcp-Session-Id", s.sessionID)
	}
	s.addAuth(httpReq)

	httpResp, err := s.http.Do(httpReq)
	if err != nil {
		return err
	}
	httpResp.Body.Close()
	return nil
}

func (s *server) addAuth(req *http.Request) {
	for k, v := range s.headers {
		req.Header.Set(k, v)
	}
}

// readJSONResult reads a JSON-RPC response and returns the result field.
func readJSONResult(r io.Reader) (json.RawMessage, error) {
	r = io.LimitReader(r, maxResponseSize)
	var envelope struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(r).Decode(&envelope); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if envelope.Error != nil {
		return nil, fmt.Errorf("rpc error %d: %s", envelope.Error.Code, envelope.Error.Message)
	}
	return envelope.Result, nil
}

// readSSEResult reads an SSE stream and extracts the first data event containing a JSON-RPC result.
func readSSEResult(r io.Reader) (json.RawMessage, error) {
	scanner := bufio.NewScanner(io.LimitReader(r, maxResponseSize))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		var envelope struct {
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &envelope); err != nil {
			continue
		}
		if envelope.Error != nil {
			return nil, fmt.Errorf("rpc error %d: %s", envelope.Error.Code, envelope.Error.Message)
		}
		if envelope.Result != nil {
			return envelope.Result, nil
		}
	}
	return nil, fmt.Errorf("no result in SSE stream")
}

// validateServerURL rejects URLs that target loopback or link-local addresses
// to reduce accidental misconfiguration and SSRF risk.
func validateServerURL(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("unsupported scheme %q (only http/https allowed)", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("missing host")
	}
	// Block obvious localhost aliases
	lower := strings.ToLower(host)
	if lower == "localhost" || strings.HasSuffix(lower, ".localhost") {
		return fmt.Errorf("loopback host not allowed: %s", host)
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return nil // hostname — allow; DNS resolution not checked at config time
	}
	if ip.IsLoopback() {
		return fmt.Errorf("loopback IP not allowed: %s", host)
	}
	if ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
		return fmt.Errorf("link-local IP not allowed: %s", host)
	}
	return nil
}
