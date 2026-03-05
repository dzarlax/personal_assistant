package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

// Client manages connections to multiple MCP servers.
type Client struct {
	servers     map[string]*server
	tools       []Tool
	toolServers map[string]string // tool name → server name
	logger      *slog.Logger
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
}

func NewClient(configs map[string]config.MCPServerConfig, logger *slog.Logger) *Client {
	c := &Client{
		servers:     make(map[string]*server),
		toolServers: make(map[string]string),
		logger:      logger,
	}
	for name, cfg := range configs {
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
			c.tools = append(c.tools, t)
			c.toolServers[t.Name] = name
			allowed++
		}
		c.logger.Info("mcp server connected", "server", name, "tools", allowed, "total", len(tools))
	}
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

// CallTool executes a tool on the appropriate MCP server.
func (c *Client) CallTool(ctx context.Context, name string, argsJSON string) (string, error) {
	serverName, ok := c.toolServers[name]
	if !ok {
		return "", fmt.Errorf("unknown tool: %s", name)
	}
	srv := c.servers[serverName]
	return srv.callTool(ctx, name, json.RawMessage(argsJSON))
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
		tools = append(tools, Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	return tools, nil
}

func (s *server) callTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
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
	scanner := bufio.NewScanner(r)
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
