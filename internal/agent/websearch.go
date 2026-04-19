package agent

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

	"telegram-agent/internal/llm"
)

const (
	webSearchToolName   = "web_search"
	webSearchMaxResults = 5
	webSearchTimeout    = 15 * time.Second

	webSearchProviderOllama = "ollama"
	webSearchProviderTavily = "tavily"

	ollamaDefaultBaseURL = "https://ollama.com"
	tavilyDefaultBaseURL = "https://api.tavily.com"

	webSearchCacheTTL       = 6 * time.Hour
	webSearchCacheThreshold = 0.95
	webSearchCacheMaxSize   = 100
)

var webSearchHTTPClient = &http.Client{Timeout: webSearchTimeout}

type webSearchCacheEntry struct {
	emb       []float32
	result    string
	expiresAt time.Time
}

// webSearchCache is a global in-memory cache of web_search results keyed by
// query embedding. Not per-chat because search results are not user-specific.
type webSearchCache struct {
	mu      sync.RWMutex
	entries []webSearchCacheEntry
}

func newWebSearchCache() *webSearchCache {
	return &webSearchCache{}
}

func (c *webSearchCache) get(emb []float32) (string, bool) {
	if len(emb) == 0 {
		return "", false
	}
	now := time.Now()
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, e := range c.entries {
		if now.After(e.expiresAt) {
			continue
		}
		if compactCosine(e.emb, emb) >= webSearchCacheThreshold {
			return e.result, true
		}
	}
	return "", false
}

func (c *webSearchCache) set(emb []float32, result string) {
	if len(emb) == 0 {
		return
	}
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	live := c.entries[:0]
	for _, e := range c.entries {
		if now.Before(e.expiresAt) {
			live = append(live, e)
		}
	}
	c.entries = live
	if len(c.entries) >= webSearchCacheMaxSize {
		c.entries = c.entries[1:]
	}
	c.entries = append(c.entries, webSearchCacheEntry{
		emb:       emb,
		result:    result,
		expiresAt: now.Add(webSearchCacheTTL),
	})
}

// WebSearchConfig holds configuration for the web search tool.
// Provider selects the backend: "ollama" (default) or "tavily".
type WebSearchConfig struct {
	Provider string
	BaseURL  string
	APIKey   string
}

// webSearchTool returns the tool definition for LLM function calling.
func webSearchTool() llm.Tool {
	return llm.Tool{
		Name:        webSearchToolName,
		Description: "Search the web for current information. Use when the user asks about recent events, real-time data (news, prices, weather, schedules, current status), explicitly asks to look something up online, or when you are not confident the answer is correct and current. Prefer answering from knowledge for stable facts (definitions, established science, coding help) where your training data is reliable. If unsure — search rather than guess.",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"query": {
					"type": "string",
					"description": "The search query"
				}
			},
			"required": ["query"]
		}`),
	}
}

// callWebSearchCached checks the query cache by embedding before hitting the
// provider. Falls back to direct call when embeddings are unavailable.
func (a *Agent) callWebSearchCached(ctx context.Context, argsJSON string) (string, error) {
	var args struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse web_search args: %w", err)
	}
	if args.Query == "" {
		return "", fmt.Errorf("web_search: query is required")
	}

	var emb []float32
	if a.mcp != nil {
		if e, err := a.mcp.EmbedText(ctx, args.Query); err == nil {
			emb = e
			if cached, ok := a.webSearchCache.get(emb); ok {
				a.logger.Info("web_search cache hit", "query", args.Query)
				return cached, nil
			}
		}
	}

	result, err := callWebSearch(ctx, *a.webSearch, argsJSON)
	if err != nil {
		return "", err
	}
	if len(emb) > 0 {
		a.webSearchCache.set(emb, result)
	}
	return result, nil
}

// callWebSearch dispatches to the configured provider.
func callWebSearch(ctx context.Context, cfg WebSearchConfig, argsJSON string) (string, error) {
	var args struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse web_search args: %w", err)
	}
	if args.Query == "" {
		return "", fmt.Errorf("web_search: query is required")
	}

	switch strings.ToLower(cfg.Provider) {
	case "", webSearchProviderOllama:
		return callOllamaWebSearch(ctx, cfg, args.Query)
	case webSearchProviderTavily:
		return callTavilySearch(ctx, cfg, args.Query)
	default:
		return "", fmt.Errorf("web_search: unknown provider %q", cfg.Provider)
	}
}

func callOllamaWebSearch(ctx context.Context, cfg WebSearchConfig, query string) (string, error) {
	reqBody, _ := json.Marshal(map[string]any{
		"query":       query,
		"max_results": webSearchMaxResults,
	})

	base := cfg.BaseURL
	if base == "" {
		base = ollamaDefaultBaseURL
	}
	url := strings.TrimRight(base, "/") + "/api/web_search"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("web_search: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	resp, err := webSearchHTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("web_search: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024))
	if err != nil {
		return "", fmt.Errorf("web_search: read response: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("web_search: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("web_search: parse response: %w", err)
	}

	if len(result.Results) == 0 {
		return "No results found.", nil
	}

	var sb strings.Builder
	for i, r := range result.Results {
		fmt.Fprintf(&sb, "%d. %s\n   %s\n   %s\n\n", i+1, r.Title, r.URL, r.Content)
	}
	return sb.String(), nil
}

func callTavilySearch(ctx context.Context, cfg WebSearchConfig, query string) (string, error) {
	if cfg.APIKey == "" {
		return "", fmt.Errorf("web_search: tavily requires api_key")
	}

	reqBody, _ := json.Marshal(map[string]any{
		"query":          query,
		"max_results":    webSearchMaxResults,
		"search_depth":   "basic",
		"include_answer": true,
	})

	base := cfg.BaseURL
	if base == "" {
		base = tavilyDefaultBaseURL
	}
	url := strings.TrimRight(base, "/") + "/search"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("web_search: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	resp, err := webSearchHTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("web_search: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024))
	if err != nil {
		return "", fmt.Errorf("web_search: read response: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("web_search: tavily HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Answer  string `json:"answer"`
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return "", fmt.Errorf("web_search: parse tavily response: %w", err)
	}

	if result.Answer == "" && len(result.Results) == 0 {
		return "No results found.", nil
	}

	var sb strings.Builder
	if result.Answer != "" {
		fmt.Fprintf(&sb, "Answer: %s\n\nSources:\n", result.Answer)
	}
	for i, r := range result.Results {
		fmt.Fprintf(&sb, "%d. %s\n   %s\n   %s\n\n", i+1, r.Title, r.URL, r.Content)
	}
	return sb.String(), nil
}
