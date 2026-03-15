package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"telegram-agent/internal/llm"
)

const (
	webSearchToolName    = "web_search"
	webSearchMaxResults  = 5
	webSearchTimeout     = 15 * time.Second
)

var webSearchHTTPClient = &http.Client{Timeout: webSearchTimeout}

// WebSearchConfig holds configuration for the Ollama web search tool.
type WebSearchConfig struct {
	BaseURL string // e.g. "https://ollama.com"
	APIKey  string
}

// webSearchTool returns the tool definition for LLM function calling.
func webSearchTool() llm.Tool {
	return llm.Tool{
		Name:        webSearchToolName,
		Description: "Search the web for current information. Use when the user asks about recent events, real-time data, or anything that may not be in your training data.",
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

// callWebSearch executes a search against the Ollama web search API.
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

	reqBody, _ := json.Marshal(map[string]any{
		"query":       args.Query,
		"max_results": webSearchMaxResults,
	})

	url := strings.TrimRight(cfg.BaseURL, "/") + "/api/web_search"
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

	body, err := io.ReadAll(io.LimitReader(resp.Body, 1*1024*1024)) // 1 MB cap
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
