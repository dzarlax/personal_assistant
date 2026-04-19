package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	readability "github.com/go-shiori/go-readability"

	"github.com/chromedp/chromedp"

	"telegram-agent/internal/llm"
)

const (
	webFetchToolName = "web_fetch"
	webFetchTimeout  = 20 * time.Second
	webFetchCDPNav   = 25 * time.Second
	webFetchMinChars = 200 // readability output shorter than this triggers CDP fallback
	webFetchMaxChars = 8000
)

var webFetchHTTPClient = &http.Client{Timeout: webFetchTimeout}

// WebFetchConfig holds configuration for the web_fetch tool.
// CDPURL is optional; when empty only the HTTP+readability path is used.
type WebFetchConfig struct {
	CDPURL string // e.g. http://infra-chrome:9222
}

func webFetchTool() llm.Tool {
	return llm.Tool{
		Name:        webFetchToolName,
		Description: "Fetch a URL and return its main article text. Use when the user shares a link and wants a summary, key facts, or to ask questions about its contents. Do not use for search queries (use web_search instead).",
		InputSchema: json.RawMessage(`{
			"type": "object",
			"properties": {
				"url": {
					"type": "string",
					"description": "Absolute http(s) URL to fetch"
				}
			},
			"required": ["url"]
		}`),
	}
}

func (a *Agent) callWebFetchCached(ctx context.Context, argsJSON string) (string, error) {
	var args struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		return "", fmt.Errorf("parse web_fetch args: %w", err)
	}
	if args.URL == "" {
		return "", fmt.Errorf("web_fetch: url is required")
	}
	parsed, err := url.Parse(args.URL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", fmt.Errorf("web_fetch: invalid url")
	}

	var emb []float32
	if a.mcp != nil {
		if e, err := a.mcp.EmbedText(ctx, args.URL); err == nil {
			emb = e
			if cached, ok := a.webSearchCache.get(emb); ok {
				a.logger.Info("web_fetch cache hit", "url", args.URL)
				return cached, nil
			}
		}
	}

	result, err := a.fetchAndExtract(ctx, args.URL)
	if err != nil {
		return "", err
	}
	if len(emb) > 0 {
		a.webSearchCache.set(emb, result)
	}
	return result, nil
}

func (a *Agent) fetchAndExtract(ctx context.Context, pageURL string) (string, error) {
	text, title, err := fetchViaHTTP(ctx, pageURL)
	if err == nil && len(text) >= webFetchMinChars {
		return formatFetchResult(title, pageURL, text), nil
	}

	// Primary failed or too short — try CDP fallback if configured.
	if a.webFetch != nil && a.webFetch.CDPURL != "" {
		a.logger.Info("web_fetch: falling back to CDP", "url", pageURL, "primary_err", err)
		text2, title2, err2 := fetchViaCDP(ctx, a.webFetch.CDPURL, pageURL)
		if err2 == nil && len(text2) >= webFetchMinChars {
			return formatFetchResult(title2, pageURL, text2), nil
		}
		if err2 != nil {
			return "", fmt.Errorf("web_fetch: both paths failed: http=%v, cdp=%v", err, err2)
		}
		// CDP returned short content; use whichever is longer.
		if len(text2) > len(text) {
			return formatFetchResult(title2, pageURL, text2), nil
		}
	}

	if err != nil {
		return "", fmt.Errorf("web_fetch: %w", err)
	}
	if len(text) == 0 {
		return "No readable content extracted.", nil
	}
	return formatFetchResult(title, pageURL, text), nil
}

func fetchViaHTTP(ctx context.Context, pageURL string) (string, string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", pageURL, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; personal-assistant/1.0)")
	resp, err := webFetchHTTPClient.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 5*1024*1024))
	if err != nil {
		return "", "", err
	}

	parsed, _ := url.Parse(pageURL)
	article, err := readability.FromReader(strings.NewReader(string(body)), parsed)
	if err != nil {
		return "", "", err
	}
	return strings.TrimSpace(article.TextContent), strings.TrimSpace(article.Title), nil
}

func fetchViaCDP(ctx context.Context, cdpURL, pageURL string) (string, string, error) {
	allocCtx, cancelAlloc := chromedp.NewRemoteAllocator(ctx, cdpURL)
	defer cancelAlloc()

	taskCtx, cancelTask := chromedp.NewContext(allocCtx)
	defer cancelTask()

	taskCtx, cancelTimeout := context.WithTimeout(taskCtx, webFetchCDPNav)
	defer cancelTimeout()

	var html, title string
	if err := chromedp.Run(taskCtx,
		chromedp.Navigate(pageURL),
		chromedp.WaitReady("body", chromedp.ByQuery),
		chromedp.Title(&title),
		chromedp.OuterHTML("html", &html, chromedp.ByQuery),
	); err != nil {
		return "", "", err
	}

	parsed, _ := url.Parse(pageURL)
	article, err := readability.FromReader(strings.NewReader(html), parsed)
	if err != nil {
		// Fallback: plain innerText
		var txt string
		_ = chromedp.Run(taskCtx, chromedp.Text("body", &txt, chromedp.ByQuery))
		return strings.TrimSpace(txt), strings.TrimSpace(title), nil
	}
	if strings.TrimSpace(article.Title) != "" {
		title = article.Title
	}
	return strings.TrimSpace(article.TextContent), strings.TrimSpace(title), nil
}

func formatFetchResult(title, pageURL, text string) string {
	if len(text) > webFetchMaxChars {
		text = text[:webFetchMaxChars] + "\n\n[truncated]"
	}
	var sb strings.Builder
	if title != "" {
		fmt.Fprintf(&sb, "Title: %s\n", title)
	}
	fmt.Fprintf(&sb, "URL: %s\n\n%s", pageURL, text)
	return sb.String()
}
