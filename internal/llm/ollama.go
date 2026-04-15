package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"telegram-agent/internal/config"
)

// --- Ollama Cloud model discovery ---

const ollamaRegistryURL = "https://ollama.com"

var (
	cloudMu        sync.Mutex
	cloudCache     *ollamaCloudCache
	cloudCachePath string // set via InitCloudModelCache
	cloudTTL       = 24 * time.Hour
)

// CloudModel represents a model available on Ollama Cloud.
type CloudModel struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

// ollamaCloudCache is the on-disk + in-memory cache for cloud models and capabilities.
type ollamaCloudCache struct {
	Models    []CloudModel          `json:"models"`
	Caps      map[string][]string   `json:"caps"`
	FetchedAt time.Time             `json:"fetched_at"`
}

// InitCloudModelCache sets the file path for persistent cloud model cache.
// Call once at startup before any Fetch calls.
func InitCloudModelCache(path string) {
	cloudMu.Lock()
	defer cloudMu.Unlock()
	cloudCachePath = path
	// Load existing cache from disk.
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	var c ollamaCloudCache
	if json.Unmarshal(data, &c) == nil && len(c.Models) > 0 {
		cloudCache = &c
	}
}

func saveCloudCache() {
	if cloudCachePath == "" || cloudCache == nil {
		return
	}
	data, err := json.Marshal(cloudCache)
	if err != nil {
		return
	}
	os.WriteFile(cloudCachePath, data, 0644) //nolint:errcheck
}

// FetchOllamaCloudModels fetches the list of cloud-available models via /api/tags.
// Uses in-memory cache (24h TTL), falls back to persistent disk cache.
func FetchOllamaCloudModels() ([]CloudModel, error) {
	cloudMu.Lock()
	defer cloudMu.Unlock()

	if cloudCache != nil && time.Since(cloudCache.FetchedAt) < cloudTTL && len(cloudCache.Models) > 0 {
		return cloudCache.Models, nil
	}

	resp, err := http.Get(ollamaRegistryURL + "/api/tags")
	if err != nil {
		// Return stale cache on network failure.
		if cloudCache != nil && len(cloudCache.Models) > 0 {
			return cloudCache.Models, nil
		}
		return nil, fmt.Errorf("fetch ollama cloud models: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		if cloudCache != nil && len(cloudCache.Models) > 0 {
			return cloudCache.Models, nil
		}
		return nil, fmt.Errorf("parse ollama cloud models: %w", err)
	}

	models := make([]CloudModel, 0, len(result.Models))
	for _, m := range result.Models {
		if m.Name != "" {
			models = append(models, CloudModel{Name: m.Name, Size: m.Size})
		}
	}

	if len(models) > 0 {
		if cloudCache == nil {
			cloudCache = &ollamaCloudCache{Caps: make(map[string][]string)}
		}
		cloudCache.Models = models
		cloudCache.FetchedAt = time.Now()
		saveCloudCache()
	}
	return models, nil
}

// FetchOllamaModelCaps fetches capabilities for a specific model via /api/show.
// Returns a slice like ["completion","thinking","tools","vision"].
// Results are cached per model persistently.
func FetchOllamaModelCaps(model string) ([]string, error) {
	cloudMu.Lock()
	defer cloudMu.Unlock()

	// Check in-memory/disk cache.
	if cloudCache != nil {
		if caps, ok := cloudCache.Caps[model]; ok {
			return caps, nil
		}
	}

	body, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return nil, err
	}
	resp, err := http.Post(ollamaRegistryURL+"/api/show", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("fetch ollama model caps: %w", err)
	}
	defer resp.Body.Close()

	var result struct {
		Capabilities []string `json:"capabilities"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("parse ollama model caps: %w", err)
	}

	if cloudCache == nil {
		cloudCache = &ollamaCloudCache{Caps: make(map[string][]string)}
	}
	cloudCache.Caps[model] = result.Capabilities
	saveCloudCache()
	return result.Capabilities, nil
}

const defaultOllamaBaseURL = "http://localhost:11434"

// ollamaProvider implements the native Ollama /api/chat protocol,
// which differs from the OpenAI-compatible /v1 endpoint.
type ollamaProvider struct {
	baseURL   string
	apiKey    string
	model     string
	maxTokens int
	provName  string
	vision    bool
	noThink   bool
	keepAlive int // seconds; -1 = keep forever; 0 = provider default
}

// NewOllama creates a provider for a local Ollama instance.
// API key is optional.
func NewOllama(cfg config.ModelConfig) (*ollamaProvider, error) {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultOllamaBaseURL
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	return &ollamaProvider{
		baseURL:   strings.TrimRight(baseURL, "/"),
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		maxTokens: maxTokens,
		provName:  "ollama",
		vision:    cfg.Vision,
		noThink:   cfg.NoThink,
		keepAlive: cfg.KeepAlive,
	}, nil
}

func (p *ollamaProvider) Name() string {
	return p.provName + "/" + p.model
}

func (p *ollamaProvider) SupportsVision() bool {
	return p.vision
}

// SetModelAndCaps changes the model name and vision flag on a running provider.
// Used by the router to switch Ollama Cloud models dynamically.
func (p *ollamaProvider) SetModelAndCaps(model string, vision bool) {
	p.model = model
	p.vision = vision
}

// OllamaCloudModel returns the current model name if this is a cloud provider
// (base URL contains "ollama.com"), or ("", false) otherwise.
func (p *ollamaProvider) OllamaCloudModel() (string, bool) {
	if strings.Contains(p.baseURL, "ollama.com") {
		return p.model, true
	}
	return "", false
}

// --- Ollama-native request/response types ---

type ollamaRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	Think     *bool           `json:"think,omitempty"`
	Tools     []ollamaTool    `json:"tools,omitempty"`
	Options   *ollamaOptions  `json:"options,omitempty"`
	KeepAlive *int            `json:"keep_alive,omitempty"` // seconds; -1 = forever
}

type ollamaMessage struct {
	Role      string            `json:"role"`
	Content   string            `json:"content"`
	Images    []string          `json:"images,omitempty"`
	ToolCalls []ollamaToolCall  `json:"tool_calls,omitempty"`
	ToolName  string            `json:"tool_name,omitempty"`
}

type ollamaToolCall struct {
	ID       string             `json:"id,omitempty"`
	Function ollamaFunctionCall `json:"function"`
}

type ollamaFunctionCall struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"` // object in response, we send string→object
}

type ollamaTool struct {
	Type     string              `json:"type"`
	Function ollamaFunctionDef   `json:"function"`
}

type ollamaFunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ollamaOptions struct {
	NumPredict int `json:"num_predict,omitempty"`
}

type ollamaResponse struct {
	Model   string        `json:"model"`
	Message ollamaMessage `json:"message"`
	Done    bool          `json:"done"`
	Error   string        `json:"error,omitempty"`
}

func (p *ollamaProvider) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	ollamaMsgs := p.buildMessages(messages, systemPrompt)

	req := ollamaRequest{
		Model:    p.model,
		Messages: ollamaMsgs,
		Stream:   false,
		Options:  &ollamaOptions{NumPredict: p.maxTokens},
	}
	if p.noThink {
		f := false
		req.Think = &f
	}
	if p.keepAlive != 0 {
		ka := p.keepAlive
		req.KeepAlive = &ka
	}
	if len(tools) > 0 {
		req.Tools = buildOllamaTools(tools)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("%s: marshal request: %w", p.provName, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("%s: create request: %w", p.provName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("%s: %w", p.provName, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, fmt.Errorf("%s: read response: %w", p.provName, err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return Response{}, &APIError{StatusCode: httpResp.StatusCode, Message: string(respBody)}
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return Response{}, fmt.Errorf("%s: parse response: %w", p.provName, err)
	}
	if ollamaResp.Error != "" {
		return Response{}, &APIError{StatusCode: httpResp.StatusCode, Message: ollamaResp.Error}
	}

	result := Response{Content: ollamaResp.Message.Content}
	for i, tc := range ollamaResp.Message.ToolCalls {
		// Ollama returns arguments as an object; serialize to JSON string for our internal format.
		argsJSON, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			argsJSON = []byte("{}")
		}
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("call_%d", i)
		}
		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:        callID,
			Name:      tc.Function.Name,
			Arguments: string(argsJSON),
		})
	}
	return result, nil
}

func (p *ollamaProvider) buildMessages(messages []Message, systemPrompt string) []ollamaMessage {
	msgs := make([]ollamaMessage, 0, len(messages)+1)
	if systemPrompt != "" {
		msgs = append(msgs, ollamaMessage{Role: "system", Content: systemPrompt})
	}
	// Build a map from tool call ID → tool name for resolving tool results.
	toolCallNames := make(map[string]string)
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			if tc.ID != "" {
				toolCallNames[tc.ID] = tc.Name
			}
		}
	}

	for _, m := range messages {
		msg := ollamaMessage{
			Role:    m.Role,
			Content: m.Content,
		}

		// Ollama uses "tool_name" instead of "tool_call_id" for tool results.
		if m.Role == "tool" && m.ToolCallID != "" {
			if name, ok := toolCallNames[m.ToolCallID]; ok {
				msg.ToolName = name
			}
		}

		// Handle multimodal parts.
		if len(m.Parts) > 0 {
			var textParts []string
			for _, part := range m.Parts {
				switch part.Type {
				case "text":
					textParts = append(textParts, part.Text)
				case "image_url":
					if part.ImageURL != nil {
						// Ollama expects raw base64 without the data URI prefix.
						b64 := part.ImageURL.URL
						if idx := strings.Index(b64, ","); idx >= 0 {
							b64 = b64[idx+1:]
						}
						msg.Images = append(msg.Images, b64)
					}
				}
			}
			if len(textParts) > 0 {
				msg.Content = strings.Join(textParts, "\n")
			}
		}

		// Convert internal tool calls to Ollama format.
		for _, tc := range m.ToolCalls {
			var args any
			if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
				args = map[string]any{}
			}
			msg.ToolCalls = append(msg.ToolCalls, ollamaToolCall{
				Function: ollamaFunctionCall{
					Name:      tc.Name,
					Arguments: args,
				},
			})
		}

		msgs = append(msgs, msg)
	}
	return msgs
}

func buildOllamaTools(tools []Tool) []ollamaTool {
	result := make([]ollamaTool, 0, len(tools))
	for _, t := range tools {
		var params any
		if len(t.InputSchema) > 0 {
			params = json.RawMessage(t.InputSchema)
		} else {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		result = append(result, ollamaTool{
			Type: "function",
			Function: ollamaFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return result
}
