package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"telegram-agent/internal/config"
)

// APIError represents an HTTP-level error from an OpenAI-compatible API.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("api error (HTTP %d): %s", e.StatusCode, e.Message)
}

type openAICompatProvider struct {
	baseURL   string
	apiKey    string
	maxTokens int
	provName  string

	// Mutable state protected by mu: model id, capabilities, vision flag.
	// OpenRouter updates these via SetModel at runtime when the admin UI
	// reassigns a slot; in-flight requests keep the snapshot they captured.
	mu      sync.RWMutex
	model   string
	caps    Capabilities
	vision  bool // if false, image_url parts in history are replaced with [image]

	// Optional per-provider extensions. Used by OpenRouter to set
	// HTTP-Referer/X-Title headers and provider/usage routing fields.
	extraHeaders    map[string]string
	extraBodyFields map[string]any
}

// snapshot captures mutable fields under the lock for a single request.
func (p *openAICompatProvider) snapshot() (model string, vision bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.model, p.vision
}

// Capabilities returns the current model's capabilities (may be zero-value
// when the store hasn't been queried or provider lacks a /models endpoint).
func (p *openAICompatProvider) Capabilities() Capabilities {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.caps
}

// SetModel swaps the model id and its capabilities atomically. Subsequent
// requests use the new id; in-flight requests are unaffected.
// Vision flag is derived from caps; callers can override by calling SetVision after.
func (p *openAICompatProvider) SetModel(modelID string, caps Capabilities) {
	p.mu.Lock()
	p.model = modelID
	p.caps = caps
	p.vision = caps.Vision
	p.mu.Unlock()
}

// CurrentModel returns the currently active model id.
func (p *openAICompatProvider) CurrentModel() string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.model
}

// mergeExtraBody merges provider-specific fields into a marshalled chat request.
// Returns the original body unchanged when there are no extras.
func (p *openAICompatProvider) mergeExtraBody(body []byte) ([]byte, error) {
	if len(p.extraBodyFields) == 0 {
		return body, nil
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, err
	}
	for k, v := range p.extraBodyFields {
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		m[k] = raw
	}
	return json.Marshal(m)
}

func (p *openAICompatProvider) applyExtraHeaders(req *http.Request) {
	for k, v := range p.extraHeaders {
		req.Header.Set(k, v)
	}
}

func newOpenAICompat(cfg config.ModelConfig, defaultBaseURL, providerName string, vision ...bool) (*openAICompatProvider, error) {
	if cfg.APIKey == "" && cfg.BaseURL == "" {
		return nil, fmt.Errorf("%s: api_key or base_url is required", providerName)
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	p := &openAICompatProvider{
		baseURL:   strings.TrimRight(baseURL, "/"),
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		maxTokens: maxTokens,
		provName:  providerName,
	}
	if len(vision) > 0 {
		p.vision = vision[0]
	}
	return p, nil
}

// --- Raw JSON types for full serialization control ---

type rawChatRequest struct {
	Model     string       `json:"model"`
	Messages  []rawMessage `json:"messages"`
	MaxTokens int          `json:"max_tokens,omitempty"`
	Tools     []rawTool    `json:"tools,omitempty"`
	Stream    bool         `json:"stream,omitempty"`
}

// rawStreamDelta is an SSE chunk from an OpenAI-compatible streaming response.
type rawStreamDelta struct {
	Choices []struct {
		Delta struct {
			Content   string        `json:"content"`
			ToolCalls []rawToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
}

// rawMessage uses `any` for Content so we can serialize null, string, or array.
type rawMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content"` // nil → null, string, or []rawContentPart
	ToolCalls  []rawToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type rawContentPart struct {
	Type       string         `json:"type"`
	Text       string         `json:"text,omitempty"`
	ImageURL   *rawImageURL   `json:"image_url,omitempty"`
	InputAudio *rawInputAudio `json:"input_audio,omitempty"`
}

type rawImageURL struct {
	URL string `json:"url"`
}

type rawInputAudio struct {
	Data   string `json:"data"`   // base64-encoded audio
	Format string `json:"format"` // "wav", "mp3", "ogg"
}

type rawToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function rawFunctionCall `json:"function"`
}

type rawFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type rawTool struct {
	Type     string             `json:"type"`
	Function rawToolFunctionDef `json:"function"`
}

type rawToolFunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type rawChatResponse struct {
	Choices []struct {
		Message struct {
			Content   string        `json:"content"`
			ToolCalls []rawToolCall `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (p *openAICompatProvider) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	model, vision := p.snapshot()
	rawMsgs := buildMessages(messages, systemPrompt, vision)
	req := rawChatRequest{
		Model:     model,
		Messages:  rawMsgs,
		MaxTokens: p.maxTokens,
	}
	if len(tools) > 0 {
		req.Tools = buildTools(tools)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("%s: marshal request: %w", p.provName, err)
	}
	body, err = p.mergeExtraBody(body)
	if err != nil {
		return Response{}, fmt.Errorf("%s: merge extra body: %w", p.provName, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("%s: create request: %w", p.provName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	p.applyExtraHeaders(httpReq)

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("%s: %w", p.provName, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, fmt.Errorf("%s: read response: %w", p.provName, err)
	}

	var chatResp rawChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return Response{}, fmt.Errorf("%s: parse response: %w", p.provName, err)
	}

	if chatResp.Error != nil {
		return Response{}, &APIError{StatusCode: httpResp.StatusCode, Message: chatResp.Error.Message}
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return Response{}, &APIError{StatusCode: httpResp.StatusCode, Message: string(respBody)}
	}
	if len(chatResp.Choices) == 0 {
		return Response{}, fmt.Errorf("%s: empty response", p.provName)
	}

	choice := chatResp.Choices[0].Message
	result := Response{Content: choice.Content}
	for _, tc := range choice.ToolCalls {
		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}
	return result, nil
}

func (p *openAICompatProvider) Name() string {
	return p.provName + "/" + p.CurrentModel()
}

func (p *openAICompatProvider) SupportsVision() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.vision
}

// buildMessages converts internal messages to raw JSON messages.
// Assistant messages with tool_calls and empty content use null (not omitted).
func buildMessages(messages []Message, systemPrompt string, vision bool) []rawMessage {
	msgs := make([]rawMessage, 0, len(messages)+1)
	if systemPrompt != "" {
		msgs = append(msgs, rawMessage{
			Role:    "system",
			Content: systemPrompt,
		})
	}
	for _, m := range messages {
		msg := rawMessage{
			Role:       m.Role,
			ToolCallID: m.ToolCallID,
		}
		if len(m.Parts) > 0 {
			var parts []rawContentPart
			for _, part := range m.Parts {
				switch part.Type {
				case "text":
					parts = append(parts, rawContentPart{Type: "text", Text: part.Text})
				case "image_url":
					if !vision {
						parts = append(parts, rawContentPart{Type: "text", Text: "[image]"})
					} else if part.ImageURL != nil {
						parts = append(parts, rawContentPart{
							Type:     "image_url",
							ImageURL: &rawImageURL{URL: part.ImageURL.URL},
						})
					}
				case "input_audio":
					if !vision {
						parts = append(parts, rawContentPart{Type: "text", Text: "[audio]"})
					} else if part.InputAudio != nil {
						parts = append(parts, rawContentPart{
							Type:       "input_audio",
							InputAudio: &rawInputAudio{Data: part.InputAudio.Data, Format: part.InputAudio.Format},
						})
					}
				case "inline_data":
					parts = append(parts, rawContentPart{Type: "text", Text: "[document]"})
				}
			}
			msg.Content = parts
		} else if m.Role == "assistant" && len(m.ToolCalls) > 0 && m.Content == "" {
			// Use null instead of omitting — providers reject messages with no content field.
			msg.Content = nil
		} else {
			msg.Content = m.Content
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, rawToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: rawFunctionCall{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			})
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

func buildTools(tools []Tool) []rawTool {
	result := make([]rawTool, 0, len(tools))
	for _, t := range tools {
		var params any
		if len(t.InputSchema) > 0 {
			params = json.RawMessage(t.InputSchema)
		} else {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		result = append(result, rawTool{
			Type: "function",
			Function: rawToolFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return result
}

// ChatStream opens an SSE stream to the OpenAI-compatible API and returns chunks via a channel.
func (p *openAICompatProvider) ChatStream(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error) {
	model, vision := p.snapshot()
	rawMsgs := buildMessages(messages, systemPrompt, vision)
	req := rawChatRequest{
		Model:     model,
		Messages:  rawMsgs,
		MaxTokens: p.maxTokens,
		Stream:    true,
	}
	if len(tools) > 0 {
		req.Tools = buildTools(tools)
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal request: %w", p.provName, err)
	}
	body, err = p.mergeExtraBody(body)
	if err != nil {
		return nil, fmt.Errorf("%s: merge extra body: %w", p.provName, err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("%s: create request: %w", p.provName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	p.applyExtraHeaders(httpReq)

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", p.provName, err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		defer httpResp.Body.Close()
		respBody, _ := io.ReadAll(httpResp.Body)
		return nil, &APIError{StatusCode: httpResp.StatusCode, Message: string(respBody)}
	}

	ch := make(chan StreamChunk, 64)
	go p.readSSE(httpResp, ch)
	return ch, nil
}

// readSSE reads SSE lines from an HTTP response and sends StreamChunks.
func (p *openAICompatProvider) readSSE(resp *http.Response, ch chan<- StreamChunk) {
	defer close(ch)
	defer resp.Body.Close()

	// Accumulate tool calls across deltas (they arrive in fragments).
	var toolCallsByIdx = make(map[int]*rawToolCall)

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var delta rawStreamDelta
		if err := json.Unmarshal([]byte(data), &delta); err != nil {
			continue
		}
		if len(delta.Choices) == 0 {
			continue
		}

		choice := delta.Choices[0]

		// Text content delta — send immediately.
		if choice.Delta.Content != "" {
			ch <- StreamChunk{Delta: choice.Delta.Content}
		}

		// Tool call deltas — accumulate silently.
		for _, tc := range choice.Delta.ToolCalls {
			idx := 0 // default index for single tool call
			if tc.ID != "" {
				// New tool call — find its index.
				idx = len(toolCallsByIdx)
				toolCallsByIdx[idx] = &rawToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: rawFunctionCall{
						Name: tc.Function.Name,
					},
				}
			} else {
				// Continuation of existing tool call — append arguments.
				idx = len(toolCallsByIdx) - 1
			}
			if existing, ok := toolCallsByIdx[idx]; ok {
				existing.Function.Arguments += tc.Function.Arguments
			}
		}
	}

	// Build final chunk.
	final := StreamChunk{Done: true}
	if len(toolCallsByIdx) > 0 {
		for i := 0; i < len(toolCallsByIdx); i++ {
			tc := toolCallsByIdx[i]
			final.ToolCalls = append(final.ToolCalls, ToolCall{
				ID:        tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}
	if err := scanner.Err(); err != nil {
		final.Err = err
	}
	ch <- final
}
