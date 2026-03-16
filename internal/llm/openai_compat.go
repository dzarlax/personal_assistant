package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

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
	model     string
	maxTokens int
	provName  string
	vision    bool // if false, image_url parts in history are replaced with [image]
}

func newOpenAICompat(cfg config.ModelConfig, defaultBaseURL, providerName string, vision ...bool) (*openAICompatProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%s: api_key is required", providerName)
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
	rawMsgs := buildMessages(messages, systemPrompt, p.vision)
	req := rawChatRequest{
		Model:     p.model,
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

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("%s: create request: %w", p.provName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

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
	return p.provName + "/" + p.model
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
