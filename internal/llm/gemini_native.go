package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"telegram-agent/internal/config"
)

const (
	geminiAPIBase        = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
	geminiDefaultTimeout = 5 * time.Minute
)

var geminiHTTPClient = &http.Client{Timeout: geminiDefaultTimeout}

// GeminiNativeProvider uses the native Gemini generateContent API.
// Supports inline audio/video/documents, Google Search grounding, and thinking mode.
type GeminiNativeProvider struct {
	model     string
	apiKey    string
	maxTokens int
	provName  string
	grounding bool // enable Google Search grounding
}

// NewGeminiNative creates a provider using the native Gemini API.
func NewGeminiNative(cfg config.ModelConfig, grounding bool) (*GeminiNativeProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gemini-native: api_key is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("gemini-native: model is required")
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	return &GeminiNativeProvider{
		model:     cfg.Model,
		apiKey:    cfg.APIKey,
		maxTokens: maxTokens,
		provName:  "gemini-native",
		grounding: grounding,
	}, nil
}

func (p *GeminiNativeProvider) Name() string {
	return p.provName + "/" + p.model
}

// --- Native Gemini request/response types ---

type geminiRequest struct {
	Contents          []geminiContent        `json:"contents"`
	SystemInstruction *geminiContent         `json:"system_instruction,omitempty"`
	Tools             []geminiToolDecl       `json:"tools,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generation_config,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

// geminiPart uses pointers/omitempty so only one field is serialised per part.
type geminiPart struct {
	Text             string                `json:"text,omitempty"`
	InlineData       *geminiInlineData     `json:"inline_data,omitempty"`
	FunctionCall     *geminiFunctionCall   `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResp   `json:"functionResponse,omitempty"`
}

type geminiInlineData struct {
	MIMEType string `json:"mime_type"`
	Data     string `json:"data"` // base64
}

type geminiFunctionCall struct {
	Name string `json:"name"`
	Args any    `json:"args"`
}

type geminiFunctionResp struct {
	Name     string `json:"name"`
	Response any    `json:"response"`
}

type geminiToolDecl struct {
	FunctionDeclarations []geminiFuncDecl `json:"functionDeclarations,omitempty"`
	GoogleSearch         *struct{}        `json:"google_search,omitempty"`
}

type geminiFuncDecl struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type geminiGenerationConfig struct {
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error"`
}

func (p *GeminiNativeProvider) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	req := geminiRequest{
		Contents: p.buildContents(messages),
		GenerationConfig: &geminiGenerationConfig{
			MaxOutputTokens: p.maxTokens,
		},
	}

	if systemPrompt != "" {
		req.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: systemPrompt}},
		}
	}

	// Build tools: function declarations + optional grounding.
	var toolDecls []geminiToolDecl
	if len(tools) > 0 {
		toolDecls = append(toolDecls, geminiToolDecl{
			FunctionDeclarations: buildGeminiFuncDecls(tools),
		})
	}
	if p.grounding {
		toolDecls = append(toolDecls, geminiToolDecl{
			GoogleSearch: &struct{}{},
		})
	}
	if len(toolDecls) > 0 {
		req.Tools = toolDecls
	}

	body, err := json.Marshal(req)
	if err != nil {
		return Response{}, fmt.Errorf("%s: marshal: %w", p.provName, err)
	}

	url := fmt.Sprintf(geminiAPIBase, p.model)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("%s: request: %w", p.provName, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", p.apiKey)

	httpResp, err := geminiHTTPClient.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("%s: %w", p.provName, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return Response{}, fmt.Errorf("%s: read: %w", p.provName, err)
	}

	var gemResp geminiResponse
	if err := json.Unmarshal(respBody, &gemResp); err != nil {
		return Response{}, fmt.Errorf("%s: parse: %w", p.provName, err)
	}
	if gemResp.Error != nil {
		return Response{}, &APIError{StatusCode: gemResp.Error.Code, Message: gemResp.Error.Message}
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return Response{}, &APIError{StatusCode: httpResp.StatusCode, Message: string(respBody)}
	}
	if len(gemResp.Candidates) == 0 {
		return Response{}, fmt.Errorf("%s: empty response", p.provName)
	}

	// Parse response parts: text and/or function calls.
	var result Response
	for i, part := range gemResp.Candidates[0].Content.Parts {
		if part.Text != "" {
			if result.Content != "" {
				result.Content += "\n"
			}
			result.Content += part.Text
		}
		if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				argsJSON = []byte("{}")
			}
			result.ToolCalls = append(result.ToolCalls, ToolCall{
				ID:        fmt.Sprintf("call_%d", i),
				Name:      part.FunctionCall.Name,
				Arguments: string(argsJSON),
			})
		}
	}
	return result, nil
}

// buildContents converts internal messages to native Gemini format.
func (p *GeminiNativeProvider) buildContents(messages []Message) []geminiContent {
	// Build a map from tool call ID → function name for functionResponse.
	toolNames := make(map[string]string)
	for _, m := range messages {
		for _, tc := range m.ToolCalls {
			toolNames[tc.ID] = tc.Name
		}
	}

	var contents []geminiContent

	for _, m := range messages {
		role := m.Role
		switch role {
		case "assistant":
			role = "model"
		case "system":
			continue
		case "tool":
			funcName := toolNames[m.ToolCallID]
			if funcName == "" {
				funcName = m.ToolCallID // fallback
			}
			contents = append(contents, geminiContent{
				Role: "user",
				Parts: []geminiPart{{
					FunctionResponse: &geminiFunctionResp{
						Name:     funcName,
						Response: map[string]any{"result": m.Content},
					},
				}},
			})
			continue
		}

		var parts []geminiPart

		// Handle multimodal parts.
		if len(m.Parts) > 0 {
			for _, mp := range m.Parts {
				switch mp.Type {
				case "text":
					parts = append(parts, geminiPart{Text: mp.Text})
				case "image_url":
					if mp.ImageURL != nil {
						mime, data := parseDataURI(mp.ImageURL.URL)
						if data != "" {
							parts = append(parts, geminiPart{
								InlineData: &geminiInlineData{MIMEType: mime, Data: data},
							})
						}
					}
				case "input_audio":
					if mp.InputAudio != nil {
						parts = append(parts, geminiPart{
							InlineData: &geminiInlineData{
								MIMEType: mp.InputAudio.Format,
								Data:     mp.InputAudio.Data,
							},
						})
					}
				case "inline_data":
					if mp.InlineData != nil {
						parts = append(parts, geminiPart{
							InlineData: &geminiInlineData{
								MIMEType: mp.InlineData.MIMEType,
								Data:     mp.InlineData.Data,
							},
						})
					}
				}
			}
		} else if role == "model" && len(m.ToolCalls) > 0 {
			// Assistant message with tool calls → functionCall parts.
			for _, tc := range m.ToolCalls {
				var args any
				json.Unmarshal([]byte(tc.Arguments), &args) //nolint:errcheck
				parts = append(parts, geminiPart{
					FunctionCall: &geminiFunctionCall{Name: tc.Name, Args: args},
				})
			}
			if m.Content != "" {
				parts = append(parts, geminiPart{Text: m.Content})
			}
		} else {
			parts = append(parts, geminiPart{Text: m.Content})
		}

		if len(parts) > 0 {
			contents = append(contents, geminiContent{Role: role, Parts: parts})
		}
	}
	return contents
}

// parseDataURI extracts MIME type and base64 data from "data:mime;base64,DATA".
func parseDataURI(uri string) (mime, data string) {
	// data:image/jpeg;base64,/9j/4AAQ...
	if !strings.HasPrefix(uri, "data:") {
		return "image/jpeg", uri // assume raw base64
	}
	rest := uri[5:] // after "data:"
	semicolon := strings.Index(rest, ";")
	if semicolon < 0 {
		return "image/jpeg", uri
	}
	mime = rest[:semicolon]
	comma := strings.Index(rest, ",")
	if comma < 0 {
		return mime, ""
	}
	return mime, rest[comma+1:]
}

func buildGeminiFuncDecls(tools []Tool) []geminiFuncDecl {
	decls := make([]geminiFuncDecl, 0, len(tools))
	for _, t := range tools {
		var params any
		if len(t.InputSchema) > 0 {
			params = json.RawMessage(t.InputSchema)
		} else {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		decls = append(decls, geminiFuncDecl{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		})
	}
	return decls
}
