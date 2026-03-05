package llm

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/sashabaranov/go-openai"
	"telegram-agent/internal/config"
)

type openAICompatProvider struct {
	client    *openai.Client
	model     string
	maxTokens int
	provName  string
}

func newOpenAICompat(cfg config.ModelConfig, defaultBaseURL, providerName string) (*openAICompatProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%s: api_key is required", providerName)
	}
	ocfg := openai.DefaultConfig(cfg.APIKey)
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	ocfg.BaseURL = baseURL

	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = 4096
	}
	return &openAICompatProvider{
		client:    openai.NewClientWithConfig(ocfg),
		model:     cfg.Model,
		maxTokens: maxTokens,
		provName:  providerName,
	}, nil
}

func (p *openAICompatProvider) Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error) {
	msgs := buildMessages(messages, systemPrompt)
	req := openai.ChatCompletionRequest{
		Model:     p.model,
		Messages:  msgs,
		MaxTokens: p.maxTokens,
	}
	if len(tools) > 0 {
		req.Tools = buildTools(tools)
	}

	resp, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return Response{}, fmt.Errorf("%s: %w", p.provName, err)
	}
	if len(resp.Choices) == 0 {
		return Response{}, fmt.Errorf("%s: empty response", p.provName)
	}

	choice := resp.Choices[0].Message
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

func buildMessages(messages []Message, systemPrompt string) []openai.ChatCompletionMessage {
	msgs := make([]openai.ChatCompletionMessage, 0, len(messages)+1)
	if systemPrompt != "" {
		msgs = append(msgs, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		})
	}
	for _, m := range messages {
		msg := openai.ChatCompletionMessage{
			Role:       m.Role,
			ToolCallID: m.ToolCallID,
		}
		if len(m.Parts) > 0 {
			for _, p := range m.Parts {
				switch p.Type {
				case "text":
					msg.MultiContent = append(msg.MultiContent, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeText,
						Text: p.Text,
					})
				case "image_url":
					if p.ImageURL != nil {
						msg.MultiContent = append(msg.MultiContent, openai.ChatMessagePart{
							Type:     openai.ChatMessagePartTypeImageURL,
							ImageURL: &openai.ChatMessageImageURL{URL: p.ImageURL.URL},
						})
					}
				}
			}
		} else {
			msg.Content = m.Content
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tc.Name,
					Arguments: tc.Arguments,
				},
			})
		}
		msgs = append(msgs, msg)
	}
	return msgs
}


func buildTools(tools []Tool) []openai.Tool {
	result := make([]openai.Tool, 0, len(tools))
	for _, t := range tools {
		var params any
		if len(t.InputSchema) > 0 {
			params = json.RawMessage(t.InputSchema)
		} else {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		result = append(result, openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}
	return result
}
