package llm

import (
	"context"
	"encoding/json"
)

type Message struct {
	Role       string        `json:"role"`
	Content    string        `json:"content"`
	Parts      []ContentPart `json:"parts,omitempty"` // set for multimodal messages
	ToolCalls  []ToolCall    `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// ContentPart represents a single part of a multimodal message.
type ContentPart struct {
	Type       string      `json:"type"` // "text", "image_url", "input_audio"
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	InputAudio *InputAudio `json:"input_audio,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"` // "data:image/jpeg;base64,..."
}

type InputAudio struct {
	Data   string `json:"data"`   // base64-encoded audio
	Format string `json:"format"` // "ogg", "mp3", "wav"
}

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type ToolCall struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

type Response struct {
	Content   string
	ToolCalls []ToolCall
}

type Provider interface {
	Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error)
	Name() string
}
