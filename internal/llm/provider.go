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
	Type       string      `json:"type"` // "text", "image_url", "input_audio", "inline_data"
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	InputAudio *InputAudio `json:"input_audio,omitempty"`
	InlineData *InlineData `json:"inline_data,omitempty"` // PDF, audio, video — native Gemini only
}

type ImageURL struct {
	URL string `json:"url"` // "data:image/jpeg;base64,..."
}

type InputAudio struct {
	Data   string `json:"data"`   // base64-encoded audio
	Format string `json:"format"` // "ogg", "mp3", "wav"
}

// InlineData holds base64-encoded file content with its MIME type.
// Supported by native Gemini provider for PDFs, audio, video, etc.
type InlineData struct {
	MIMEType string `json:"mime_type"` // e.g. "application/pdf", "audio/ogg"
	Data     string `json:"data"`      // base64-encoded
}

type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type ToolCall struct {
	ID               string `json:"id"`
	Name             string `json:"name"`
	Arguments        string `json:"arguments"` // JSON string
	ThoughtSignature string `json:"thought_signature,omitempty"` // Gemini: required when replaying function calls in history
}

type Response struct {
	Content   string
	ToolCalls []ToolCall
}

type Provider interface {
	Chat(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (Response, error)
	Name() string
}

// StreamChunk represents a single piece of a streaming response.
type StreamChunk struct {
	Delta     string     // incremental text content
	ToolCalls []ToolCall // populated only in the final Done chunk if the LLM returned tool calls
	Done      bool       // true when the stream is finished
	Err       error      // non-nil if the stream encountered an error
}

// StreamProvider is an optional interface for providers that support streaming.
// Providers that do not implement this fall back to synchronous Chat().
type StreamProvider interface {
	ChatStream(ctx context.Context, messages []Message, systemPrompt string, tools []Tool) (<-chan StreamChunk, error)
}

// VisionProvider is an optional interface for providers that support image input.
// When the active provider (override or primary) implements this, image messages
// are sent directly to it instead of being routed to the dedicated multimodal model.
type VisionProvider interface {
	SupportsVision() bool
}
