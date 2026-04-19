package llm

import (
	"testing"
)

func TestParseOpenRouterModels(t *testing.T) {
	body := []byte(`{
		"data": [
			{
				"id": "anthropic/claude-sonnet-4.5",
				"context_length": 200000,
				"architecture": {
					"input_modalities": ["text", "image", "file"],
					"output_modalities": ["text"]
				},
				"pricing": {
					"prompt":     "0.000003",
					"completion": "0.000015"
				},
				"supported_parameters": ["tools", "tool_choice", "reasoning"]
			},
			{
				"id": "google/gemini-2.0-flash-exp:free",
				"context_length": 1048576,
				"architecture": {
					"input_modalities": ["text", "image"],
					"output_modalities": ["text"]
				},
				"pricing": {
					"prompt":     "0",
					"completion": "0"
				},
				"supported_parameters": ["tools"]
			},
			{
				"id": "deepseek/deepseek-chat-v3.1",
				"context_length": 65536,
				"architecture": {
					"input_modalities": ["text"],
					"output_modalities": ["text"]
				},
				"pricing": {
					"prompt":     "0.00000027",
					"completion": "0.0000011"
				},
				"supported_parameters": ["tools"]
			}
		]
	}`)

	caps, err := parseOpenRouterModels(body)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(caps) != 3 {
		t.Fatalf("want 3 models, got %d", len(caps))
	}

	claude := caps["anthropic/claude-sonnet-4.5"]
	if !claude.Vision {
		t.Error("claude should have vision")
	}
	if !claude.Tools {
		t.Error("claude should support tools")
	}
	if !claude.Reasoning {
		t.Error("claude should support reasoning")
	}
	if claude.PromptPrice != 3.0 {
		t.Errorf("claude prompt price: want 3.0 (USD/M), got %v", claude.PromptPrice)
	}
	if claude.CompletionPrice != 15.0 {
		t.Errorf("claude completion price: want 15.0, got %v", claude.CompletionPrice)
	}
	if claude.ContextLength != 200000 {
		t.Errorf("claude context: want 200000, got %d", claude.ContextLength)
	}

	gemini := caps["google/gemini-2.0-flash-exp:free"]
	if !gemini.Free() {
		t.Error("gemini:free should be Free()")
	}
	if !gemini.Vision {
		t.Error("gemini should have vision")
	}
	if gemini.Reasoning {
		t.Error("gemini should not report reasoning (not in supported_parameters)")
	}

	deepseek := caps["deepseek/deepseek-chat-v3.1"]
	if deepseek.Vision {
		t.Error("deepseek chat should not have vision")
	}
	if !deepseek.Tools {
		t.Error("deepseek should support tools")
	}
	// 0.00000027 * 1M = 0.27
	if deepseek.PromptPrice < 0.269 || deepseek.PromptPrice > 0.271 {
		t.Errorf("deepseek prompt price: want ~0.27, got %v", deepseek.PromptPrice)
	}
}

func TestCapabilitiesFree(t *testing.T) {
	tests := []struct {
		name string
		caps Capabilities
		want bool
	}{
		{"explicit zero", Capabilities{}, true},
		{"prompt only paid", Capabilities{PromptPrice: 1.0}, false},
		{"completion only paid", Capabilities{CompletionPrice: 0.5}, false},
		{"both paid", Capabilities{PromptPrice: 1, CompletionPrice: 1}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.caps.Free(); got != tc.want {
				t.Errorf("Free() = %v, want %v", got, tc.want)
			}
		})
	}
}
