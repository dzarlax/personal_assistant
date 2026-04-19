package llm

import "telegram-agent/internal/config"

// NewOpenRouter creates an OpenRouter provider. OpenRouter exposes many models
// (Anthropic, DeepSeek, Qwen, Llama, Gemini, ...) behind one OpenAI-compatible
// endpoint — the specific model is selected via cfg.Model (e.g.
// "deepseek/deepseek-chat-v3.1").
//
// Beyond the base OpenAI-compatible surface, this wrapper enables three
// OpenRouter-specific features on every request:
//
//  1. App attribution (HTTP-Referer, X-Title) — improves rate-limit standing
//     and surfaces our usage on OpenRouter's leaderboards.
//  2. usage.include — returns token counts so they can be logged upstream.
//  3. provider.require_parameters + allow_fallbacks — only routes to
//     upstream providers that support the request's parameters (e.g. tool
//     calling), and allows OpenRouter to transparently retry a different
//     upstream if the first one fails.
func NewOpenRouter(cfg config.ModelConfig) (*openAICompatProvider, error) {
	p, err := newOpenAICompat(cfg, "https://openrouter.ai/api/v1", "openrouter", cfg.Vision)
	if err != nil {
		return nil, err
	}
	p.extraHeaders = map[string]string{
		"HTTP-Referer": "https://github.com/dzarlax/personal_assistant",
		"X-Title":      "personal-assistant",
	}
	p.extraBodyFields = map[string]any{
		"usage": map[string]any{"include": true},
		"provider": map[string]any{
			"require_parameters": true,
			"allow_fallbacks":    true,
		},
	}
	return p, nil
}
