package llm

import "telegram-agent/internal/config"

func NewGemini(cfg config.ModelConfig) (*openAICompatProvider, error) {
	return newOpenAICompat(cfg, "", "gemini")
}
