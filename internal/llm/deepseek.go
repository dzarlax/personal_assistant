package llm

import "telegram-agent/internal/config"

func NewDeepSeek(cfg config.ModelConfig) (*openAICompatProvider, error) {
	return newOpenAICompat(cfg, "", "deepseek")
}
