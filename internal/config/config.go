package config

import (
	"encoding/json"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Telegram   TelegramConfig   `yaml:"telegram"`
	Models     ModelsConfig     `yaml:"models"`
	Routing    RoutingConfig    `yaml:"routing"`
	ToolFilter ToolFilterConfig `yaml:"tool_filter"`
	WebSearch  WebSearchConfig  `yaml:"web_search"`
}

type WebSearchConfig struct {
	Enabled bool   `yaml:"enabled"`
	BaseURL string `yaml:"base_url"` // default: https://ollama.com
	APIKey  string `yaml:"api_key"`
}

type ToolFilterConfig struct {
	TopK int `yaml:"top_k"` // 0 = disabled
}

type MCPServerConfig struct {
	URL        string            `json:"url"`
	Headers    map[string]string `json:"headers"`
	DenyTools  []string          `json:"denyTools"`
	AllowTools []string          `json:"allowTools"`
}

type TelegramConfig struct {
	BotToken       string  `yaml:"bot_token"`
	AllowedChatIDs []int64 `yaml:"allowed_chat_ids"`
	OwnerChatID    int64   `yaml:"owner_chat_id"`
}

type ModelConfig struct {
	Provider  string `yaml:"provider"`
	Model     string `yaml:"model"`
	APIKey    string `yaml:"api_key"`
	MaxTokens int    `yaml:"max_tokens"`
	BaseURL   string `yaml:"base_url"`
}

type ModelsConfig struct {
	DeepSeek        ModelConfig `yaml:"deepseek"`
	DeepSeekR1      ModelConfig `yaml:"deepseek-r1"`
	GeminiFlashLite ModelConfig `yaml:"gemini-flash-lite"`
	GeminiFlash     ModelConfig `yaml:"gemini-flash"`
	Embedding       ModelConfig `yaml:"embedding"`
	QwenFlash       ModelConfig `yaml:"qwen-flash"`
	QwenPlus        ModelConfig `yaml:"qwen3.5-plus"`
	QwenMax         ModelConfig `yaml:"qwen-max"`
	Ollama          ModelConfig `yaml:"ollama"`
}


type RoutingConfig struct {
	Default             string `yaml:"default"`
	Fallback            string `yaml:"fallback"`
	Multimodal          string `yaml:"multimodal"`
	Reasoner            string `yaml:"reasoner"`
	Classifier          string `yaml:"classifier"`            // model for reasoning classifier; empty = use primary
	CompactionModel     string `yaml:"compaction_model"`
	ClassifierMinLength int    `yaml:"classifier_min_length"` // 0 = disabled
}

// LoadMCPServers loads mcp.json in Claude Desktop format.
// Returns empty map if file doesn't exist (MCP is optional).
func LoadMCPServers(path string) (map[string]MCPServerConfig, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read mcp config: %w", err)
	}

	expanded := os.ExpandEnv(string(data))

	var wrapper struct {
		MCPServers map[string]MCPServerConfig `json:"mcpServers"`
	}
	if err := json.Unmarshal([]byte(expanded), &wrapper); err != nil {
		return nil, fmt.Errorf("parse mcp config: %w", err)
	}

	return wrapper.MCPServers, nil
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	expanded := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &cfg, nil
}

// Validate checks config values for consistency.
func (c *Config) Validate() error {
	if c.Telegram.BotToken == "" {
		return fmt.Errorf("telegram.bot_token is required")
	}
	if c.Routing.Default == "" {
		return fmt.Errorf("routing.default is required")
	}
	if c.ToolFilter.TopK < 0 {
		return fmt.Errorf("tool_filter.top_k must be >= 0, got %d", c.ToolFilter.TopK)
	}
	if c.Routing.ClassifierMinLength < 0 {
		return fmt.Errorf("routing.classifier_min_length must be >= 0, got %d", c.Routing.ClassifierMinLength)
	}
	return nil
}
