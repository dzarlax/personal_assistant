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
	Default    ModelConfig `yaml:"default"`
	Reasoner   ModelConfig `yaml:"reasoner"`
	FlashLite  ModelConfig `yaml:"flash_lite"`
	Multimodal ModelConfig `yaml:"multimodal"`
	Embedding  ModelConfig `yaml:"embedding"`
}

type RoutingConfig struct {
	Default             string `yaml:"default"`
	Fallback            string `yaml:"fallback"`
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

	return &cfg, nil
}
