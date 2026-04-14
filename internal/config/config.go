package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// expandEnvWithDefaults expands ${VAR} and ${VAR:-default} in s.
func expandEnvWithDefaults(s string) string {
	return os.Expand(s, func(key string) string {
		if name, def, ok := strings.Cut(key, ":-"); ok {
			if v := os.Getenv(name); v != "" {
				return v
			}
			return def
		}
		return os.Getenv(key)
	})
}

type Config struct {
	Telegram   TelegramConfig   `yaml:"telegram"`
	Models     ModelsConfig     `yaml:"models"`
	Routing    RoutingConfig    `yaml:"routing"`
	ToolFilter ToolFilterConfig `yaml:"tool_filter"`
	WebSearch  WebSearchConfig  `yaml:"web_search"`
	Filesystem FilesystemConfig `yaml:"filesystem"`
	TTS        TTSConfig        `yaml:"tts"`
	VoiceAPI   VoiceAPIConfig   `yaml:"voice_api"`
}

type WebSearchConfig struct {
	Enabled bool   `yaml:"enabled"`
	BaseURL string `yaml:"base_url"` // default: https://ollama.com
	APIKey  string `yaml:"api_key"`
}

type ToolFilterConfig struct {
	TopK int `yaml:"top_k"` // 0 = disabled
}

type FilesystemConfig struct {
	Enabled bool   `yaml:"enabled"`
	Root    string `yaml:"root"` // absolute path on the host/container
}

type TTSConfig struct {
	Enabled bool   `yaml:"enabled"`
	Voice   string `yaml:"voice"`  // e.g. "ru-RU-DmitryNeural"
	Rate    string `yaml:"rate"`   // e.g. "+0%", "+20%"
	Pitch   string `yaml:"pitch"`  // e.g. "+0Hz"
	Volume  string `yaml:"volume"` // e.g. "+0%"
}

type VoiceAPIConfig struct {
	Enabled bool   `yaml:"enabled"`
	Listen  string `yaml:"listen"`  // e.g. ":8086"
	Token   string `yaml:"token"`   // Bearer auth token
	ChatID  int64  `yaml:"chat_id"` // dedicated conversation chat_id
}

type MCPServerConfig struct {
	Type       string            `json:"type,omitempty"` // "http" — used by Claude Code, ignored by bot
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
	Vision    bool   `yaml:"vision"`    // true if the model supports image input
	NoThink   bool   `yaml:"no_think"`  // disable thinking mode (qwen3)
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
	OllamaLocal     ModelConfig `yaml:"ollama-local"`
	OllamaCloud     ModelConfig `yaml:"ollama-cloud"`
	Claude          ModelConfig `yaml:"claude"`
	Local           ModelConfig `yaml:"local"`
}


type RoutingConfig struct {
	Local               string `yaml:"local"`                 // level 1: simple tasks (local model)
	Default             string `yaml:"default"`               // level 2: moderate tasks (cloud model)
	Fallback            string `yaml:"fallback"`
	Multimodal          string `yaml:"multimodal"`
	Reasoner            string `yaml:"reasoner"`              // level 3: complex reasoning
	Classifier          string `yaml:"classifier"`              // model that rates complexity 1/2/3
	ClassifierTimeout   int    `yaml:"classifier_timeout"`     // seconds; default 15
	ClassifierPrompt    string `yaml:"classifier_prompt"`      // system prompt; has default
	CompactionModel     string `yaml:"compaction_model"`
	ClassifierMinLength int    `yaml:"classifier_min_length"` // 0 = always; <0 = disabled
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

	expanded := expandEnvWithDefaults(string(data))

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

	expanded := expandEnvWithDefaults(string(data))

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
