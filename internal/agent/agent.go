package agent

import (
	"context"
	"fmt"
	"log/slog"

	"telegram-agent/internal/llm"
	"telegram-agent/internal/mcp"
	"telegram-agent/internal/store"
)

const maxToolIterations = 5

type Agent struct {
	router    *llm.Router
	store     store.Store
	mcp       *mcp.Client
	compacter *Compacter
	sysPrompt string
	logger    *slog.Logger
}

func New(router *llm.Router, s store.Store, mcpClient *mcp.Client, compacter *Compacter, sysPrompt string, logger *slog.Logger) *Agent {
	return &Agent{
		router:    router,
		store:     s,
		mcp:       mcpClient,
		compacter: compacter,
		sysPrompt: sysPrompt,
		logger:    logger,
	}
}

// Process runs the agentic loop. onToolCall is called before each tool execution (may be nil).
func (a *Agent) Process(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(toolName string)) (string, error) {
	a.store.AddMessage(chatID, userMsg)

	// Auto-compact if needed
	if a.compacter != nil && NeedsCompaction(a.store, chatID) {
		a.logger.Info("auto-compacting conversation", "chat_id", chatID)
		if err := a.compacter.Compact(ctx, chatID, a.store); err != nil {
			a.logger.Warn("auto compaction failed", "err", err)
		}
	}

	var tools []llm.Tool
	if a.mcp != nil {
		tools = a.mcp.LLMTools()
	}

	for i := 0; i < maxToolIterations; i++ {
		history := a.store.GetHistory(chatID)

		resp, err := a.router.Chat(ctx, history, a.sysPrompt, tools)
		if err != nil {
			return "", fmt.Errorf("llm: %w", err)
		}

		if len(resp.ToolCalls) == 0 {
			a.store.AddMessage(chatID, llm.Message{Role: "assistant", Content: resp.Content})
			return resp.Content, nil
		}

		a.store.AddMessage(chatID, llm.Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		for _, tc := range resp.ToolCalls {
			if onToolCall != nil {
				onToolCall(tc.Name)
			}
			a.logger.Info("tool call", "tool", tc.Name)
			result, err := a.mcp.CallTool(ctx, tc.Name, tc.Arguments)
			if err != nil {
				a.logger.Warn("tool call failed", "tool", tc.Name, "err", err)
				result = fmt.Sprintf("Error: %s", err.Error())
			}
			a.logger.Info("tool result", "tool", tc.Name, "result_len", len(result))
			a.store.AddMessage(chatID, llm.Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	return "", fmt.Errorf("exceeded maximum tool iterations (%d)", maxToolIterations)
}

func (a *Agent) ClearHistory(chatID int64) {
	a.store.ClearHistory(chatID)
}

func (a *Agent) Compact(ctx context.Context, chatID int64) error {
	if a.compacter == nil {
		return fmt.Errorf("compaction not available (requires SQLite store)")
	}
	return a.compacter.Compact(ctx, chatID, a.store)
}

func (a *Agent) SetModel(override string) {
	a.router.SetOverride(override)
}

func (a *Agent) ModelName() string {
	return a.router.Name()
}

func (a *Agent) ModelOverride() string {
	return a.router.GetOverride()
}

type ToolInfo struct {
	Name       string
	ServerName string
}

func (a *Agent) ListTools() []ToolInfo {
	if a.mcp == nil {
		return nil
	}
	raw := a.mcp.Tools()
	result := make([]ToolInfo, len(raw))
	for i, t := range raw {
		result[i] = ToolInfo{Name: t.Name, ServerName: t.ServerName}
	}
	return result
}
