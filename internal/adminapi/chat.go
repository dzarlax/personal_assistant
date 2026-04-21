package adminapi

import (
	"context"
	"html"
	"net/http"
	"strings"
	"time"

	"telegram-agent/internal/llm"
)

// adminChatID is the dedicated chat ID for the admin web UI — negative so it
// never collides with real Telegram user IDs.
const adminChatID int64 = -1

// ChatAgent is the subset of agent.Agent the chat handler needs.
type ChatAgent interface {
	Process(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(string)) (string, error)
	GetChatHistory(chatID int64) []llm.Message
	ClearChatHistory(chatID int64)
}

// SetAgent wires the agent so the Chat tab can process messages.
func (s *Server) SetAgent(a ChatAgent) { s.agent = a }

// routedModel returns the actual model ID used on the last router call,
// falling back to the slot name when the provider doesn't expose CurrentModel.
func routedModel(r interface {
	LastRouted() string
	Provider(name string) (llm.Provider, bool)
}) string {
	slot := r.LastRouted()
	if p, ok := r.Provider(slot); ok {
		if cp, ok := p.(llm.ConfigurableProvider); ok {
			if m := cp.CurrentModel(); m != "" {
				return m
			}
		}
	}
	return slot
}

type chatMsgView struct {
	Role string // "user" or "assistant"
	Body string
}

type chatData struct {
	ActiveTab string
	History   []chatMsgView
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	data := chatData{ActiveTab: "chat"}
	if s.agent != nil {
		for _, m := range s.agent.GetChatHistory(adminChatID) {
			if m.Role != "user" && m.Role != "assistant" {
				continue
			}
			data.History = append(data.History, chatMsgView{Role: m.Role, Body: m.Content})
		}
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewChat, data); err != nil {
		s.logger.Error("chat render", "err", err)
	}
}

func (s *Server) handleChatSend(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	text := strings.TrimSpace(r.FormValue("message"))
	if text == "" {
		w.WriteHeader(http.StatusOK)
		return
	}
	if s.agent == nil {
		writeChatFragment(w, text, "", nil, "agent not configured", "")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()

	var tools []string
	resp, err := s.agent.Process(ctx, adminChatID, llm.Message{Role: "user", Content: text}, func(tool string) {
		tools = append(tools, tool)
	})

	errStr := ""
	if err != nil {
		errStr = err.Error()
	}

	model := routedModel(s.router)
	writeChatFragment(w, text, resp, tools, errStr, model)
}

func (s *Server) handleChatClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.agent != nil {
		s.agent.ClearChatHistory(adminChatID)
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(`<div id="chat-messages" class="chat-messages"></div>`)) //nolint:errcheck
}

func writeChatFragment(w http.ResponseWriter, userText, botText string, tools []string, errStr, model string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	ts := time.Now().Format("15:04")

	var sb strings.Builder

	// User bubble.
	sb.WriteString(`<div class="chat-msg chat-msg--user">`)
	sb.WriteString(`<div class="chat-msg__meta">You &middot; `)
	sb.WriteString(ts)
	sb.WriteString(`</div><div class="chat-msg__body">`)
	sb.WriteString(html.EscapeString(userText))
	sb.WriteString(`</div></div>`)

	// Assistant bubble.
	sb.WriteString(`<div class="chat-msg chat-msg--bot">`)
	// Meta line: model + tools.
	sb.WriteString(`<div class="chat-msg__meta">`)
	if model != "" {
		sb.WriteString(html.EscapeString(model))
		sb.WriteString(` &middot; `)
	}
	sb.WriteString(ts)
	if len(tools) > 0 {
		sb.WriteString(` &middot; <span class="chat-tools">`)
		for i, t := range tools {
			if i > 0 {
				sb.WriteString(`, `)
			}
			sb.WriteString(html.EscapeString(t))
		}
		sb.WriteString(`</span>`)
	}
	sb.WriteString(`</div>`)

	// Body: raw markdown stored in data-md, rendered by marked.js on the client.
	if errStr != "" {
		sb.WriteString(`<div class="chat-msg__body"><span class="chat-error">Error: `)
		sb.WriteString(html.EscapeString(errStr))
		sb.WriteString(`</span></div>`)
	} else {
		sb.WriteString(`<div class="chat-msg__body md" data-md="`)
		sb.WriteString(html.EscapeString(botText))
		sb.WriteString(`"></div>`)
	}

	sb.WriteString(`</div>`) // .chat-msg--bot
	sb.WriteString(`<div class="chat-scroll-sentinel"></div>`)

	w.Write([]byte(sb.String())) //nolint:errcheck
}
