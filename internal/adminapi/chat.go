package adminapi

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"net/http"
	"strings"
	"sync"
	"time"

	"telegram-agent/internal/llm"
	"telegram-agent/internal/store"
)

// defaultAdminChatID is used when no forward-auth user is present (local dev
// via bearer token). Negative so it never collides with real Telegram user IDs.
const defaultAdminChatID int64 = -1

// ChatAgent is the subset of agent.Agent the chat handler needs.
type ChatAgent interface {
	Process(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(string)) (string, error)
	ProcessStream(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(string), onChunk func(string)) (string, error)
	GetChatHistory(chatID int64) []llm.Message
	GetDisplayHistory(chatID int64, limit, offset int) []store.HistoryItem
	ClearChatHistory(chatID int64)
	PopLastUserTurn(chatID int64) (string, bool)
}

// SetAgent wires the agent so the Chat tab can process messages.
func (s *Server) SetAgent(a ChatAgent) { s.agent = a }

// chatIDFor returns the chat_id the web admin should read/write. Priority:
//  1. Telegram OwnerChatID when configured — unifies the web chat with the
//     owner's Telegram conversation so a message sent via bot shows up in
//     the admin UI and vice versa. This is the common single-user setup.
//  2. FNV-64 hash of X-authentik-username, mapped to the negative range,
//     for multi-admin deployments.
//  3. defaultAdminChatID (-1) for local dev / bearer-token auth.
func (s *Server) chatIDFor(r *http.Request) int64 {
	if s.cfgRef != nil && s.cfgRef.Telegram.OwnerChatID != 0 {
		return s.cfgRef.Telegram.OwnerChatID
	}
	user := strings.TrimSpace(r.Header.Get(s.cfg.ForwardAuthHeader))
	if user == "" {
		return defaultAdminChatID
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(user))
	return -int64(h.Sum64()&0x7FFFFFFFFFFFFFFF) - 2
}

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
	Role      string   // "user" | "assistant" | "break"
	Body      string   // message text or break reason
	ImageURLs []string // image_url parts rendered as <img> in the bubble
	Time      string   // HH:MM for bubble, date for break markers
	BreakDate string   // full "2006-01-02 15:04" for dividers
}

type chatData struct {
	ActiveTab  string
	History    []chatMsgView
	HasMore    bool
	NextOffset int
}

const (
	// displayHistoryLimit is how many rows the Chat page loads on open.
	displayHistoryLimit = 50
	// displayHistoryChunk is how many rows each "Load earlier" fetch returns.
	displayHistoryChunk = 50
)

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	data := chatData{ActiveTab: "chat"}
	if s.agent != nil {
		chatID := s.chatIDFor(r)
		// Fetch one extra row to detect whether older messages exist.
		raw := s.agent.GetDisplayHistory(chatID, displayHistoryLimit+1, 0)
		if len(raw) > displayHistoryLimit {
			data.HasMore = true
			data.NextOffset = displayHistoryLimit
			raw = raw[:displayHistoryLimit] // trim the probe row
		}
		for _, it := range raw {
			v := itemToView(it)
			if v == nil {
				continue
			}
			data.History = append(data.History, *v)
		}
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewChat, data); err != nil {
		s.logger.Error("chat render", "err", err)
	}
}

// handleChatHistory serves older messages for the "Load earlier" button.
// Returns JSON: {html, has_more, next_offset}.
func (s *Server) handleChatHistory(w http.ResponseWriter, r *http.Request) {
	if s.agent == nil {
		http.Error(w, "agent not configured", http.StatusServiceUnavailable)
		return
	}
	offset := 0
	if v := r.URL.Query().Get("offset"); v != "" {
		fmt.Sscanf(v, "%d", &offset)
	}
	if offset < 0 {
		offset = 0
	}
	chatID := s.chatIDFor(r)
	raw := s.agent.GetDisplayHistory(chatID, displayHistoryChunk+1, offset)
	hasMore := len(raw) > displayHistoryChunk
	if hasMore {
		raw = raw[:displayHistoryChunk]
	}

	var sb strings.Builder
	for _, it := range raw {
		v := itemToView(it)
		if v == nil {
			continue
		}
		if v.Role == "break" {
			sb.WriteString(`<div class="divider--labeled"><span class="divider__label">`)
			sb.WriteString(v.BreakDate)
			if v.Body != "" {
				sb.WriteString(" &middot; ")
				sb.WriteString(v.Body)
			}
			sb.WriteString(`</span></div>`)
		} else {
			role := v.Role
			if role == "assistant" {
				role = "bot"
			}
			sb.WriteString(`<div class="chat-msg chat-msg--`)
			sb.WriteString(role)
			sb.WriteString(`"><div class="chat-msg__meta">`)
			if v.Role == "user" {
				sb.WriteString("You")
			} else {
				sb.WriteString("Assistant")
			}
			if v.Time != "" {
				sb.WriteString(" &middot; ")
				sb.WriteString(v.Time)
			}
			sb.WriteString(`</div>`)
			if len(v.ImageURLs) > 0 {
				sb.WriteString(`<div class="chat-msg__body">`)
				for _, u := range v.ImageURLs {
					sb.WriteString(`<img src="`)
					sb.WriteString(u)
					sb.WriteString(`" alt="attachment">`)
				}
				if v.Body != "" {
					sb.WriteString(`<div>`)
					sb.WriteString(v.Body)
					sb.WriteString(`</div>`)
				}
				sb.WriteString(`</div>`)
			} else if v.Role == "assistant" {
				// data-md will be picked up by renderMarkdown() on the client.
				sb.WriteString(`<div class="chat-msg__body md" data-md="`)
				sb.WriteString(htmlEscapeAttr(v.Body))
				sb.WriteString(`"></div>`)
			} else {
				sb.WriteString(`<div class="chat-msg__body">`)
				sb.WriteString(v.Body)
				sb.WriteString(`</div>`)
			}
			sb.WriteString(`</div>`)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	_ = enc.Encode(map[string]any{
		"html":        sb.String(),
		"has_more":    hasMore,
		"next_offset": offset + displayHistoryChunk,
	})
}

func itemToView(it store.HistoryItem) *chatMsgView {
	if it.Role == "tool" {
		return nil
	}
	v := &chatMsgView{Role: it.Role, Body: it.Content, ImageURLs: it.ImageURLs}
	if it.Role == "break" {
		v.BreakDate = it.CreatedAt.Local().Format("2006-01-02 15:04")
		v.Body = humanizeBreakReason(it.Content)
		return v
	}
	if it.Role == "assistant" && v.Body == "" && len(v.ImageURLs) == 0 {
		return nil
	}
	if !it.CreatedAt.IsZero() {
		v.Time = it.CreatedAt.Local().Format("15:04")
	}
	return v
}

func htmlEscapeAttr(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, `"`, "&quot;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// handleChatStream is a Server-Sent Events endpoint that streams the
// agent's response back to the client. Event types:
//
//	token       — {"delta": "..."} incremental assistant text
//	tool_call   — {"name": "..."} a tool is being invoked
//	done        — {"model": "...", "tools": [...], "text": "..."} final reply
//	error       — {"message": "..."} unrecoverable failure
//
// The endpoint also emits ":ping" comment lines every 15 s to keep the
// connection alive through reverse proxies.
func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	// Allow up to 25 MB of request body so base64-encoded images fit.
	r.Body = http.MaxBytesReader(w, r.Body, 25*1024*1024)
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	text := strings.TrimSpace(r.FormValue("message"))
	images := r.Form["image"]
	if text == "" && len(images) == 0 {
		http.Error(w, "empty message", http.StatusBadRequest)
		return
	}
	if s.agent == nil {
		http.Error(w, "agent not configured", http.StatusServiceUnavailable)
		return
	}

	userMsg := buildUserMessage(text, images)
	chatID := s.chatIDFor(r)

	// Per-request routing role override from the UI dropdown.
	ctxRole := strings.TrimSpace(r.FormValue("role"))
	switch ctxRole {
	case "", "simple", "default", "complex":
		// valid (empty = auto-route)
	default:
		ctxRole = ""
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // disable nginx/traefik buffering
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()
	if ctxRole != "" {
		ctx = llm.WithForcedRole(ctx, ctxRole)
	}

	// All writes to w go through sendMu. ProcessStream calls callbacks on its
	// goroutine; the heartbeat goroutine ticks on another; serialising is
	// cheaper than a funnel channel for this low rate.
	var sendMu sync.Mutex
	write := func(event, data string) {
		sendMu.Lock()
		defer sendMu.Unlock()
		writeSSE(w, flusher, event, data)
	}

	// Heartbeat keeps the connection open through idle-timeout proxies.
	hbStop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-hbStop:
				return
			case <-ticker.C:
				sendMu.Lock()
				_, _ = fmt.Fprint(w, ": ping\n\n")
				flusher.Flush()
				sendMu.Unlock()
			}
		}
	}()
	defer close(hbStop)

	var toolsMu sync.Mutex
	var tools []string

	var lastLen int
	onChunk := func(accumulated string) {
		if len(accumulated) <= lastLen {
			return
		}
		delta := accumulated[lastLen:]
		lastLen = len(accumulated)
		b, _ := json.Marshal(map[string]string{"delta": delta})
		write("token", string(b))
	}
	onToolCall := func(name string) {
		toolsMu.Lock()
		tools = append(tools, name)
		toolsMu.Unlock()
		b, _ := json.Marshal(map[string]string{"name": name})
		write("tool_call", string(b))
		// Each tool_call resets the streaming buffer on the client (the agent
		// will re-run the LLM after executing tools, producing a new text
		// stream). Reset the server-side length tracker so we don't clip the
		// next assistant text.
		lastLen = 0
	}

	resp, err := s.agent.ProcessStream(ctx, chatID, userMsg, onToolCall, onChunk)

	if err != nil {
		b, _ := json.Marshal(map[string]string{"message": err.Error()})
		write("error", string(b))
		return
	}

	toolsMu.Lock()
	finalTools := append([]string(nil), tools...)
	toolsMu.Unlock()
	payload := map[string]any{
		"model": routedModel(s.router),
		"tools": finalTools,
		"text":  resp,
	}
	b, _ := json.Marshal(payload)
	write("done", string(b))
}

// handleChatPop drops the last user turn and returns the removed text. The
// UI uses this for Edit — the text goes back into the input field so the user
// can revise and resubmit. For Regenerate the UI calls this then immediately
// calls /chat/stream with the same text.
func (s *Server) handleChatPop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.agent == nil {
		http.Error(w, "agent not configured", http.StatusServiceUnavailable)
		return
	}
	text, ok := s.agent.PopLastUserTurn(s.chatIDFor(r))
	w.Header().Set("Content-Type", "application/json")
	payload := map[string]any{"ok": ok, "text": text}
	b, _ := json.Marshal(payload)
	_, _ = w.Write(b)
}

// handleChatClear inserts a session-break marker in the history (non-
// destructive — prior messages stay visible with a divider above the new
// session). Returns 204; the UI inserts the divider client-side.
func (s *Server) handleChatClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.agent != nil {
		s.agent.ClearChatHistory(s.chatIDFor(r))
	}
	w.WriteHeader(http.StatusNoContent)
}

// humanizeBreakReason turns the internal session-break reason string into a
// short label for the UI divider. Unknown reasons fall through verbatim so
// future break types remain visible without a code change.
func humanizeBreakReason(raw string) string {
	switch raw {
	case "CONTEXT_RESET":
		return "Cleared"
	case "IDLE_4H":
		return "4h idle"
	case "":
		return ""
	default:
		return raw
	}
}

// buildUserMessage packs text + any attached images into an llm.Message.
// When images are present we build a multimodal Parts payload; otherwise
// we stick with plain Content so non-vision providers see a simple string.
// Accepts data URIs ("data:image/...;base64,...") or raw http(s) URLs.
func buildUserMessage(text string, images []string) llm.Message {
	if len(images) == 0 {
		return llm.Message{Role: "user", Content: text}
	}
	parts := make([]llm.ContentPart, 0, len(images)+1)
	if text != "" {
		parts = append(parts, llm.ContentPart{Type: "text", Text: text})
	}
	for _, url := range images {
		url = strings.TrimSpace(url)
		if url == "" {
			continue
		}
		parts = append(parts, llm.ContentPart{
			Type:     "image_url",
			ImageURL: &llm.ImageURL{URL: url},
		})
	}
	return llm.Message{Role: "user", Parts: parts}
}

// writeSSE emits one SSE event. Multi-line data payloads are split into
// separate `data: ` lines per the SSE spec.
func writeSSE(w http.ResponseWriter, f http.Flusher, event, data string) {
	if event != "" {
		_, _ = fmt.Fprintf(w, "event: %s\n", event)
	}
	for _, line := range strings.Split(data, "\n") {
		_, _ = fmt.Fprintf(w, "data: %s\n", line)
	}
	_, _ = fmt.Fprint(w, "\n")
	f.Flush()
}
