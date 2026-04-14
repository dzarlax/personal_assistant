package telegram

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"

	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

const (
	maxMessageLen         = 4096
	requestTimeout        = 5 * time.Minute
	forwardTTL            = 5 * time.Minute
	forwardEmbedTimeout   = 10 * time.Second
	batchTimeout          = 2 * time.Second
	maxImagesPerBatch     = 5
	downloadTimeout       = 30 * time.Second
	maxInputLen           = 50 * 1024 // 50 KB cap on incoming text
	forwardFilterMinSize  = 3         // only filter by relevance when more than this many forwards are buffered
	forwardSelectThresh   = 0.25      // min cosine similarity to include a buffered forward
	maxConcurrentUpdates  = 10        // limit concurrent goroutines processing updates
	maxDocumentSize       = 20 * 1024 * 1024 // 20 MB cap on document uploads
)

// supportedDocMIME lists MIME types accepted as inline documents for the LLM.
var supportedDocMIME = map[string]bool{
	"application/pdf":    true,
	"text/plain":         true,
	"text/csv":           true,
	"text/html":          true,
	"text/markdown":      true,
	"application/json":   true,
	"application/xml":    true,
}

func isSupportedDocument(mime string) bool {
	return supportedDocMIME[mime]
}

// forwardEntry is a single forwarded message with its pre-computed embedding.
// text already includes the "[Forwarded from ...]" header prefix.
type forwardEntry struct {
	text string
	emb  []float32
}

// forwardedContent holds buffered forwarded messages waiting for a user follow-up question.
type forwardedContent struct {
	entries   []forwardEntry
	parts     []llm.ContentPart
	expiresAt time.Time
}

// pendingBatch accumulates messages for a single chat during the debounce window.
type pendingBatch struct {
	msgs    []*tgbotapi.Message
	timer   *time.Timer
	version int // incremented on each new message to detect stale timer callbacks
}

type Handler struct {
	bot     *tgbotapi.BotAPI
	agent   *agent.Agent
	allowed map[int64]bool
	ownerID int64
	logger  *slog.Logger

	forwardMu  sync.Mutex
	forwardBuf map[int64]*forwardedContent

	batchMu sync.Mutex
	batches map[int64]*pendingBatch

	sem chan struct{} // concurrency limiter for handleUpdate goroutines
}

func NewHandler(cfg config.TelegramConfig, ag *agent.Agent, logger *slog.Logger) (*Handler, error) {
	bot, err := tgbotapi.NewBotAPI(cfg.BotToken)
	if err != nil {
		return nil, fmt.Errorf("telegram init: %w", err)
	}

	allowed := make(map[int64]bool, len(cfg.AllowedChatIDs))
	for _, id := range cfg.AllowedChatIDs {
		allowed[id] = true
	}

	logger.Info("telegram bot authorized", "username", bot.Self.UserName)

	if err := registerCommands(bot); err != nil {
		logger.Warn("failed to register bot commands", "err", err)
	}

	return &Handler{
		bot:        bot,
		agent:      ag,
		allowed:    allowed,
		ownerID:    cfg.OwnerChatID,
		logger:     logger,
		forwardBuf: make(map[int64]*forwardedContent),
		batches:    make(map[int64]*pendingBatch),
		sem:        make(chan struct{}, maxConcurrentUpdates),
	}, nil
}

func (h *Handler) Start(ctx context.Context) {
	u := tgbotapi.NewUpdate(0)
	u.Timeout = 60
	updates := h.bot.GetUpdatesChan(u)

	for {
		select {
		case <-ctx.Done():
			h.bot.StopReceivingUpdates()
			return
		case update, ok := <-updates:
			if !ok {
				return
			}
			h.sem <- struct{}{} // acquire slot; blocks if maxConcurrentUpdates reached
			go func() {
				defer func() { <-h.sem }() // release slot
				h.handleUpdate(update)
			}()
		}
	}
}

func (h *Handler) handleUpdate(update tgbotapi.Update) {
	if update.CallbackQuery != nil {
		h.handleCallbackQuery(update.CallbackQuery)
		return
	}

	if update.Message == nil {
		return
	}
	msg := update.Message

	if msg.From == nil {
		return
	}

	chatID := msg.Chat.ID

	if !h.allowed[chatID] || msg.From.ID != h.ownerID {
		h.logger.Warn("unauthorized access attempt",
			"chat_id", chatID,
			"user_id", msg.From.ID,
			"username", msg.From.UserName,
		)
		h.notifyOwner(msg)
		return
	}

	// Commands bypass batching — they are interactive and must respond immediately.
	if msg.IsCommand() {
		h.handleCommand(msg)
		return
	}

	h.queueMessage(msg)
}

// queueMessage adds a message to the per-chat debounce batch.
// The batch is flushed after batchTimeout of inactivity.
func (h *Handler) queueMessage(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID

	h.batchMu.Lock()
	b := h.batches[chatID]
	if b == nil {
		b = &pendingBatch{}
		h.batches[chatID] = b
	}
	b.msgs = append(b.msgs, msg)
	b.version++
	ver := b.version
	if b.timer != nil {
		b.timer.Stop()
	}
	b.timer = time.AfterFunc(batchTimeout, func() {
		h.processBatch(chatID, ver)
	})
	h.batchMu.Unlock()
}

// processBatch is called by the debounce timer. It verifies the version to
// avoid processing a batch that was superseded by a newer message.
func (h *Handler) processBatch(chatID int64, version int) {
	h.batchMu.Lock()
	b := h.batches[chatID]
	if b == nil || b.version != version {
		h.batchMu.Unlock()
		return
	}
	delete(h.batches, chatID)
	h.batchMu.Unlock()

	h.runBatch(chatID, b)
}

// Drain flushes all pending batches synchronously. Call after stopping updates
// to avoid losing messages queued but not yet fired by their timers.
func (h *Handler) Drain() {
	h.batchMu.Lock()
	pending := h.batches
	h.batches = make(map[int64]*pendingBatch)
	h.batchMu.Unlock()

	for chatID, b := range pending {
		if b.timer != nil {
			b.timer.Stop()
		}
		h.runBatch(chatID, b)
	}
}

// runBatch merges all accumulated messages and sends them to the LLM as one request.
func (h *Handler) runBatch(chatID int64, b *pendingBatch) {
	if b == nil || len(b.msgs) == 0 {
		return
	}

	var forwardTexts []string  // forwarded messages in this batch
	var questionTexts []string // regular (non-forwarded) messages in this batch
	var imageParts []llm.ContentPart
	var hasVoice bool

	for _, msg := range b.msgs {
		isForward := msg.ForwardDate != 0

		text := msg.Text
		if text == "" {
			text = msg.Caption
		}
		text = appendTextLinks(text, msg.Entities, msg.CaptionEntities)

		// Transcribe voice/audio messages to text.
		if msg.Voice != nil || msg.Audio != nil {
			hasVoice = true
			voiceText := h.transcribeVoice(chatID, msg)
			if voiceText != "" {
				if text != "" {
					text = text + "\n" + voiceText
				} else {
					text = voiceText
				}
			}
		}

		if isForward {
			header := buildForwardHeader(msg)
			entry := header
			if text != "" {
				entry = header + "\n" + text
			}
			forwardTexts = append(forwardTexts, entry)
		} else {
			// If this message is a reply, prepend the quoted original so the LLM has context.
			if msg.ReplyToMessage != nil {
				text = buildReplyQuote(msg.ReplyToMessage) + text
			}
			if text != "" {
				questionTexts = append(questionTexts, text)
			}
		}

		// Download photo from any message in the batch (forwarded or not)
		if msg.Photo != nil && len(imageParts) < maxImagesPerBatch {
			photo := msg.Photo[len(msg.Photo)-1]
			data, err := h.downloadFile(photo.FileID)
			if err != nil {
				h.logger.Error("failed to download photo in batch", "err", err)
				continue
			}
			imageParts = append(imageParts, llm.ContentPart{
				Type: "image_url",
				ImageURL: &llm.ImageURL{
					URL: "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data),
				},
			})
		}

		// Download documents (PDF, etc.) — native Gemini only
		if msg.Document != nil && isSupportedDocument(msg.Document.MimeType) {
			data, err := h.downloadFile(msg.Document.FileID)
			if err != nil {
				h.logger.Error("failed to download document", "err", err, "file", msg.Document.FileName)
				continue
			}
			if len(data) > maxDocumentSize {
				h.logger.Warn("document too large, skipping", "file", msg.Document.FileName, "size", len(data))
				continue
			}
			h.logger.Info("document attached", "file", msg.Document.FileName, "mime", msg.Document.MimeType, "size", len(data))
			imageParts = append(imageParts, llm.ContentPart{
				Type: "inline_data",
				InlineData: &llm.InlineData{
					MIMEType: msg.Document.MimeType,
					Data:     base64.StdEncoding.EncodeToString(data),
				},
			})
		}
	}

	// If only forwarded messages arrived (no regular user message), buffer and ack.
	if len(questionTexts) == 0 {
		h.bufferForwards(chatID, forwardTexts, imageParts)
		return
	}

	// Consume any previously buffered forwards (slow follow-up path).
	h.forwardMu.Lock()
	fwd := h.forwardBuf[chatID]
	if fwd != nil && time.Now().After(fwd.expiresAt) {
		fwd = nil
	}
	delete(h.forwardBuf, chatID)
	h.forwardMu.Unlock()

	var allTextParts []string

	if fwd != nil {
		// Embed the user question to select only relevant buffered forwards.
		questionText := strings.Join(questionTexts, "\n\n")
		embedCtx, cancel := context.WithTimeout(context.Background(), forwardEmbedTimeout)
		questionEmb, _ := h.agent.EmbedText(embedCtx, questionText)
		cancel()

		selected := selectForwards(fwd.entries, questionEmb)
		allTextParts = append(allTextParts, selected...)
		imageParts = append(fwd.parts, imageParts...)
	}

	allTextParts = append(allTextParts, forwardTexts...)
	allTextParts = append(allTextParts, questionTexts...)

	// Build the LLM message.
	combined := strings.Join(allTextParts, "\n\n")
	if len(combined) > maxInputLen {
		combined = combined[:maxInputLen]
	}
	var userMsg llm.Message

	if len(imageParts) > 0 {
		if combined == "" {
			combined = "What is in this image?"
		}
		parts := append([]llm.ContentPart{{Type: "text", Text: combined}}, imageParts...)
		userMsg = llm.Message{Role: "user", Parts: parts}
	} else {
		if combined == "" {
			return
		}
		userMsg = llm.Message{Role: "user", Content: combined}
	}

	h.executeMessage(chatID, userMsg, hasVoice)
}

// bufferForwards embeds each forwarded text and stores them in forwardBuf for
// later relevance-based filtering when the user's follow-up question arrives.
func (h *Handler) bufferForwards(chatID int64, texts []string, parts []llm.ContentPart) {
	embedCtx, cancel := context.WithTimeout(context.Background(), forwardEmbedTimeout)
	defer cancel()

	entries := make([]forwardEntry, 0, len(texts))
	for _, t := range texts {
		emb, _ := h.agent.EmbedText(embedCtx, t)
		entries = append(entries, forwardEntry{text: t, emb: emb})
	}

	h.forwardMu.Lock()
	h.forwardBuf[chatID] = &forwardedContent{
		entries:   entries,
		parts:     parts,
		expiresAt: time.Now().Add(forwardTTL),
	}
	h.forwardMu.Unlock()
	h.sendPlain(chatID, "✓ Received. Add your question or comment.")
}

// selectForwards returns the texts from entries most relevant to questionEmb.
// Falls back to all entries when embeddings are unavailable or entry count is small.
func selectForwards(entries []forwardEntry, questionEmb []float32) []string {
	if len(entries) == 0 {
		return nil
	}
	// With few entries or no question embedding, include everything.
	if len(questionEmb) == 0 || !forwardHasEmbs(entries) || len(entries) <= forwardFilterMinSize {
		texts := make([]string, len(entries))
		for i, e := range entries {
			texts[i] = e.text
		}
		return texts
	}

	type scored struct {
		idx   int
		score float64
	}
	scores := make([]scored, len(entries))
	for i, e := range entries {
		s := 0.0
		if len(e.emb) > 0 {
			s = forwardCosine(questionEmb, e.emb)
		}
		scores[i] = scored{idx: i, score: s}
	}

	// Sort descending by score to find relevant ones; always keep at least 2.
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	var selected []int
	for i, s := range scores {
		if s.score >= forwardSelectThresh || i < 2 {
			selected = append(selected, s.idx)
		}
	}

	// Restore original order.
	sort.Ints(selected)

	texts := make([]string, 0, len(selected))
	for _, idx := range selected {
		texts = append(texts, entries[idx].text)
	}
	return texts
}

func forwardHasEmbs(entries []forwardEntry) bool {
	for _, e := range entries {
		if len(e.emb) > 0 {
			return true
		}
	}
	return false
}

func forwardCosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

const transcribeTimeout = 30 * time.Second

// transcribeVoice downloads a voice/audio message and transcribes it to text via the LLM.
func (h *Handler) transcribeVoice(chatID int64, msg *tgbotapi.Message) string {
	var fileID string
	if msg.Voice != nil {
		fileID = msg.Voice.FileID
	} else if msg.Audio != nil {
		fileID = msg.Audio.FileID
	}
	if fileID == "" {
		return ""
	}

	data, err := h.downloadFile(fileID)
	if err != nil {
		h.logger.Error("failed to download voice", "chat_id", chatID, "err", err)
		return ""
	}

	ctx, cancel := context.WithTimeout(context.Background(), transcribeTimeout)
	defer cancel()

	text, err := h.agent.TranscribeAudio(ctx, data, "audio/ogg")
	if err != nil {
		h.logger.Error("transcription failed", "chat_id", chatID, "err", err)
		return ""
	}

	h.logger.Info("voice transcribed", "chat_id", chatID, "text_len", len(text))
	return text
}

// executeMessage sends a prepared LLM message and streams the response back to Telegram.
// If voiceReply is true and TTS is enabled, the response is also sent as a voice message.
func (h *Handler) executeMessage(chatID int64, userMsg llm.Message, voiceReply bool) {
	typingCtx, stopTyping := context.WithCancel(context.Background())
	defer stopTyping()
	go h.sendTypingLoop(chatID, typingCtx)

	h.logger.Info("processing message", "chat_id", chatID, "has_parts", len(userMsg.Parts) > 0)

	var toolsUsed []string
	var statusMsgID int

	onToolCall := func(toolName string) {
		toolsUsed = append(toolsUsed, toolName)
		text := "⚙️ " + strings.Join(toolsUsed, " → ")
		if statusMsgID == 0 {
			m, err := h.bot.Send(tgbotapi.NewMessage(chatID, text))
			if err == nil {
				statusMsgID = m.MessageID
			}
		} else {
			edit := tgbotapi.NewEditMessageText(chatID, statusMsgID, text)
			h.bot.Send(edit) //nolint:errcheck
		}
	}

	reqCtx, cancelReq := context.WithTimeout(context.Background(), requestTimeout)
	defer cancelReq()

	var response string
	var err error

	if h.agent.SupportsStreaming() {
		// Streaming path: update message in real-time via editMessageText.
		var streamMsgID int
		var lastEdit time.Time
		const editInterval = 800 * time.Millisecond
		var streamStopped bool

		onChunk := func(accumulated string) {
			if streamStopped {
				return
			}
			// Stop live editing if text gets too long for a single message.
			if len(accumulated) > maxMessageLen-200 {
				streamStopped = true
				return
			}
			now := time.Now()
			if streamMsgID == 0 {
				// Send initial message with first chunk.
				msg := tgbotapi.NewMessage(chatID, accumulated+"▍")
				m, sendErr := h.bot.Send(msg)
				if sendErr == nil {
					streamMsgID = m.MessageID
					lastEdit = now
					stopTyping() // no need for typing indicator during streaming
				}
				return
			}
			if now.Sub(lastEdit) < editInterval {
				return // rate-limit edits
			}
			edit := tgbotapi.NewEditMessageText(chatID, streamMsgID, accumulated+"▍")
			if _, editErr := h.bot.Send(edit); editErr == nil {
				lastEdit = now
			}
		}

		response, err = h.agent.ProcessStream(reqCtx, chatID, userMsg, onToolCall, onChunk)
		stopTyping()

		// Clean up streaming message — replace with final formatted response.
		if streamMsgID != 0 && err == nil {
			if statusMsgID != 0 {
				h.bot.Request(tgbotapi.NewDeleteMessage(chatID, statusMsgID)) //nolint:errcheck
				statusMsgID = 0
			}
			var suffixParts []string
			if model := h.agent.LastRouted(); model != "" {
				suffixParts = append(suffixParts, model)
			}
			if len(toolsUsed) > 0 {
				suffixParts = append(suffixParts, strings.Join(toolsUsed, " · "))
			}
			suffix := ""
			if len(suffixParts) > 0 {
				suffix = "\n\n`⚙️ " + strings.Join(suffixParts, " · ") + "`"
			}
			finalText := response + suffix
			htmlText := markdownToTelegramHTML(finalText)

			if len(htmlText) < maxMessageLen {
				edit := tgbotapi.NewEditMessageText(chatID, streamMsgID, htmlText)
				edit.ParseMode = tgbotapi.ModeHTML
				if _, editErr := h.bot.Send(edit); editErr == nil {
					h.logger.Info("response sent", "chat_id", chatID, "len", len(response), "mode", "stream")
					if statusMsgID != 0 {
						h.bot.Request(tgbotapi.NewDeleteMessage(chatID, statusMsgID)) //nolint:errcheck
					}
					if voiceReply && h.agent.TTSEnabled() {
						go h.sendVoiceReply(chatID, response)
					}
					return
				}
			}
			// HTML too long or edit failed — delete streaming msg and fall through to sendResponse.
			h.bot.Request(tgbotapi.NewDeleteMessage(chatID, streamMsgID)) //nolint:errcheck
		}
	} else {
		// Non-streaming path (Claude Bridge, etc.)
		response, err = h.agent.Process(reqCtx, chatID, userMsg, onToolCall)
		stopTyping()
	}

	if statusMsgID != 0 {
		h.bot.Request(tgbotapi.NewDeleteMessage(chatID, statusMsgID)) //nolint:errcheck
	}

	if err != nil {
		h.logger.Error("agent error", "err", err)
		h.sendPlain(chatID, "Error: "+err.Error())
		return
	}

	var suffixParts []string
	if model := h.agent.LastRouted(); model != "" {
		suffixParts = append(suffixParts, model)
	}
	if len(toolsUsed) > 0 {
		suffixParts = append(suffixParts, strings.Join(toolsUsed, " · "))
	}
	if len(suffixParts) > 0 {
		response += "\n\n`⚙️ " + strings.Join(suffixParts, " · ") + "`"
	}

	h.sendResponse(chatID, response)
	h.logger.Info("response sent", "chat_id", chatID, "len", len(response), "mode", "fallback")

	// If the input was a voice message and TTS is enabled, also send a voice reply.
	if voiceReply && h.agent.TTSEnabled() {
		go h.sendVoiceReply(chatID, response)
	}
}

// appendTextLinks appends hidden URLs from text_link entities to the message text.
// Plain URLs are already visible in the text and need no special handling.
func appendTextLinks(text string, entitySets ...[]tgbotapi.MessageEntity) string {
	var links []string
	seen := make(map[string]bool)
	for _, entities := range entitySets {
		for _, e := range entities {
			if e.Type == "text_link" && e.URL != "" && !seen[e.URL] {
				seen[e.URL] = true
				links = append(links, e.URL)
			}
		}
	}
	if len(links) == 0 {
		return text
	}
	return text + "\n" + strings.Join(links, "\n")
}

// buildReplyQuote formats the replied-to message as a quoted prefix so the LLM
// understands what the user is responding to.
func buildReplyQuote(reply *tgbotapi.Message) string {
	text := reply.Text
	if text == "" {
		text = reply.Caption
	}
	if text == "" {
		// replied to a photo/sticker/etc with no text
		text = "[media]"
	}
	const maxQuoteLen = 300
	if len([]rune(text)) > maxQuoteLen {
		runes := []rune(text)
		text = string(runes[:maxQuoteLen]) + "…"
	}

	sender := "bot"
	if reply.From != nil && !reply.From.IsBot {
		sender = "you"
	}
	return fmt.Sprintf("[Replying to %s: \"%s\"]\n", sender, text)
}

// buildForwardHeader builds a "[Forwarded from ...]" label from a forwarded message.
func buildForwardHeader(msg *tgbotapi.Message) string {
	switch {
	case msg.ForwardFrom != nil:
		if msg.ForwardFrom.UserName != "" {
			return fmt.Sprintf("[Forwarded from @%s]", msg.ForwardFrom.UserName)
		}
		return fmt.Sprintf("[Forwarded from %s %s]", msg.ForwardFrom.FirstName, msg.ForwardFrom.LastName)
	case msg.ForwardFromChat != nil:
		if msg.ForwardFromChat.UserName != "" {
			return fmt.Sprintf("[Forwarded from @%s]", msg.ForwardFromChat.UserName)
		}
		return fmt.Sprintf("[Forwarded from %s]", msg.ForwardFromChat.Title)
	default:
		return "[Forwarded]"
	}
}

func (h *Handler) handleCommand(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID

	switch msg.Command() {
	case "start":
		h.send(chatID, fmt.Sprintf(
			"Hi\\! I'm your personal AI assistant\\.\n\n"+
				"Model: `%s`\n\n"+
				"/clear — reset context\n"+
				"/compact — compress history\n"+
				"/model list — available models\n"+
				"/tools — available tools\n"+
				"/help — help",
			h.agent.ModelName(),
		))
	case "help":
		h.send(chatID, fmt.Sprintf(
			"*Commands:*\n\n"+
				"/clear — reset conversation context\n"+
				"/compact — compress history \\(summarise\\)\n"+
				"/stats — history size, model, last compact\n"+
				"/model — show current model\n"+
				"/model list — available models\n"+
				"/model <name> — switch model\n"+
				"/model reset — back to auto\\-routing\n"+
				"/claude <question> — enter Claude mode\n"+
				"/exit — exit Claude mode\n"+
				"/tools — list MCP tools\n"+
				"/mcp update — reload MCP servers\n"+
				"/help — this help\n\n"+
				"*Model:* `%s`\n"+
				"Responses longer than 4096 chars are sent as a `.md` file\\.",
			h.agent.ModelName(),
		))
	case "clear":
		h.agent.ClearHistory(chatID)
		h.send(chatID, "Context cleared\\.")
	case "compact":
		h.sendPlain(chatID, "Compressing history...")
		if err := h.agent.Compact(context.Background(), chatID); err != nil {
			h.sendPlain(chatID, "Error: "+err.Error())
		} else {
			h.send(chatID, "History compressed\\.")
		}
	case "model":
		arg := strings.TrimSpace(msg.CommandArguments())
		switch arg {
		case "":
			override := h.agent.ModelOverride()
			mode := override
			if mode == "" {
				mode = "auto"
			}
			modelDisplay := h.agent.ModelName()
			// If override is a dynamic ollama cloud key, show it more clearly.
			if strings.HasPrefix(override, "ollama-cloud:") {
				modelDisplay = "ollama/" + strings.TrimPrefix(override, "ollama-cloud:")
			}
			h.send(chatID, fmt.Sprintf("Model: `%s` \\(override: %s\\)\n\n/model list — available models", escapeMarkdown(modelDisplay), escapeMarkdown(mode)))
		case "list":
			names := h.agent.ListModels()
			var sb strings.Builder
			sb.WriteString("*Available models:*\n")
			for _, n := range names {
				sb.WriteString("  `" + escapeMarkdown(n) + "`\n")
			}
			sb.WriteString("\nUse `/model <name>` to switch\\.")

			// Append Ollama Cloud section if an Ollama Cloud provider is configured.
			if _, _, _, ok := h.agent.OllamaCloudBaseConfig(); ok {
				cloudModels, err := llm.FetchOllamaCloudModels()
				if err != nil {
					h.logger.Warn("failed to fetch Ollama Cloud models", "err", err)
				}
				if len(cloudModels) > 0 {
					sb.WriteString("\n\n*Ollama Cloud:*")
					msg := tgbotapi.NewMessage(chatID, sb.String())
					msg.ParseMode = tgbotapi.ModeMarkdownV2
					var rows [][]tgbotapi.InlineKeyboardButton
					var row []tgbotapi.InlineKeyboardButton
					for i, m := range cloudModels {
						row = append(row, tgbotapi.NewInlineKeyboardButtonData(m, "model_ollama_cloud:"+m))
						if len(row) == 3 || i == len(cloudModels)-1 {
							rows = append(rows, row)
							row = nil
						}
					}
					kb := tgbotapi.NewInlineKeyboardMarkup(rows...)
					msg.ReplyMarkup = kb
					h.bot.Send(msg) //nolint:errcheck
					return
				}
			}
			h.send(chatID, sb.String())
		case "default", "reset":
			h.agent.SetModel("") //nolint:errcheck
			h.send(chatID, fmt.Sprintf("Model: `%s` \\(auto\\)", h.agent.ModelName()))
		default:
			if err := h.agent.SetModel(arg); err != nil {
				names := h.agent.ListModels()
				escaped := make([]string, len(names))
				for i, n := range names {
					escaped[i] = "`" + escapeMarkdown(n) + "`"
				}
				h.send(chatID, "Unknown model\\. Available: "+strings.Join(escaped, ", "))
			} else {
				h.send(chatID, fmt.Sprintf("Model: `%s`", escapeMarkdown(h.agent.ModelName())))
			}
		}
	case "claude":
		arg := strings.TrimSpace(msg.CommandArguments())
		if err := h.agent.SetModel("claude"); err != nil {
			h.send(chatID, "Claude Bridge is not configured\\.")
			return
		}
		h.send(chatID, "Claude mode on\\. Use /exit to return\\.")
		if arg != "" {
			// Forward the question as a regular message so the agent processes it.
			msg.Text = arg
			msg.Entities = nil
			h.queueMessage(msg)
		}
	case "exit":
		h.agent.ResetProviderSession("claude")
		h.agent.SetModel("") //nolint:errcheck
		h.send(chatID, fmt.Sprintf("Back to auto\\-routing\\. Model: `%s`", escapeMarkdown(h.agent.ModelName())))
	case "routing":
		cfg := h.agent.GetRouting()
		msg := tgbotapi.NewMessage(chatID, routingMenuText(cfg))
		msg.ParseMode = tgbotapi.ModeMarkdownV2
		msg.ReplyMarkup = routingMenuKeyboard(cfg)
		h.bot.Send(msg) //nolint:errcheck
	case "tools":
		h.handleToolsCommand(chatID)
	case "mcp":
		h.handleMCPCommand(chatID, msg.CommandArguments())
	case "stats":
		h.handleStatsCommand(chatID)
	default:
		h.send(chatID, "Unknown command\\. /help for help\\.")
	}
}

var downloadHTTPClient = &http.Client{Timeout: downloadTimeout}

func (h *Handler) downloadFile(fileID string) ([]byte, error) {
	file, err := h.bot.GetFile(tgbotapi.FileConfig{FileID: fileID})
	if err != nil {
		return nil, err
	}
	url := file.Link(h.bot.Token)
	resp, err := downloadHTTPClient.Get(url) //nolint:gosec
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (h *Handler) sendResponse(chatID int64, text string) {
	// Fast path: short message.
	htmlText := markdownToTelegramHTML(text)
	if len(htmlText) < maxMessageLen {
		h.sendHTMLWithFallback(chatID, text, htmlText)
		return
	}

	// Split markdown into chunks, convert each to HTML.
	// Use conservative limit — HTML tags expand the text ~30%.
	mdChunks := splitMessage(text, maxMessageLen*2/3)
	var htmlChunks []htmlChunk
	for _, md := range mdChunks {
		htm := markdownToTelegramHTML(md)
		if len(htm) < maxMessageLen {
			htmlChunks = append(htmlChunks, htmlChunk{raw: md, html: htm})
		} else {
			// HTML still too long — re-split the markdown piece more aggressively.
			subMD := splitMessage(md, maxMessageLen/3)
			for _, sm := range subMD {
				sh := markdownToTelegramHTML(sm)
				htmlChunks = append(htmlChunks, htmlChunk{raw: sm, html: sh})
			}
		}
	}

	allOK := true
	for i, ch := range htmlChunks {
		if len(ch.html) < maxMessageLen {
			if !h.sendHTMLMsg(chatID, ch.html) {
				// Retry as plain text.
				if !h.sendPlainMsg(chatID, ch.raw) {
					h.logger.Warn("chunk send failed", "chunk", i+1, "of", len(htmlChunks))
					allOK = false
					break
				}
			}
		} else {
			// Still too long after aggressive split — send plain, truncated.
			if !h.sendPlainMsg(chatID, ch.raw[:min(len(ch.raw), maxMessageLen-50)]) {
				allOK = false
				break
			}
		}
	}
	if allOK {
		return
	}

	// Last resort: send as file.
	h.sendAsFile(chatID, text)
}

type htmlChunk struct {
	raw  string
	html string
}

// sendHTMLWithFallback sends a single HTML message, falling back to plain text.
func (h *Handler) sendHTMLWithFallback(chatID int64, rawText, htmlText string) {
	if h.sendHTMLMsg(chatID, htmlText) {
		return
	}
	h.logger.Warn("html send failed, retrying as plain text")
	msg := tgbotapi.NewMessage(chatID, rawText)
	if _, err := h.bot.Send(msg); err != nil {
		h.logger.Error("failed to send response", "chat_id", chatID, "err", err)
	}
}

// sendHTMLMsg sends an HTML message. Returns true on success.
func (h *Handler) sendHTMLMsg(chatID int64, htmlText string) bool {
	msg := tgbotapi.NewMessage(chatID, htmlText)
	msg.ParseMode = tgbotapi.ModeHTML
	_, err := h.bot.Send(msg)
	return err == nil
}

// sendPlainMsg sends a plain-text message. Returns true on success.
func (h *Handler) sendPlainMsg(chatID int64, text string) bool {
	msg := tgbotapi.NewMessage(chatID, text)
	_, err := h.bot.Send(msg)
	return err == nil
}

// sendVoiceReply synthesizes text to speech and sends it as a Telegram voice message.
func (h *Handler) sendVoiceReply(chatID int64, text string) {
	defer func() {
		if r := recover(); r != nil {
			h.logger.Error("TTS panic", "err", r, "chat_id", chatID)
		}
	}()

	// Strip markdown formatting for cleaner speech output.
	plain := stripMarkdownForTTS(text)
	if plain == "" {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	audio, err := h.agent.SynthesizeSpeech(ctx, plain)
	if err != nil {
		h.logger.Warn("TTS synthesis failed", "err", err, "chat_id", chatID)
		return
	}
	if len(audio) == 0 {
		return
	}

	h.logger.Info("TTS synthesized", "chat_id", chatID, "audio_bytes", len(audio), "text_len", len(plain))

	file := tgbotapi.FileBytes{Name: "response.mp3", Bytes: audio}

	// Try voice message first; fall back to audio document if forbidden.
	voice := tgbotapi.NewVoice(chatID, file)
	if _, err := h.bot.Send(voice); err != nil {
		if strings.Contains(err.Error(), "VOICE_MESSAGES_FORBIDDEN") {
			audio := tgbotapi.NewAudio(chatID, file)
			if _, err2 := h.bot.Send(audio); err2 != nil {
				h.logger.Warn("failed to send audio file", "err", err2, "chat_id", chatID)
			} else {
				h.logger.Info("audio reply sent", "chat_id", chatID, "mode", "audio")
			}
		} else {
			h.logger.Warn("failed to send voice message", "err", err, "chat_id", chatID)
		}
	} else {
		h.logger.Info("voice reply sent", "chat_id", chatID, "mode", "voice")
	}
}

// stripMarkdownForTTS removes markdown formatting that sounds bad when spoken.
func stripMarkdownForTTS(s string) string {
	// Remove code blocks
	for {
		start := strings.Index(s, "```")
		if start == -1 {
			break
		}
		end := strings.Index(s[start+3:], "```")
		if end == -1 {
			s = s[:start]
			break
		}
		s = s[:start] + s[start+3+end+3:]
	}
	// Remove inline code, bold, italic markers
	r := strings.NewReplacer(
		"`", "",
		"**", "",
		"__", "",
		"*", "",
		"_", "",
		"#", "",
		"⚙️ ", "",
	)
	s = r.Replace(s)
	return strings.TrimSpace(s)
}

func (h *Handler) sendAsFile(chatID int64, text string) {
	caption := text
	if len(caption) > 200 {
		caption = caption[:200] + "..."
	}

	doc := tgbotapi.NewDocument(chatID, tgbotapi.FileBytes{
		Name:  "response.md",
		Bytes: []byte(text),
	})
	doc.Caption = caption

	if _, err := h.bot.Send(doc); err != nil {
		h.logger.Error("failed to send document", "err", err)
		h.sendPlain(chatID, text[:maxMessageLen-50]+"...\n\n_(response truncated)_")
	}
}

// splitMessage splits text into chunks of at most maxLen bytes,
// breaking at paragraph boundaries (\n\n), then line boundaries (\n).
func splitMessage(text string, maxLen int) []string {
	if len(text) <= maxLen {
		return []string{text}
	}

	var parts []string
	for len(text) > 0 {
		if len(text) <= maxLen {
			parts = append(parts, text)
			break
		}

		chunk := text[:maxLen]

		// Try to break at a paragraph boundary.
		if idx := strings.LastIndex(chunk, "\n\n"); idx > maxLen/4 {
			parts = append(parts, text[:idx])
			text = strings.TrimLeft(text[idx:], "\n")
			continue
		}

		// Try to break at a line boundary.
		if idx := strings.LastIndex(chunk, "\n"); idx > maxLen/4 {
			parts = append(parts, text[:idx])
			text = text[idx+1:]
			continue
		}

		// Hard break at maxLen.
		parts = append(parts, chunk)
		text = text[maxLen:]
	}
	return parts
}

// send sends a bot-generated message with MarkdownV2 (text must be pre-escaped).
func (h *Handler) send(chatID int64, text string) {
	msg := tgbotapi.NewMessage(chatID, text)
	msg.ParseMode = tgbotapi.ModeMarkdownV2
	if _, err := h.bot.Send(msg); err != nil {
		h.logger.Error("failed to send message", "chat_id", chatID, "err", err)
	}
}

// sendPlain sends a message without any markdown parsing.
func (h *Handler) sendPlain(chatID int64, text string) {
	msg := tgbotapi.NewMessage(chatID, text)
	if _, err := h.bot.Send(msg); err != nil {
		h.logger.Error("failed to send plain message", "chat_id", chatID, "err", err)
	}
}

func (h *Handler) sendTypingLoop(chatID int64, ctx context.Context) {
	h.bot.Send(tgbotapi.NewChatAction(chatID, tgbotapi.ChatTyping)) //nolint:errcheck
	ticker := time.NewTicker(4 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			h.bot.Send(tgbotapi.NewChatAction(chatID, tgbotapi.ChatTyping)) //nolint:errcheck
		}
	}
}

// --- Routing inline keyboard ---

func routingMenuText(cfg llm.RouterConfig) string {
	classifierStatus := "off"
	if cfg.ClassifierMinLen > 0 {
		classifierStatus = fmt.Sprintf("min %d chars", cfg.ClassifierMinLen)
	}
	return fmt.Sprintf(
		"⚙️ *Routing Configuration*\n\n"+
			"Local \\(1\\): `%s`\n"+
			"Primary \\(2\\): `%s`\n"+
			"Reasoner \\(3\\): `%s`\n"+
			"Fallback: `%s`\n"+
			"Classifier: `%s` \\(%s\\)\n"+
			"Multimodal: `%s`",
		escapeMarkdown(cfg.Local),
		escapeMarkdown(cfg.Primary),
		escapeMarkdown(cfg.Reasoner),
		escapeMarkdown(cfg.Fallback),
		escapeMarkdown(cfg.Classifier),
		classifierStatus,
		escapeMarkdown(cfg.Multimodal),
	)
}

func routingMenuKeyboard(cfg llm.RouterConfig) tgbotapi.InlineKeyboardMarkup {
	return tgbotapi.NewInlineKeyboardMarkup(
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonData("1️⃣ Local: "+cfg.Local, "rt:role:local"),
			tgbotapi.NewInlineKeyboardButtonData("2️⃣ Primary: "+cfg.Primary, "rt:role:primary"),
		),
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonData("3️⃣ Reasoner: "+cfg.Reasoner, "rt:role:reasoner"),
			tgbotapi.NewInlineKeyboardButtonData("✏️ Fallback: "+cfg.Fallback, "rt:role:fallback"),
		),
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonData("✏️ Classifier: "+cfg.Classifier, "rt:role:classifier"),
			tgbotapi.NewInlineKeyboardButtonData("✏️ Multimodal: "+cfg.Multimodal, "rt:role:multimodal"),
		),
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonData("✏️ Classifier threshold", "rt:min"),
		),
	)
}

func roleMenuKeyboard(role, current string, models []string) tgbotapi.InlineKeyboardMarkup {
	var rows [][]tgbotapi.InlineKeyboardButton
	var row []tgbotapi.InlineKeyboardButton
	for i, m := range models {
		label := m
		if m == current {
			label = "✓ " + m
		}
		row = append(row, tgbotapi.NewInlineKeyboardButtonData(label, "rt:set:"+role+":"+m))
		if len(row) == 2 || i == len(models)-1 {
			rows = append(rows, row)
			row = nil
		}
	}
	rows = append(rows, tgbotapi.NewInlineKeyboardRow(
		tgbotapi.NewInlineKeyboardButtonData("← Back", "rt:menu"),
	))
	return tgbotapi.NewInlineKeyboardMarkup(rows...)
}

func minLenMenuKeyboard(current int) tgbotapi.InlineKeyboardMarkup {
	options := []int{0, 50, 100, 200, 500}
	labels := []string{"Off (0)", "50", "100", "200", "500"}
	var row []tgbotapi.InlineKeyboardButton
	for i, v := range options {
		label := labels[i]
		if v == current {
			label = "✓ " + label
		}
		row = append(row, tgbotapi.NewInlineKeyboardButtonData(label, fmt.Sprintf("rt:min:%d", v)))
	}
	return tgbotapi.NewInlineKeyboardMarkup(
		row,
		tgbotapi.NewInlineKeyboardRow(
			tgbotapi.NewInlineKeyboardButtonData("← Back", "rt:menu"),
		),
	)
}

func (h *Handler) handleCallbackQuery(q *tgbotapi.CallbackQuery) {
	// Only owner can interact
	if q.From == nil || q.From.ID != h.ownerID {
		h.bot.Request(tgbotapi.NewCallback(q.ID, "Unauthorized")) //nolint:errcheck
		return
	}

	h.bot.Request(tgbotapi.NewCallback(q.ID, "")) //nolint:errcheck

	data := q.Data
	chatID := q.Message.Chat.ID
	msgID := q.Message.MessageID

	editText := func(text string, kb tgbotapi.InlineKeyboardMarkup) {
		edit := tgbotapi.NewEditMessageText(chatID, msgID, text)
		edit.ParseMode = tgbotapi.ModeMarkdownV2
		edit.ReplyMarkup = &kb
		h.bot.Send(edit) //nolint:errcheck
	}

	switch {
	case data == "rt:menu":
		cfg := h.agent.GetRouting()
		editText(routingMenuText(cfg), routingMenuKeyboard(cfg))

	case strings.HasPrefix(data, "rt:role:"):
		role := strings.TrimPrefix(data, "rt:role:")
		cfg := h.agent.GetRouting()
		current := roleValue(cfg, role)
		models := h.agent.ListModels()
		kb := roleMenuKeyboard(role, current, models)
		edit := tgbotapi.NewEditMessageText(chatID, msgID,
			fmt.Sprintf("⚙️ *Select model for* `%s`\\:", escapeMarkdown(role)))
		edit.ParseMode = tgbotapi.ModeMarkdownV2
		edit.ReplyMarkup = &kb
		h.bot.Send(edit) //nolint:errcheck

	case strings.HasPrefix(data, "rt:set:"):
		// rt:set:<role>:<model>
		rest := strings.TrimPrefix(data, "rt:set:")
		idx := strings.Index(rest, ":")
		if idx < 0 {
			return
		}
		role, model := rest[:idx], rest[idx+1:]
		h.logger.Info("routing change requested", "role", role, "model", model)
		if err := h.agent.SetRoutingRole(role, model); err != nil {
			h.logger.Warn("routing change failed", "role", role, "model", model, "err", err)
			h.bot.Request(tgbotapi.NewCallback(q.ID, "Error: "+err.Error())) //nolint:errcheck
			return
		}
		h.logger.Info("routing change applied", "role", role, "model", model)
		cfg := h.agent.GetRouting()
		editText(routingMenuText(cfg), routingMenuKeyboard(cfg))

	case data == "rt:min":
		cfg := h.agent.GetRouting()
		edit := tgbotapi.NewEditMessageText(chatID, msgID,
			"⚙️ *Classifier threshold*\n\nMinimum message length to run classifier \\(0 \\= disabled\\)\\:")
		edit.ParseMode = tgbotapi.ModeMarkdownV2
		kb := minLenMenuKeyboard(cfg.ClassifierMinLen)
		edit.ReplyMarkup = &kb
		h.bot.Send(edit) //nolint:errcheck

	case strings.HasPrefix(data, "rt:min:"):
		var n int
		fmt.Sscanf(strings.TrimPrefix(data, "rt:min:"), "%d", &n)
		h.agent.SetClassifierMinLen(n)
		cfg := h.agent.GetRouting()
		editText(routingMenuText(cfg), routingMenuKeyboard(cfg))

	case strings.HasPrefix(data, "model_ollama_cloud:"):
		modelName := strings.TrimPrefix(data, "model_ollama_cloud:")
		baseURL, apiKey, maxTokens, ok := h.agent.OllamaCloudBaseConfig()
		if !ok {
			h.sendPlain(chatID, "No Ollama Cloud provider configured.")
			return
		}
		p, err := llm.NewOllama(config.ModelConfig{
			Model:     modelName,
			BaseURL:   baseURL,
			APIKey:    apiKey,
			MaxTokens: maxTokens,
			Vision:    true,
		})
		if err != nil {
			h.logger.Warn("failed to create Ollama Cloud provider", "model", modelName, "err", err)
			h.sendPlain(chatID, "Error creating provider: "+err.Error())
			return
		}
		key := "ollama-cloud:" + modelName
		h.agent.AddModel(key, p)
		if err := h.agent.SetModel(key); err != nil {
			h.logger.Warn("failed to set Ollama Cloud model", "model", modelName, "err", err)
			h.sendPlain(chatID, "Error switching model: "+err.Error())
			return
		}
		h.logger.Info("switched to Ollama Cloud model", "model", modelName)
		h.send(chatID, fmt.Sprintf("Switched to ollama cloud: `%s`", escapeMarkdown(modelName)))
	}
}

// NotifyMissingRouting sends a Telegram message to the owner for each routing role
// that references a provider not present in the providers map.
func (h *Handler) NotifyMissingRouting() {
	if h.ownerID == 0 {
		return
	}
	cfg := h.agent.GetRouting()
	available := make(map[string]bool)
	for _, m := range h.agent.ListModels() {
		available[m] = true
	}

	roles := []struct{ name, model string }{
		{"local", cfg.Local},
		{"fallback", cfg.Fallback},
		{"reasoner", cfg.Reasoner},
		{"classifier", cfg.Classifier},
		{"multimodal", cfg.Multimodal},
	}
	for _, r := range roles {
		if r.model != "" && !available[r.model] {
			text := fmt.Sprintf(
				"⚠️ *Routing*: role `%s` — model `%s` is not available\\.\n\nSelect a replacement:",
				escapeMarkdown(r.name), escapeMarkdown(r.model),
			)
			msg := tgbotapi.NewMessage(h.ownerID, text)
			msg.ParseMode = tgbotapi.ModeMarkdownV2
			kb := roleMenuKeyboard(r.name, "", h.agent.ListModels())
			msg.ReplyMarkup = kb
			h.bot.Send(msg) //nolint:errcheck
		}
	}
}

// roleValue returns the current model name for a given routing role.
func roleValue(cfg llm.RouterConfig, role string) string {
	switch role {
	case "local":
		return cfg.Local
	case "primary":
		return cfg.Primary
	case "fallback":
		return cfg.Fallback
	case "reasoner":
		return cfg.Reasoner
	case "classifier":
		return cfg.Classifier
	case "multimodal":
		return cfg.Multimodal
	}
	return ""
}

func registerCommands(bot *tgbotapi.BotAPI) error {
	commands := []tgbotapi.BotCommand{
		{Command: "clear", Description: "Reset conversation context"},
		{Command: "compact", Description: "Compress history (summarise)"},
		{Command: "model", Description: "Show / switch model"},
		{Command: "claude", Description: "Enter Claude mode (heavy tasks)"},
		{Command: "exit", Description: "Exit Claude mode, back to auto-routing"},
		{Command: "routing", Description: "Configure routing (inline UI)"},
		{Command: "tools", Description: "List connected MCP tools"},
		{Command: "mcp", Description: "MCP management (update/reload)"},
		{Command: "stats", Description: "Show history size, model, last compact"},
		{Command: "help", Description: "Help"},
	}
	_, err := bot.Request(tgbotapi.NewSetMyCommands(commands...))
	return err
}

func (h *Handler) handleMCPCommand(chatID int64, args string) {
	switch strings.TrimSpace(args) {
	case "update", "reload":
		h.sendPlain(chatID, "Reloading MCP servers...")
		configs, err := config.LoadMCPServers("config/mcp.json")
		if err != nil {
			h.sendPlain(chatID, "Error loading mcp.json: "+err.Error())
			return
		}
		if len(configs) == 0 {
			h.sendPlain(chatID, "No MCP servers configured in mcp.json.")
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		toolCount, err := h.agent.ReloadMCP(ctx, configs)
		if err != nil {
			h.sendPlain(chatID, "Error: "+err.Error())
			return
		}
		h.sendPlain(chatID, fmt.Sprintf("MCP reloaded: %d servers, %d tools.", len(configs), toolCount))
	default:
		h.send(chatID,
			"*MCP commands:*\n\n"+
				"/mcp update — reload mcp\\.json and reconnect all servers")
	}
}

func (h *Handler) handleToolsCommand(chatID int64) {
	tools := h.agent.ListTools()
	if len(tools) == 0 {
		h.sendPlain(chatID, "No MCP tools connected.")
		return
	}
	byServer := make(map[string][]string)
	order := make([]string, 0)
	for _, t := range tools {
		if _, exists := byServer[t.ServerName]; !exists {
			order = append(order, t.ServerName)
		}
		byServer[t.ServerName] = append(byServer[t.ServerName], t.Name)
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Tools (%d):\n", len(tools)))
	for _, srv := range order {
		sb.WriteString(fmt.Sprintf("\n%s:\n", srv))
		for _, name := range byServer[srv] {
			sb.WriteString(fmt.Sprintf("  • %s\n", name))
		}
	}
	h.sendPlain(chatID, sb.String())
}

func (h *Handler) handleStatsCommand(chatID int64) {
	model := h.agent.ModelName()
	override := h.agent.ModelOverride()

	var sb strings.Builder
	sb.WriteString("*Stats*\n\n")
	sb.WriteString(fmt.Sprintf("Model: `%s`", escapeMarkdown(model)))
	if override != "" {
		sb.WriteString(fmt.Sprintf(" \\(override: `%s`\\)", escapeMarkdown(override)))
	} else {
		sb.WriteString(" \\(auto\\)")
	}
	sb.WriteString("\n")

	stats, ok := h.agent.GetStats(chatID)
	if !ok {
		sb.WriteString("\n_Stats not available \\(memory store\\)_")
		h.send(chatID, sb.String())
		return
	}

	sb.WriteString(fmt.Sprintf("\nHistory: *%d* messages / *%s*",
		stats.ActiveMessages,
		escapeMarkdown(formatBytes(stats.ActiveChars)),
	))

	if !stats.LastCompactAt.IsZero() {
		sb.WriteString(fmt.Sprintf("\nLast compact: %s", escapeMarkdown(stats.LastCompactAt.Format("2 Jan 2006, 15:04"))))
	} else {
		sb.WriteString("\nLast compact: _never_")
	}

	if !stats.LastMessageAt.IsZero() {
		sb.WriteString(fmt.Sprintf("\nLast message: %s", escapeMarkdown(stats.LastMessageAt.Format("2 Jan 2006, 15:04"))))
	}

	h.send(chatID, sb.String())
}

// formatBytes formats a byte count as a human-readable string.
func formatBytes(n int) string {
	switch {
	case n >= 1024*1024:
		return fmt.Sprintf("%.1f MB", float64(n)/1024/1024)
	case n >= 1024:
		return fmt.Sprintf("%.1f KB", float64(n)/1024)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// escapeMarkdown escapes special characters for Telegram MarkdownV2.
func escapeMarkdown(s string) string {
	replacer := strings.NewReplacer(
		"_", "\\_", "*", "\\*", "[", "\\[", "]", "\\]",
		"(", "\\(", ")", "\\)", "~", "\\~", "`", "\\`",
		">", "\\>", "#", "\\#", "+", "\\+", "-", "\\-",
		"=", "\\=", "|", "\\|", "{", "\\{", "}", "\\}",
		".", "\\.", "!", "\\!",
	)
	return replacer.Replace(s)
}

func (h *Handler) notifyOwner(msg *tgbotapi.Message) {
	if h.ownerID == 0 {
		return
	}
	text := fmt.Sprintf("Access attempt: @%s (chat_id: %d, user_id: %d)",
		msg.From.UserName, msg.Chat.ID, msg.From.ID)
	notification := tgbotapi.NewMessage(h.ownerID, text)
	if _, err := h.bot.Send(notification); err != nil {
		h.logger.Error("failed to notify owner", "err", err)
	}
}
