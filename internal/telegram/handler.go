package telegram

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	tgbotapi "github.com/go-telegram-bot-api/telegram-bot-api/v5"

	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

const maxMessageLen = 4096

type Handler struct {
	bot     *tgbotapi.BotAPI
	agent   *agent.Agent
	allowed map[int64]bool
	ownerID int64
	logger  *slog.Logger
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
		bot:     bot,
		agent:   ag,
		allowed: allowed,
		ownerID: cfg.OwnerChatID,
		logger:  logger,
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
			go h.handleUpdate(update)
		}
	}
}

func (h *Handler) handleUpdate(update tgbotapi.Update) {
	if update.Message == nil {
		return
	}
	msg := update.Message

	if msg.From == nil {
		return
	}

	chatID := msg.Chat.ID

	// Access control: chat must be in allowlist AND sender must be the owner
	if !h.allowed[chatID] || msg.From.ID != h.ownerID {
		h.logger.Warn("unauthorized access attempt",
			"chat_id", chatID,
			"user_id", msg.From.ID,
			"username", msg.From.UserName,
		)
		h.notifyOwner(msg)
		return
	}

	if msg.IsCommand() {
		h.handleCommand(msg)
		return
	}

	if msg.Photo != nil {
		h.handlePhoto(msg)
		return
	}

	if msg.Text != "" {
		h.handleText(msg)
	}
}

func (h *Handler) handleCommand(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID

	switch msg.Command() {
	case "start":
		h.send(chatID, fmt.Sprintf(
			"Привет\\! Я твой персональный AI\\-ассистент\\.\n\n"+
				"Модель: `%s`\n\n"+
				"/clear — сбросить контекст\n"+
				"/compact — сжать историю\n"+
				"/model \\[default\\|reasoner\\] — переключить модель\n"+
				"/tools — доступные инструменты\n"+
				"/help — справка",
			h.agent.ModelName(),
		))
	case "help":
		h.send(chatID, fmt.Sprintf(
			"*Команды:*\n\n"+
				"/clear — сбросить контекст разговора\n"+
				"/compact — сжать историю \\(суммаризация\\)\n"+
				"/model — показать текущую модель\n"+
				"/model default — переключить на основную\n"+
				"/model reasoner — переключить на рассуждения\n"+
				"/tools — список MCP\\-инструментов\n"+
				"/help — эта справка\n\n"+
				"*Модель:* `%s`\n"+
				"Ответы длиннее 4096 символов отправляются как `.md` файл\\.",
			h.agent.ModelName(),
		))
	case "clear":
		h.agent.ClearHistory(chatID)
		h.send(chatID, "Контекст сброшен\\.")
	case "compact":
		h.sendPlain(chatID, "Сжимаю историю...")
		if err := h.agent.Compact(context.Background(), chatID); err != nil {
			h.sendPlain(chatID, "Ошибка: "+err.Error())
		} else {
			h.send(chatID, "История сжата\\.")
		}
	case "model":
		arg := strings.TrimSpace(msg.CommandArguments())
		switch arg {
		case "":
			override := h.agent.ModelOverride()
			if override == "" {
				override = "default"
			}
			h.send(chatID, fmt.Sprintf("Текущая модель: `%s` \\(режим: %s\\)", h.agent.ModelName(), override))
		case "default", "reset":
			h.agent.SetModel("")
			h.send(chatID, fmt.Sprintf("Модель: `%s`", h.agent.ModelName()))
		case "reasoner":
			h.agent.SetModel("reasoner")
			h.send(chatID, fmt.Sprintf("Модель: `%s`", h.agent.ModelName()))
		default:
			h.send(chatID, "Доступные режимы: `default`, `reasoner`")
		}
	case "tools":
		h.handleToolsCommand(chatID)
	default:
		h.send(chatID, "Неизвестная команда\\. /help — справка\\.")
	}
}

func (h *Handler) handleText(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID

	typingCtx, stopTyping := context.WithCancel(context.Background())
	defer stopTyping()
	go h.sendTypingLoop(chatID, typingCtx)

	h.logger.Info("processing message", "chat_id", chatID, "len", len(msg.Text))

	// Track tool calls for live status and response footnote
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

	response, err := h.agent.Process(context.Background(), chatID, llm.Message{Role: "user", Content: msg.Text}, onToolCall)
	stopTyping()

	// Delete live status message
	if statusMsgID != 0 {
		h.bot.Request(tgbotapi.NewDeleteMessage(chatID, statusMsgID)) //nolint:errcheck
	}

	if err != nil {
		h.logger.Error("agent error", "err", err)
		h.sendPlain(chatID, "Произошла ошибка: "+err.Error())
		return
	}

	// Append tool footnote to response
	if len(toolsUsed) > 0 {
		response += "\n\n`⚙️ " + strings.Join(toolsUsed, " · ") + "`"
	}

	h.sendResponse(chatID, response)
}

func (h *Handler) handlePhoto(msg *tgbotapi.Message) {
	chatID := msg.Chat.ID

	// Pick the highest resolution photo
	photo := msg.Photo[len(msg.Photo)-1]
	data, err := h.downloadFile(photo.FileID)
	if err != nil {
		h.logger.Error("failed to download photo", "err", err)
		h.sendPlain(chatID, "Не удалось загрузить фото.")
		return
	}

	parts := []llm.ContentPart{
		{Type: "image_url", ImageURL: &llm.ImageURL{
			URL: "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data),
		}},
	}
	caption := msg.Caption
	if caption == "" {
		caption = "Что на этом изображении?"
	}
	parts = append([]llm.ContentPart{{Type: "text", Text: caption}}, parts...)

	typingCtx, stopTyping := context.WithCancel(context.Background())
	defer stopTyping()
	go h.sendTypingLoop(chatID, typingCtx)

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
			h.bot.Send(tgbotapi.NewEditMessageText(chatID, statusMsgID, text)) //nolint:errcheck
		}
	}

	response, err := h.agent.Process(context.Background(), chatID, llm.Message{Role: "user", Parts: parts}, onToolCall)
	stopTyping()

	if statusMsgID != 0 {
		h.bot.Request(tgbotapi.NewDeleteMessage(chatID, statusMsgID)) //nolint:errcheck
	}
	if err != nil {
		h.sendPlain(chatID, "Произошла ошибка: "+err.Error())
		return
	}
	if len(toolsUsed) > 0 {
		response += "\n\n`⚙️ " + strings.Join(toolsUsed, " · ") + "`"
	}
	h.sendResponse(chatID, response)
}

func (h *Handler) downloadFile(fileID string) ([]byte, error) {
	file, err := h.bot.GetFile(tgbotapi.FileConfig{FileID: fileID})
	if err != nil {
		return nil, err
	}
	url := file.Link(h.bot.Token)
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (h *Handler) sendResponse(chatID int64, text string) {
	if len(text) >= maxMessageLen {
		h.sendAsFile(chatID, text)
		return
	}

	htmlText := markdownToTelegramHTML(text)
	msg := tgbotapi.NewMessage(chatID, htmlText)
	msg.ParseMode = tgbotapi.ModeHTML
	if _, err := h.bot.Send(msg); err != nil {
		h.logger.Warn("html send failed, retrying as plain text", "err", err)
		msg.ParseMode = ""
		msg.Text = text
		if _, err := h.bot.Send(msg); err != nil {
			h.logger.Error("failed to send response", "chat_id", chatID, "err", err)
		}
	}
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
		h.sendPlain(chatID, text[:maxMessageLen-50]+"...\n\n_(ответ обрезан)_")
	}
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

func registerCommands(bot *tgbotapi.BotAPI) error {
	commands := []tgbotapi.BotCommand{
		{Command: "clear", Description: "Сбросить контекст разговора"},
		{Command: "compact", Description: "Сжать историю (суммаризация)"},
		{Command: "model", Description: "Показать / переключить модель"},
		{Command: "tools", Description: "Список подключённых MCP-инструментов"},
		{Command: "help", Description: "Справка"},
	}
	_, err := bot.Request(tgbotapi.NewSetMyCommands(commands...))
	return err
}

func (h *Handler) handleToolsCommand(chatID int64) {
	tools := h.agent.ListTools()
	if len(tools) == 0 {
		h.sendPlain(chatID, "MCP-инструменты не подключены.")
		return
	}
	// Group by server
	byServer := make(map[string][]string)
	order := make([]string, 0)
	for _, t := range tools {
		if _, exists := byServer[t.ServerName]; !exists {
			order = append(order, t.ServerName)
		}
		byServer[t.ServerName] = append(byServer[t.ServerName], t.Name)
	}
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Инструменты (%d):\n", len(tools)))
	for _, srv := range order {
		sb.WriteString(fmt.Sprintf("\n%s:\n", srv))
		for _, name := range byServer[srv] {
			sb.WriteString(fmt.Sprintf("  • %s\n", name))
		}
	}
	h.sendPlain(chatID, sb.String())
}

func (h *Handler) notifyOwner(msg *tgbotapi.Message) {
	if h.ownerID == 0 {
		return
	}
	text := fmt.Sprintf("Попытка доступа: @%s (chat_id: %d, user_id: %d)",
		msg.From.UserName, msg.Chat.ID, msg.From.ID)
	notification := tgbotapi.NewMessage(h.ownerID, text)
	if _, err := h.bot.Send(notification); err != nil {
		h.logger.Error("failed to notify owner", "err", err)
	}
}
