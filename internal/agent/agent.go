package agent

import (
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
	"telegram-agent/internal/mcp"
	"telegram-agent/internal/store"
)

func encodeBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

const (
	maxToolIterations = 10
	semanticRecentN   = 10 // always include last N messages in current session
	semanticTopK      = 20 // up to K older turns selected by similarity within session

	crossSessionTopK      = 5     // max snippets from past sessions
	crossSessionMinScore  = 0.75  // cosine threshold for relevance
	crossSessionMaxChars  = 3000  // total budget for cross-session block in system prompt
	snippetUserMaxChars   = 200   // per-snippet user text truncation
	snippetBotMaxChars    = 300   // per-snippet assistant text truncation

	toolResultSummarizeThreshold = 2000 // summarize tool results longer than this (chars)
)

type Agent struct {
	router         *llm.Router
	store          store.Store
	mcp            *mcp.Client
	compacter      *Compacter
	cache          *ResponseCache
	sysPrompt      string
	logger         *slog.Logger
	webSearch      *WebSearchConfig  // nil = disabled
	webSearchCache *webSearchCache
	webFetch       *WebFetchConfig   // nil = disabled
	transcribe     *TranscribeConfig // nil = disabled
	filesystem     *FilesystemConfig // nil = disabled
	tts            *TTSConfig        // nil = disabled
}

func New(router *llm.Router, s store.Store, mcpClient *mcp.Client, compacter *Compacter, sysPrompt string, logger *slog.Logger) *Agent {
	return &Agent{
		router:         router,
		store:          s,
		mcp:            mcpClient,
		compacter:      compacter,
		cache:          newResponseCache(),
		webSearchCache: newWebSearchCache(),
		sysPrompt:      sysPrompt,
		logger:         logger,
	}
}

// TranscribeAudio transcribes audio data to text via the native Gemini API.
// Returns the transcribed text. The audio is not stored in conversation history.
func (a *Agent) TranscribeAudio(ctx context.Context, audioData []byte, mimeType string) (string, error) {
	if a.transcribe == nil {
		return "", fmt.Errorf("transcription not configured")
	}
	text, err := transcribeViaGemini(ctx, *a.transcribe, audioData, mimeType)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(text), nil
}

// EnableTranscription configures audio transcription via native Gemini API.
func (a *Agent) EnableTranscription(cfg TranscribeConfig) {
	a.transcribe = &cfg
}

// EnableWebSearch activates the Ollama web search tool.
func (a *Agent) EnableWebSearch(cfg WebSearchConfig) {
	a.webSearch = &cfg
}

// EnableWebFetch activates the built-in web_fetch tool (HTTP+readability,
// optional CDP fallback to a headless Chrome).
func (a *Agent) EnableWebFetch(cfg WebFetchConfig) {
	a.webFetch = &cfg
}

// EnableFilesystem activates built-in filesystem tools scoped to cfg.Root.
func (a *Agent) EnableFilesystem(cfg FilesystemConfig) {
	a.filesystem = &cfg
}

// EnableTTS activates text-to-speech via Edge TTS.
func (a *Agent) EnableTTS(cfg TTSConfig) {
	a.tts = &cfg
}

// SynthesizeSpeech converts text to OGG Opus audio. Returns nil if TTS is disabled.
func (a *Agent) SynthesizeSpeech(ctx context.Context, text string) ([]byte, error) {
	if a.tts == nil {
		return nil, fmt.Errorf("TTS not configured")
	}
	return a.tts.Synthesize(ctx, text)
}

// TTSEnabled returns true if text-to-speech is configured.
func (a *Agent) TTSEnabled() bool {
	return a.tts != nil
}

// Process runs the agentic loop. onToolCall is called before each tool execution (may be nil).
func (a *Agent) Process(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(toolName string)) (string, error) {
	tr := newRequestTrace()
	queryText := messageText(userMsg)

	// Store user message; embed it if semantic store + MCP embeddings are both available.
	endEmbed := tr.begin("embed")
	queryEmb := a.storeUserMessage(ctx, chatID, userMsg, queryText)
	endEmbed()

	// Auto-compact if needed
	if a.compacter != nil && NeedsCompaction(a.store, chatID) {
		a.logger.Info("auto-compacting conversation", "chat_id", chatID)
		endCompact := tr.begin("compact")
		if err := a.compacter.Compact(ctx, chatID, a.store); err != nil {
			a.logger.Warn("auto compaction failed", "err", err)
		}
		endCompact()
	}

	endToolsSel := tr.begin("tools_select")
	var tools []llm.Tool
	if a.mcp != nil {
		tools = a.mcp.LLMToolsForQuery(ctx, queryText)
	}
	if a.webSearch != nil {
		tools = append(tools, webSearchTool())
	}
	if a.webFetch != nil {
		tools = append(tools, webFetchTool())
	}
	if a.filesystem != nil {
		tools = append(tools, filesystemTools()...)
	}
	endToolsSel()

	endCross := tr.begin("cross_session")
	crossSessionCtx := a.buildCrossSessionContext(ctx, chatID, queryEmb)
	endCross()

	// Check response cache before calling the LLM.
	endCache := tr.begin("cache")
	cached, cacheHit := a.cache.Get(chatID, queryEmb)
	endCache()
	if cacheHit {
		a.logger.Info("cache hit", "chat_id", chatID)
		a.store.AddMessage(chatID, llm.Message{Role: "assistant", Content: cached})
		tr.log(a.logger, chatID, "cache_hit")
		return cached, nil
	}

	for i := 0; i < maxToolIterations; i++ {
		endHist := tr.begin(fmt.Sprintf("iter%d_history", i))
		history := a.getHistory(chatID, queryEmb)
		endHist()

		sysPrompt := "Current date and time: " + time.Now().Format("Monday, 2 January 2006, 15:04 MST") + "\n\n" + a.sysPrompt
		if crossSessionCtx != "" {
			sysPrompt += "\n\n" + crossSessionCtx
		}

		iterCtx, timing := llm.WithTimings(ctx)
		llmStart := time.Now()
		resp, err := a.router.Chat(iterCtx, history, sysPrompt, tools)
		llmTotal := time.Since(llmStart)
		if timing.ClassifyRan {
			tr.addSpan(fmt.Sprintf("iter%d_classify", i), timing.ClassifyDur)
			tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal-timing.ClassifyDur)
		} else {
			tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal)
		}
		if err != nil {
			tr.log(a.logger, chatID, "llm_error")
			return "", fmt.Errorf("llm: %w", err)
		}

		if len(resp.ToolCalls) == 0 {
			if resp.Content == "" {
				a.logger.Warn("empty response from LLM", "chat_id", chatID, "iteration", i)
			}
			a.store.AddMessage(chatID, llm.Message{Role: "assistant", Content: resp.Content})
			// Cache only pure direct responses (first iteration, no tool calls).
			if i == 0 {
				a.cache.Set(chatID, queryEmb, resp.Content)
			}
			tr.log(a.logger, chatID, "ok")
			return resp.Content, nil
		}

		a.store.AddMessage(chatID, llm.Message{
			Role:      "assistant",
			Content:   resp.Content,
			ToolCalls: resp.ToolCalls,
		})

		endTools := tr.begin(fmt.Sprintf("iter%d_tools", i))
		a.executeToolCalls(ctx, chatID, resp.ToolCalls, onToolCall)
		endTools()
	}

	tr.log(a.logger, chatID, "max_iterations")
	return "", fmt.Errorf("exceeded maximum tool iterations (%d)", maxToolIterations)
}

// ProcessStream is like Process but streams the final text response via onChunk.
// Tool-calling iterations remain synchronous. onChunk receives the accumulated text so far.
func (a *Agent) ProcessStream(ctx context.Context, chatID int64, userMsg llm.Message, onToolCall func(string), onChunk func(accumulated string)) (string, error) {
	tr := newRequestTrace()
	queryText := messageText(userMsg)

	endEmbed := tr.begin("embed")
	queryEmb := a.storeUserMessage(ctx, chatID, userMsg, queryText)
	endEmbed()

	if a.compacter != nil && NeedsCompaction(a.store, chatID) {
		a.logger.Info("auto-compacting conversation", "chat_id", chatID)
		endCompact := tr.begin("compact")
		if err := a.compacter.Compact(ctx, chatID, a.store); err != nil {
			a.logger.Warn("auto compaction failed", "err", err)
		}
		endCompact()
	}

	endToolsSel := tr.begin("tools_select")
	var tools []llm.Tool
	if a.mcp != nil {
		tools = a.mcp.LLMToolsForQuery(ctx, queryText)
	}
	if a.webSearch != nil {
		tools = append(tools, webSearchTool())
	}
	if a.webFetch != nil {
		tools = append(tools, webFetchTool())
	}
	if a.filesystem != nil {
		tools = append(tools, filesystemTools()...)
	}
	endToolsSel()

	endCross := tr.begin("cross_session")
	crossSessionCtx := a.buildCrossSessionContext(ctx, chatID, queryEmb)
	endCross()

	endCache := tr.begin("cache")
	cached, cacheHit := a.cache.Get(chatID, queryEmb)
	endCache()
	if cacheHit {
		a.logger.Info("cache hit", "chat_id", chatID)
		a.store.AddMessage(chatID, llm.Message{Role: "assistant", Content: cached})
		tr.log(a.logger, chatID, "cache_hit")
		return cached, nil
	}

	for i := 0; i < maxToolIterations; i++ {
		endHist := tr.begin(fmt.Sprintf("iter%d_history", i))
		history := a.getHistory(chatID, queryEmb)
		endHist()

		sysPrompt := "Current date and time: " + time.Now().Format("Monday, 2 January 2006, 15:04 MST") + "\n\n" + a.sysPrompt
		if crossSessionCtx != "" {
			sysPrompt += "\n\n" + crossSessionCtx
		}

		iterCtx, timing := llm.WithTimings(ctx)
		llmStart := time.Now()
		ch, err := a.router.ChatStream(iterCtx, history, sysPrompt, tools)
		if err != nil {
			llmTotal := time.Since(llmStart)
			if timing.ClassifyRan {
				tr.addSpan(fmt.Sprintf("iter%d_classify", i), timing.ClassifyDur)
				tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal-timing.ClassifyDur)
			} else {
				tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal)
			}
			tr.log(a.logger, chatID, "llm_error")
			return "", fmt.Errorf("llm: %w", err)
		}

		var accumulated string
		var toolCalls []llm.ToolCall
		hasToolCalls := false

		for chunk := range ch {
			if chunk.Err != nil {
				llmTotal := time.Since(llmStart)
				if timing.ClassifyRan {
					tr.addSpan(fmt.Sprintf("iter%d_classify", i), timing.ClassifyDur)
					tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal-timing.ClassifyDur)
				} else {
					tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal)
				}
				tr.log(a.logger, chatID, "stream_error")
				return "", fmt.Errorf("llm stream: %w", chunk.Err)
			}
			if chunk.Delta != "" {
				accumulated += chunk.Delta
				if !hasToolCalls && onChunk != nil {
					onChunk(accumulated)
				}
			}
			if chunk.Done {
				toolCalls = chunk.ToolCalls
				if len(toolCalls) > 0 {
					hasToolCalls = true
				}
			}
		}
		llmTotal := time.Since(llmStart)
		if timing.ClassifyRan {
			tr.addSpan(fmt.Sprintf("iter%d_classify", i), timing.ClassifyDur)
			tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal-timing.ClassifyDur)
		} else {
			tr.addSpan(fmt.Sprintf("iter%d_llm", i), llmTotal)
		}

		if len(toolCalls) == 0 {
			a.store.AddMessage(chatID, llm.Message{Role: "assistant", Content: accumulated})
			if i == 0 {
				a.cache.Set(chatID, queryEmb, accumulated)
			}
			tr.log(a.logger, chatID, "ok")
			return accumulated, nil
		}

		// Tool calls — process them (text deltas were suppressed by hasToolCalls or absent).
		a.store.AddMessage(chatID, llm.Message{
			Role:      "assistant",
			Content:   accumulated,
			ToolCalls: toolCalls,
		})

		endTools := tr.begin(fmt.Sprintf("iter%d_tools", i))
		a.executeToolCalls(ctx, chatID, toolCalls, onToolCall)
		endTools()
	}

	tr.log(a.logger, chatID, "max_iterations")
	return "", fmt.Errorf("exceeded maximum tool iterations (%d)", maxToolIterations)
}

// SupportsStreaming returns true if the current provider supports streaming.
func (a *Agent) SupportsStreaming() bool {
	return a.router.SupportsStreaming()
}

// executeToolCalls runs tool calls in parallel and stores results in order.
func (a *Agent) executeToolCalls(ctx context.Context, chatID int64, toolCalls []llm.ToolCall, onToolCall func(string)) {
	type toolResult struct {
		tc     llm.ToolCall
		result string
	}

	results := make([]toolResult, len(toolCalls))
	var wg sync.WaitGroup

	for i, tc := range toolCalls {
		if onToolCall != nil {
			onToolCall(tc.Name)
		}
		a.logger.Info("tool call", "tool", tc.Name)

		wg.Add(1)
		go func(idx int, tc llm.ToolCall) {
			defer wg.Done()
			result, err := a.callTool(ctx, tc.Name, tc.Arguments)
			if err != nil {
				a.logger.Warn("tool call failed", "tool", tc.Name, "err", err)
				result = fmt.Sprintf("Error: %s", err.Error())
			}
			if len(result) > toolResultSummarizeThreshold {
				if summarized, sErr := a.summarizeToolResult(ctx, tc.Name, result); sErr == nil {
					a.logger.Info("tool result summarized", "tool", tc.Name, "original_len", len(result), "summary_len", len(summarized))
					result = summarized
				}
			}
			a.logger.Info("tool result", "tool", tc.Name, "result_len", len(result))
			results[idx] = toolResult{tc: tc, result: result}
		}(i, tc)
	}

	wg.Wait()

	// Store results in original order.
	for _, r := range results {
		a.store.AddMessage(chatID, llm.Message{
			Role:       "tool",
			Content:    r.result,
			ToolCallID: r.tc.ID,
		})
	}
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

func (a *Agent) SetModel(override string) error {
	return a.router.SetOverride(override)
}

func (a *Agent) ModelName() string {
	return a.router.Name()
}

func (a *Agent) ModelOverride() string {
	return a.router.GetOverride()
}

func (a *Agent) LastRouted() string {
	return a.router.LastRouted()
}

func (a *Agent) ListModels() []string {
	return a.router.ProviderNames()
}

func (a *Agent) ResetProviderSession(name string) {
	a.router.ResetProviderSession(name)
}

func (a *Agent) GetRouting() llm.RouterConfig {
	return a.router.GetConfig()
}

func (a *Agent) SetRoutingRole(role, model string) error {
	return a.router.SetRole(role, model)
}

func (a *Agent) SetClassifierMinLen(n int) {
	a.router.SetClassifierMinLen(n)
}

// SetOllamaCloudModel changes the model on the Ollama Cloud provider and persists.
func (a *Agent) SetOllamaCloudModel(model string, vision bool) bool {
	return a.router.SetOllamaCloudModel(model, vision)
}

// OllamaCloudModelName returns the current Ollama Cloud model name, or ("", false).
func (a *Agent) OllamaCloudModelName() (string, bool) {
	return a.router.OllamaCloudModelName()
}

type ToolInfo struct {
	Name       string
	ServerName string
}

// GetStats returns per-chat statistics. Only available when using SQLite store.
func (a *Agent) GetStats(chatID int64) (store.ChatStats, bool) {
	cs, ok := a.store.(store.CompactableStore)
	if !ok {
		return store.ChatStats{}, false
	}
	return cs.GetStats(chatID), true
}

// EmbedText returns an embedding for text using the configured embedding model.
// Returns nil, nil when embeddings are not configured.
func (a *Agent) EmbedText(ctx context.Context, text string) ([]float32, error) {
	if a.mcp == nil {
		return nil, nil
	}
	return a.mcp.EmbedText(ctx, text)
}

// ReloadMCP re-reads the MCP config file, reconnects all servers, and re-discovers tools.
// Returns the number of tools available after reload.
func (a *Agent) ReloadMCP(ctx context.Context, configs map[string]config.MCPServerConfig) (int, error) {
	if a.mcp == nil {
		return 0, fmt.Errorf("MCP client not initialized")
	}
	return a.mcp.Reconnect(ctx, configs)
}

func (a *Agent) ListTools() []ToolInfo {
	var result []ToolInfo
	if a.mcp != nil {
		for _, t := range a.mcp.Tools() {
			result = append(result, ToolInfo{Name: t.Name, ServerName: t.ServerName})
		}
	}
	if a.webSearch != nil {
		serverName := a.webSearch.Provider
		if serverName == "" {
			serverName = webSearchProviderOllama
		}
		result = append(result, ToolInfo{Name: webSearchToolName, ServerName: serverName})
	}
	if a.webFetch != nil {
		result = append(result, ToolInfo{Name: webFetchToolName, ServerName: "web_fetch"})
	}
	if a.filesystem != nil {
		for _, t := range filesystemTools() {
			result = append(result, ToolInfo{Name: t.Name, ServerName: "filesystem"})
		}
	}
	return result
}

const toolSummarizePrompt = `Summarize this tool output concisely, preserving all key facts, numbers, and actionable data. Remove formatting noise, redundant fields, and verbose structure. Output only the summary.`

// summarizeToolResult condenses a large tool result to save tokens in conversation history.
func (a *Agent) summarizeToolResult(ctx context.Context, toolName, result string) (string, error) {
	msgs := []llm.Message{{
		Role:    "user",
		Content: fmt.Sprintf("Tool '%s' returned:\n\n%s", toolName, result),
	}}
	resp, err := a.router.Chat(ctx, msgs, toolSummarizePrompt, nil)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// callTool dispatches a tool call to the appropriate handler:
// built-in tools are handled directly; everything else goes to MCP.
func (a *Agent) callTool(ctx context.Context, name, argsJSON string) (string, error) {
	if name == webSearchToolName && a.webSearch != nil {
		return a.callWebSearchCached(ctx, argsJSON)
	}
	if name == webFetchToolName && a.webFetch != nil {
		return a.callWebFetchCached(ctx, argsJSON)
	}
	if a.filesystem != nil {
		switch name {
		case fsListFilesTool, fsReadFileTool, fsWriteFileTool, fsAppendFileTool, fsDeleteFileTool, fsSearchFilesTool:
			return callFilesystem(*a.filesystem, name, argsJSON)
		}
	}
	if a.mcp != nil {
		return a.mcp.CallTool(ctx, name, argsJSON)
	}
	return "", fmt.Errorf("unknown tool: %s", name)
}

// storeUserMessage saves the user message. If the store and MCP client both support
// embeddings, it also computes and stores the embedding for semantic history retrieval.
// Returns the embedding (may be nil if unavailable).
func (a *Agent) storeUserMessage(ctx context.Context, chatID int64, msg llm.Message, text string) []float32 {
	sem, hasSem := a.store.(store.SemanticStore)
	if hasSem && a.mcp != nil {
		emb, err := a.mcp.EmbedText(ctx, text)
		if err == nil {
			sem.AddMessageWithEmbedding(chatID, msg, emb)
			return emb
		}
		a.logger.Info("embedding unavailable, storing without", "chat_id", chatID, "err", err)
	}
	a.store.AddMessage(chatID, msg)
	return nil
}

// buildCrossSessionContext searches past sessions and returns a formatted block
// ready to append to the system prompt. Returns "" when nothing relevant is found
// or when semantic search is unavailable.
func (a *Agent) buildCrossSessionContext(_ context.Context, chatID int64, queryEmb []float32) string {
	if len(queryEmb) == 0 {
		return ""
	}
	sem, ok := a.store.(store.SemanticStore)
	if !ok {
		return ""
	}
	snippets := sem.SearchAllSessions(chatID, queryEmb, crossSessionTopK, crossSessionMinScore)
	if len(snippets) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("---\nRelevant context from previous conversations:\n")
	total := sb.Len()

	for _, s := range snippets {
		userText := truncateChars(s.UserText, snippetUserMaxChars)
		botText := truncateChars(s.BotText, snippetBotMaxChars)
		line := fmt.Sprintf("[%s] You: %s\nAssistant: %s\n\n",
			s.Date.Format("2006-01-02"), userText, botText)
		if total+len(line) > crossSessionMaxChars {
			break
		}
		sb.WriteString(line)
		total += len(line)
	}
	return sb.String()
}

// truncateChars truncates s to at most n characters (rune-aware), appending "…" if cut.
func truncateChars(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	runes := []rune(s)
	return string(runes[:n]) + "…"
}

// getHistory returns conversation history. Uses semantic retrieval when a query
// embedding is available; falls back to the standard last-N approach otherwise.
func (a *Agent) getHistory(chatID int64, queryEmb []float32) []llm.Message {
	if sem, ok := a.store.(store.SemanticStore); ok && len(queryEmb) > 0 {
		return sem.GetSemanticHistory(chatID, queryEmb, semanticRecentN, semanticTopK)
	}
	return a.store.GetHistory(chatID)
}

// messageText extracts plain text from a message (handles both Content and Parts).
func messageText(msg llm.Message) string {
	if msg.Content != "" {
		return msg.Content
	}
	var sb strings.Builder
	for _, p := range msg.Parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}
