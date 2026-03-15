package agent

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"telegram-agent/internal/llm"
	"telegram-agent/internal/store"
)

const (
	compactionKeepLast           = 10
	compactionTokenThreshold     = 16000 // trigger compaction at ~16K tokens
	compactionCharPrecheck       = 32000 // cheap SQL char pre-check to skip token counting when far below threshold
	imageTokenCost               = 1000  // approximate visual token cost per image
	clusterSimilarityThreshold   = 0.65  // cosine threshold for grouping turns into the same topic cluster
	compactionTimeout            = 2 * time.Minute // max time for entire compaction operation
)

const compactionSystemPrompt = `Summarise the conversation history into a concise summary in the same language as the conversation.
Preserve: key facts about the user, decisions made, pending tasks, and important context.
Write only the essential content — no preamble or filler.`

// Compacter summarizes old conversation history.
type Compacter struct {
	provider llm.Provider
}

func NewCompacter(provider llm.Provider) *Compacter {
	return &Compacter{provider: provider}
}

// NeedsCompaction returns true if the conversation history should be compacted.
// Uses a two-step check: cheap SQL char count as a pre-filter, then accurate
// token estimation over the actual messages.
func NeedsCompaction(s store.Store, chatID int64) bool {
	cs, ok := s.(store.CompactableStore)
	if !ok {
		return false
	}
	// Fast path: if we're well below threshold even in chars, skip the full load.
	if cs.ActiveCharCount(chatID) < compactionCharPrecheck {
		return false
	}
	rows, err := cs.GetAllActive(chatID)
	if err != nil {
		return false
	}
	total := 0
	for _, row := range rows {
		total += EstimateTokens(row.Message)
	}
	return total > compactionTokenThreshold
}

// EstimateTokens returns a rough token count for a single message.
// Heuristic: 1 token ≈ 4 bytes of UTF-8 text; each image costs ~1000 tokens.
func EstimateTokens(msg llm.Message) int {
	total := 0
	if msg.Content != "" {
		total += len(msg.Content) / 4
	}
	for _, p := range msg.Parts {
		switch p.Type {
		case "text":
			total += len(p.Text) / 4
		case "image_url":
			total += imageTokenCost
		}
	}
	return total
}

// Compact summarizes old messages and marks them as archived.
// If pre-stored embeddings are available it clusters turns by topic first and
// summarises each cluster separately, producing a more structured result.
func (c *Compacter) Compact(ctx context.Context, chatID int64, s store.Store) error {
	ctx, cancel := context.WithTimeout(ctx, compactionTimeout)
	defer cancel()

	cs, ok := s.(store.CompactableStore)
	if !ok {
		return fmt.Errorf("store does not support compaction")
	}

	rows, err := cs.GetAllActive(chatID)
	if err != nil {
		return fmt.Errorf("get active messages: %w", err)
	}

	slog.Info("compact: active messages", "count", len(rows), "char_count", cs.ActiveCharCount(chatID))

	boundary := findBoundary(rows, compactionKeepLast)
	slog.Info("compact: boundary", "boundary", boundary, "keep_last", compactionKeepLast)
	if boundary == 0 {
		return nil // nothing to compact
	}

	toCompact := rows[:boundary]

	var summary string
	if hasEmbeddings(toCompact) {
		summary, err = c.semanticCompact(ctx, toCompact)
	} else {
		summary, err = c.simpleCompact(ctx, toCompact)
	}
	if err != nil {
		return fmt.Errorf("summarize: %w", err)
	}

	// Insert summary before marking old messages as compacted
	cs.AddSummary(chatID, "[Summary of previous conversation]\n\n"+summary)

	ids := make([]int64, len(toCompact))
	for i, row := range toCompact {
		ids[i] = row.ID
	}
	return cs.MarkCompacted(ids)
}

// simpleCompact summarises all rows as a single conversation history.
func (c *Compacter) simpleCompact(ctx context.Context, rows []store.MessageRow) (string, error) {
	history := make([]llm.Message, len(rows))
	for i, row := range rows {
		history[i] = row.Message
	}
	var (
		resp llm.Response
		err  error
	)
	for attempt := range 2 {
		resp, err = c.provider.Chat(ctx, history, compactionSystemPrompt, nil)
		if err == nil {
			break
		}
		slog.Warn("compaction attempt failed", "attempt", attempt+1, "err", err)
	}
	return resp.Content, err
}

// semanticCompact clusters rows by topic (using stored embeddings) and
// summarises each cluster separately, then joins the results.
// Falls back to simpleCompact when only one cluster is found.
func (c *Compacter) semanticCompact(ctx context.Context, rows []store.MessageRow) (string, error) {
	clusters := clusterByEmbedding(rows, clusterSimilarityThreshold)
	slog.Info("compact: semantic clustering", "clusters", len(clusters))

	if len(clusters) <= 1 {
		return c.simpleCompact(ctx, rows)
	}

	var summaries []string
	for i, cluster := range clusters {
		history := make([]llm.Message, len(cluster))
		for j, row := range cluster {
			history[j] = row.Message
		}
		var (
			resp llm.Response
			err  error
		)
		for attempt := range 2 {
			resp, err = c.provider.Chat(ctx, history, compactionSystemPrompt, nil)
			if err == nil {
				break
			}
			slog.Warn("cluster compaction attempt failed", "cluster", i, "attempt", attempt+1, "err", err)
		}
		if err != nil {
			slog.Warn("cluster compaction failed, skipping", "cluster", i, "err", err)
			continue
		}
		summaries = append(summaries, resp.Content)
	}

	if len(summaries) == 0 {
		return c.simpleCompact(ctx, rows)
	}

	return strings.Join(summaries, "\n\n---\n\n"), nil
}

// clusterByEmbedding groups rows into clusters of consecutive turns using
// greedy cosine similarity between user-message embeddings.
// A new cluster is started when similarity to the current centroid drops below threshold.
func clusterByEmbedding(rows []store.MessageRow, threshold float64) [][]store.MessageRow {
	// Group rows into turns: each turn starts at a user message.
	type turn struct {
		rows []store.MessageRow
		emb  []float32
	}
	var turns []turn
	var cur []store.MessageRow
	for _, r := range rows {
		if r.Message.Role == "user" && len(cur) > 0 {
			turns = append(turns, turn{rows: cur, emb: userEmbedding(cur)})
			cur = nil
		}
		cur = append(cur, r)
	}
	if len(cur) > 0 {
		turns = append(turns, turn{rows: cur, emb: userEmbedding(cur)})
	}

	if len(turns) == 0 {
		return nil
	}

	// Greedy clustering: extend current cluster while similar, else start new.
	var clusters [][]store.MessageRow
	var clusterRows []store.MessageRow
	var centroid []float32

	for _, t := range turns {
		switch {
		case centroid == nil:
			// First turn initialises the cluster.
			clusterRows = append(clusterRows, t.rows...)
			centroid = t.emb
		case len(t.emb) == 0:
			// No embedding — keep in current cluster.
			clusterRows = append(clusterRows, t.rows...)
		case compactCosine(centroid, t.emb) >= threshold:
			clusterRows = append(clusterRows, t.rows...)
			centroid = avgVec(centroid, t.emb)
		default:
			clusters = append(clusters, clusterRows)
			clusterRows = append([]store.MessageRow{}, t.rows...)
			centroid = t.emb
		}
	}
	if len(clusterRows) > 0 {
		clusters = append(clusters, clusterRows)
	}

	return clusters
}

// userEmbedding returns the embedding of the first user message in rows.
func userEmbedding(rows []store.MessageRow) []float32 {
	for _, r := range rows {
		if r.Message.Role == "user" && len(r.Embedding) > 0 {
			return r.Embedding
		}
	}
	return nil
}

// hasEmbeddings returns true if at least one user-message row has an embedding.
func hasEmbeddings(rows []store.MessageRow) bool {
	for _, r := range rows {
		if r.Message.Role == "user" && len(r.Embedding) > 0 {
			return true
		}
	}
	return false
}

func compactCosine(a, b []float32) float64 {
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

func avgVec(a, b []float32) []float32 {
	if len(a) != len(b) {
		return a
	}
	result := make([]float32, len(a))
	for i := range a {
		result[i] = (a[i] + b[i]) / 2
	}
	return result
}

// findBoundary finds the split point: messages before this index get compacted.
// Snaps to a user message boundary to avoid splitting tool call sequences.
func findBoundary(rows []store.MessageRow, keepLast int) int {
	if len(rows) <= keepLast {
		return 0
	}
	boundary := len(rows) - keepLast
	// Snap back to the nearest user message so we don't split mid-sequence
	for boundary > 0 && rows[boundary].Message.Role != "user" {
		boundary--
	}
	return boundary
}
