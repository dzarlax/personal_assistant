package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var aaHTTPClient = &http.Client{Timeout: 20 * time.Second}

// FetchArtificialAnalysisScores fetches the Intelligence Index scores from the
// Artificial Analysis API and returns a map[modelSlug]score. The slug format
// used by AA differs from OpenRouter IDs, so callers should merge by AA slug
// after mapping through the openrouter_slug field.
//
// API docs: https://artificialanalysis.ai/api-reference
// Free tier: 1000 req/day, attribution required.
func FetchArtificialAnalysisScores(ctx context.Context, apiKey string) (map[string]float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://artificialanalysis.ai/api/v2/data/llms/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", apiKey)

	resp, err := aaHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("artificialanalysis /data/llms/models: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("artificialanalysis /data/llms/models: read: %w", err)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("artificialanalysis /api/v2/data/llms/models: HTTP %d: %s", resp.StatusCode, string(body))
	}
	return parseAAModels(body)
}

type aaModelsResponse struct {
	Status json.RawMessage `json:"status"`
	Data   []struct {
		Slug         string `json:"slug"`
		ModelCreator struct {
			Slug string `json:"slug"`
		} `json:"model_creator"`
		Evaluations struct {
			ArtificialAnalysisIntelligenceIndex *float64 `json:"artificial_analysis_intelligence_index"`
		} `json:"evaluations"`
	} `json:"data"`
}

func parseAAModels(body []byte) (map[string]float64, error) {
	var resp aaModelsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("artificialanalysis /data/llms/models: parse: %w", err)
	}
	out := make(map[string]float64, len(resp.Data))
	for _, m := range resp.Data {
		score := m.Evaluations.ArtificialAnalysisIntelligenceIndex
		if score == nil || *score == 0 {
			continue
		}
		// AA v2 uses {creator_slug}/{model_slug} format, which maps to OpenRouter IDs
		// after normalizing dots↔dashes (done in MergeAAScores).
		if m.ModelCreator.Slug != "" && m.Slug != "" {
			out[m.ModelCreator.Slug+"/"+m.Slug] = *score
		}
	}
	return out, nil
}

// MergeAAScores overlays AA Intelligence Index scores onto an existing
// capabilities map (keyed by OpenRouter model ID). Models not found in scores
// keep their existing Score value.
//
// AA slugs use dashes for version separators (gemini-2-5-pro) while OpenRouter
// uses dots (gemini-2.5-pro), so we try both exact and dot→dash normalized match.
func MergeAAScores(caps map[string]Capabilities, scores map[string]float64) {
	// Build a dot→dash normalized lookup so OR IDs like "google/gemini-2.5-pro"
	// match AA slugs like "google/gemini-2-5-pro".
	normed := make(map[string]float64, len(scores))
	for k, v := range scores {
		normed[strings.ReplaceAll(k, ".", "-")] = v
	}

	for id, c := range caps {
		if s, ok := scores[id]; ok {
			c.Score = s
			caps[id] = c
			continue
		}
		if s, ok := normed[strings.ReplaceAll(id, ".", "-")]; ok {
			c.Score = s
			caps[id] = c
		}
	}
}
