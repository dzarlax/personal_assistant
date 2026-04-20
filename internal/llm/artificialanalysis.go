package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://artificialanalysis.ai/data/llms/models", nil)
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
		return nil, fmt.Errorf("artificialanalysis /data/llms/models: HTTP %d: %s", resp.StatusCode, string(body))
	}
	return parseAAModels(body)
}

type aaModelsResponse struct {
	Status string `json:"status"`
	Data   []struct {
		Slug        string `json:"slug"`
		Evaluations struct {
			ArtificialAnalysisIntelligenceIndex *float64 `json:"artificial_analysis_intelligence_index"`
		} `json:"evaluations"`
		// AA uses its own slugs; OpenRouter slug is also provided.
		OpenrouterSlug *string `json:"openrouter_slug"`
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
		// Index by OpenRouter slug when available (matches our capability store keys).
		if m.OpenrouterSlug != nil && *m.OpenrouterSlug != "" {
			out[*m.OpenrouterSlug] = *score
		}
		// Also index by AA slug as fallback.
		if m.Slug != "" {
			out[m.Slug] = *score
		}
	}
	return out, nil
}

// MergeAAScores overlays AA Intelligence Index scores onto an existing
// capabilities map (keyed by OpenRouter model ID). Models not found in scores
// keep their existing Score value.
func MergeAAScores(caps map[string]Capabilities, scores map[string]float64) {
	for id, c := range caps {
		if s, ok := scores[id]; ok {
			c.Score = s
			caps[id] = c
		}
	}
}
