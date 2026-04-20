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

const aaSettingsKey = "aa.models.cache"
const aaCacheTTL = 24 * time.Hour

var aaHTTPClient = &http.Client{Timeout: 20 * time.Second}

// AAModelInfo holds normalized AA data for a single model, keyed by the
// OR-compatible slug (dots replaced with dashes in version segments).
type AAModelInfo struct {
	AASlug      string  `json:"aa_slug"`
	CreatorSlug string  `json:"creator_slug"`
	Score       float64 `json:"score,omitempty"`       // Intelligence Index
	CodingIndex float64 `json:"coding_index,omitempty"` // Coding Index
	MathIndex   float64 `json:"math_index,omitempty"`   // Math Index
	SpeedTPS    float64 `json:"speed_tps,omitempty"`    // median output tokens/sec
	TTFT        float64 `json:"ttft_s,omitempty"`       // median time-to-first-token, seconds
	PriceInput  float64 `json:"price_input_1m,omitempty"`
	PriceOutput float64 `json:"price_output_1m,omitempty"`
}

// AACache is the kv_settings blob stored under aaSettingsKey.
type AACache struct {
	FetchedAt time.Time              `json:"fetched_at"`
	Models    map[string]AAModelInfo `json:"models"` // key = normalized OR-compatible ID
}

// FetchArtificialAnalysisData fetches full model data from the AA API and
// returns it as a normalized map keyed by OR-compatible IDs.
//
// API docs: https://artificialanalysis.ai/api-reference
// Free tier: 1000 req/day, attribution required.
func FetchArtificialAnalysisData(ctx context.Context, apiKey string) (map[string]AAModelInfo, error) {
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
	return parseAAData(body)
}

type aaModelsResponse struct {
	Status json.RawMessage `json:"status"`
	Data   []struct {
		Slug         string `json:"slug"`
		ModelCreator struct {
			Slug string `json:"slug"`
		} `json:"model_creator"`
		Evaluations struct {
			IntelligenceIndex *float64 `json:"artificial_analysis_intelligence_index"`
			CodingIndex       *float64 `json:"artificial_analysis_coding_index"`
			MathIndex         *float64 `json:"artificial_analysis_math_index"`
		} `json:"evaluations"`
		MedianOutputTPS  float64 `json:"median_output_tokens_per_second"`
		MedianTTFT       float64 `json:"median_time_to_first_token_seconds"`
		Pricing          struct {
			Input  float64 `json:"price_1m_input_tokens"`
			Output float64 `json:"price_1m_output_tokens"`
		} `json:"pricing"`
	} `json:"data"`
}

// normalizeAAKey converts an AA {creator}/{slug} into an OR-compatible key
// by replacing dots with dashes (AA uses dashes, OR uses dots for versions).
func normalizeAAKey(creator, slug string) string {
	return strings.ReplaceAll(creator+"/"+slug, ".", "-")
}

func parseAAData(body []byte) (map[string]AAModelInfo, error) {
	var resp aaModelsResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("artificialanalysis /data/llms/models: parse: %w", err)
	}
	out := make(map[string]AAModelInfo, len(resp.Data))
	for _, m := range resp.Data {
		if m.ModelCreator.Slug == "" || m.Slug == "" {
			continue
		}
		info := AAModelInfo{
			AASlug:      m.Slug,
			CreatorSlug: m.ModelCreator.Slug,
			SpeedTPS:    m.MedianOutputTPS,
			TTFT:        m.MedianTTFT,
			PriceInput:  m.Pricing.Input,
			PriceOutput: m.Pricing.Output,
		}
		if v := m.Evaluations.IntelligenceIndex; v != nil {
			info.Score = *v
		}
		if v := m.Evaluations.CodingIndex; v != nil {
			info.CodingIndex = *v
		}
		if v := m.Evaluations.MathIndex; v != nil {
			info.MathIndex = *v
		}
		out[normalizeAAKey(m.ModelCreator.Slug, m.Slug)] = info
	}
	return out, nil
}

// StoreAACache persists the AA model map into kv_settings.
func StoreAACache(ctx context.Context, settings SettingsStore, models map[string]AAModelInfo) error {
	cache := AACache{FetchedAt: time.Now(), Models: models}
	data, err := json.Marshal(cache)
	if err != nil {
		return fmt.Errorf("aa cache marshal: %w", err)
	}
	return settings.PutSetting(ctx, aaSettingsKey, string(data))
}

// LoadAACache reads the AA model cache from kv_settings.
// Returns (nil, nil) when absent or expired.
func LoadAACache(ctx context.Context, settings SettingsStore) (*AACache, error) {
	raw, ok, err := settings.GetSetting(ctx, aaSettingsKey)
	if err != nil || !ok {
		return nil, err
	}
	var cache AACache
	if err := json.Unmarshal([]byte(raw), &cache); err != nil {
		return nil, fmt.Errorf("aa cache unmarshal: %w", err)
	}
	if time.Since(cache.FetchedAt) > aaCacheTTL {
		return nil, nil
	}
	return &cache, nil
}

// creatorAliases maps OR creator prefixes that differ from AA's.
var creatorAliases = map[string]string{
	"moonshotai": "moonshot",
}

// MergeAAScores overlays AA Intelligence Index scores onto an existing
// capabilities map (keyed by OpenRouter model ID). Models not found in the
// AA cache keep their existing Score value.
//
// Matching strategy (tried in order, first win):
//  1. Normalize dots→dashes, exact match
//  2. Strip OR variant suffix (:free/:nitro/:beta/etc.), exact match
//  3. Prefix match for date/preview variants
//  4. Token-level transforms (instruct suffix, -it strip, thinking→reasoning, creator alias, size/instruct swap)
//  5. Anthropic word-order swap: claude-{family}-{ver} ↔ claude-{ver}-{family}
func MergeAAScores(caps map[string]Capabilities, models map[string]AAModelInfo) {
	for id, c := range caps {
		if info := lookupAAInfo(id, models); info != nil && info.Score > 0 {
			c.Score = info.Score
			caps[id] = c
		}
	}
}

// lookupAAInfo tries all mapping strategies for a single OR model ID.
func lookupAAInfo(orID string, models map[string]AAModelInfo) *AAModelInfo {
	norm := strings.ReplaceAll(orID, ".", "-")

	// 1. Exact match after dot→dash normalization.
	if info, ok := models[norm]; ok {
		return &info
	}

	// 2. Strip OR variant suffix (:free, :nitro, :beta, :thinking, etc.).
	base := norm
	if i := strings.Index(norm, ":"); i != -1 {
		base = norm[:i]
	}
	if base != norm {
		if info, ok := models[base]; ok {
			return &info
		}
	}

	// 3. Prefix match for date/preview variants.
	if info := prefixMatch(base, models); info != nil {
		return info
	}

	// 4. Token-level transforms — generate candidate keys, try each.
	for _, candidate := range tokenTransforms(base) {
		if info, ok := models[candidate]; ok {
			return &info
		}
		// Also apply prefix match on each transformed candidate.
		if info := prefixMatch(candidate, models); info != nil {
			return info
		}
	}

	// 5. Anthropic word-order swap: claude-{family}-{ver} ↔ claude-{ver}-{family}.
	if strings.HasPrefix(norm, "anthropic/claude-") {
		if info := anthropicSwap(base, models); info != nil {
			return info
		}
	}

	return nil
}

// tokenTransforms returns alternative candidate keys for an OR base slug by
// applying known naming convention differences between OR and AA.
func tokenTransforms(base string) []string {
	var out []string

	creator, slug, ok := strings.Cut(base, "/")
	if !ok {
		return nil
	}

	// Creator alias (e.g. moonshotai → moonshot).
	if alias, hasAlias := creatorAliases[creator]; hasAlias {
		out = append(out, alias+"/"+slug)
		// Also try the alias with further transforms below.
		creator = alias
		base = creator + "/" + slug
	}

	// Strip -it suffix (Google Gemma: gemma-3-27b-it → gemma-3-27b).
	if strings.HasSuffix(slug, "-it") {
		out = append(out, creator+"/"+slug[:len(slug)-3])
	}

	// Add -instruct suffix when absent (OR often omits it: qwen3-14b → qwen3-14b-instruct).
	if !strings.HasSuffix(slug, "-instruct") && !strings.Contains(slug, "-instruct-") {
		out = append(out, base+"-instruct")
	}

	// -thinking → -instruct + -reasoning variants:
	//   qwen3-235b-a22b-thinking-2507 → qwen3-235b-a22b-instruct-2507-reasoning
	if strings.Contains(slug, "-thinking") {
		withInstruct := strings.ReplaceAll(slug, "-thinking", "-instruct")
		out = append(out, creator+"/"+withInstruct+"-reasoning")
		out = append(out, creator+"/"+withInstruct)
	}

	// Inject -instruct before a 4-digit date suffix when not already present:
	//   qwen3-235b-a22b-2507 → qwen3-235b-a22b-instruct-2507
	if !strings.Contains(slug, "-instruct") {
		tokens := strings.Split(slug, "-")
		for i, t := range tokens {
			if len(t) == 4 && t[0] >= '2' && t[0] <= '9' && isDigits(t) {
				before := strings.Join(tokens[:i], "-")
				after := strings.Join(tokens[i:], "-")
				out = append(out, creator+"/"+before+"-instruct-"+after)
				break
			}
		}
	}

	// Meta-Llama size/instruct word order: llama-3-1-70b-instruct → llama-3-1-instruct-70b.
	if creator == "meta-llama" && strings.HasSuffix(slug, "-instruct") {
		tokens := strings.Split(strings.TrimSuffix(slug, "-instruct"), "-")
		// Find last size token (ends in 'b' preceded by digits).
		for i := len(tokens) - 1; i >= 0; i-- {
			t := tokens[i]
			if len(t) > 1 && t[len(t)-1] == 'b' && t[len(t)-2] >= '0' && t[len(t)-2] <= '9' {
				reordered := make([]string, 0, len(tokens)+1)
				reordered = append(reordered, tokens[:i]...)
				reordered = append(reordered, "instruct")
				reordered = append(reordered, t)
				reordered = append(reordered, tokens[i+1:]...)
				out = append(out, creator+"/"+strings.Join(reordered, "-"))
				break
			}
		}
	}

	return out
}

// previewMarkers are suffixes that OR appends to a base model slug to indicate
// a preview or dated release. We only prefix-match on these to avoid false
// positives between distinct model families.
var previewMarkers = []string{"-preview", "-exp", "-0", "-1", "-2"}

func prefixMatch(orBase string, models map[string]AAModelInfo) *AAModelInfo {
	var best *AAModelInfo
	bestLen := 0
	for k, info := range models {
		if !strings.HasPrefix(orBase, k) {
			continue
		}
		rest := orBase[len(k):]
		matched := rest == ""
		if !matched {
			for _, m := range previewMarkers {
				if strings.HasPrefix(rest, m) {
					matched = true
					break
				}
			}
		}
		if matched && len(k) > bestLen {
			cp := info
			best = &cp
			bestLen = len(k)
		}
	}
	return best
}

// anthropicSwap handles the OR/AA naming inversion for Claude models.
// OR uses claude-{family}-{ver} (e.g. claude-sonnet-4-5),
// AA uses claude-{ver}-{family} (e.g. claude-4-5-sonnet).
func anthropicSwap(orBase string, models map[string]AAModelInfo) *AAModelInfo {
	const pfx = "anthropic/claude-"
	rest := strings.TrimPrefix(orBase, pfx)
	tokens := strings.Split(rest, "-")
	if len(tokens) < 2 {
		return nil
	}
	for i, tok := range tokens {
		if !isAlpha(tok) {
			continue
		}
		others := make([]string, 0, len(tokens)-1)
		others = append(others, tokens[:i]...)
		others = append(others, tokens[i+1:]...)

		// alpha at end (OR: sonnet-4-5 → AA: 4-5-sonnet)
		candidate := pfx + strings.Join(append(others, tok), "-")
		if info, ok := models[candidate]; ok {
			return &info
		}
		// alpha at front
		candidate = pfx + strings.Join(append([]string{tok}, others...), "-")
		if info, ok := models[candidate]; ok {
			return &info
		}
	}
	return nil
}

func isAlpha(s string) bool {
	for _, r := range s {
		if r < 'a' || r > 'z' {
			return false
		}
	}
	return s != ""
}

func isDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return s != ""
}
