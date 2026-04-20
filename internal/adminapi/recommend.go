package adminapi

import (
	"regexp"
	"sort"

	"telegram-agent/internal/llm"
)

// Role-driven recommendation engine. When the user clicks "Suggest" next to a
// routing role in the admin UI, the model browser applies the matching
// preset: a set of include/exclude filters + a Pareto frontier on
// (role-specific quality metric, price). Result: every model shown is a
// valid trade-off — no model in the list is strictly dominated (worse AND
// more expensive) by another.

var multilingualRegex = regexp.MustCompile(
	`^(` +
		`deepseek/(deepseek-chat|deepseek-r1|deepseek-v3)` +
		`|qwen/qwen3(\.[0-9]+)?(-|$)` +
		`|qwen/qwen-(plus|max|turbo)` +
		`|qwen/qwq` +
		`|z-ai/glm-4\.[5-9]` +
		`|moonshotai/kimi-` +
		`|google/gemini-(2\.5|3|3\.1)-(flash|pro)` +
		`|mistralai/mistral-(large|medium)` +
		`|x-ai/grok-[34]` +
		`)`,
)

var excludedVendorsRegex = regexp.MustCompile(`^(anthropic|openai)/`)

var specialisedCoderRegex = regexp.MustCompile(`-coder(-|$|:)`)

var specialisedVisionRegex = regexp.MustCompile(`-vl-`)

// thinkingRegex matches model ids that are actually frontier reasoners — the
// `reasoning: true` capability flag alone is not enough (8B models also set
// it just because the API accepts a reasoning parameter).
var thinkingRegex = regexp.MustCompile(
	`(-thinking|:thinking|/qwq|/deepseek-r[0-9]|-r1(-|$)|-reasoner)`,
)

// paretoAxes returns (quality, price) for a model under a given role.
type paretoAxes func(m uiModel) (quality, price float64)

// rolePreset describes how to filter + rank models for a given routing role.
type rolePreset struct {
	Description string
	Filter      func(caps llm.Capabilities, modelID string, aa llm.AAModelInfo) bool
	Axes        paretoAxes
}

// bestAgentic — use AA Agentic Index when available; fall back to Intelligence
// Index for untested models (scaled down to de-rank vs. tested models).
func bestAgentic(m uiModel) float64 {
	if m.AgenticIndex > 0 {
		return m.AgenticIndex
	}
	return m.Score
}

// inverseTTFT — classifier emits one digit; TTFT (time-to-first-token)
// dominates total latency. Returns 0 when no TTFT data (excluded by
// paretoFrontier's quality>0 guard).
func inverseTTFT(m uiModel) float64 {
	if m.TTFT <= 0 {
		return 0
	}
	return 1.0 / m.TTFT
}

var rolePresets = map[string]rolePreset{
	"simple": {
		Description: "tools + multilingual, ≤ $0.2/M prompt, ctx ≥ 32k. Pareto frontier on (AA Agentic Index, prompt price).",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Tools &&
				c.ContextLength >= 32000 &&
				c.PromptPrice > 0 && c.PromptPrice <= 0.2
		},
		Axes: func(m uiModel) (float64, float64) { return bestAgentic(m), m.PromptPrice },
	},

	"default": {
		Description: "tools + multilingual, ≤ $2/M prompt, ctx ≥ 32k. Pareto frontier on (AA Agentic Index, prompt price).",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Tools &&
				c.ContextLength >= 32000 &&
				c.PromptPrice > 0 && c.PromptPrice <= 2.0
		},
		Axes: func(m uiModel) (float64, float64) { return bestAgentic(m), m.PromptPrice },
	},

	"complex": {
		Description: "frontier reasoners (thinking/r1/qwq) with tools + multilingual, ≤ $5/M prompt, ctx ≥ 64k. Pareto frontier on (AA Agentic Index, prompt price). Claude via bridge is preferred when configured.",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				thinkingRegex.MatchString(id) &&
				c.Tools && c.Reasoning &&
				c.ContextLength >= 64000 &&
				c.PromptPrice > 0 && c.PromptPrice <= 5.0
		},
		Axes: func(m uiModel) (float64, float64) { return bestAgentic(m), m.PromptPrice },
	},

	"multimodal": {
		Description: "vision + tools + multilingual, ≤ $2/M prompt, ctx ≥ 32k. Pareto frontier on (AA Intelligence Index, prompt price).",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Vision && c.Tools &&
				c.ContextLength >= 32000 &&
				c.PromptPrice > 0 && c.PromptPrice <= 2.0
		},
		Axes: func(m uiModel) (float64, float64) { return m.Score, m.PromptPrice },
	},

	"compaction": {
		Description: "multilingual, ctx ≥ 64k (long history in, short summary out), completion ≤ $2/M. Pareto frontier on (AA Intelligence Index, completion price).",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.ContextLength >= 64000 &&
				c.CompletionPrice > 0 && c.CompletionPrice <= 2.0
		},
		Axes: func(m uiModel) (float64, float64) { return m.Score, m.CompletionPrice },
	},

	"classifier": {
		Description: "≤ $0.1/M prompt, multilingual, no :free (rate-limited on OR). Pareto frontier on (1/TTFT, prompt price) when speed data available, else (AA Intelligence Index, prompt price). Local Ollama stays the primary recommendation.",
		Filter: func(c llm.Capabilities, id string, aa llm.AAModelInfo) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.PromptPrice > 0 && c.PromptPrice <= 0.1
		},
		// Axes selected dynamically in applyPreset (see classifierAxes).
		Axes: nil,
	},

	// "fallback" has no preset — it should point at a DIRECT provider
	// (different vendor from the default) to survive an OpenRouter outage.
}

func isFreeVariant(modelID string) bool {
	return len(modelID) > 5 && modelID[len(modelID)-5:] == ":free"
}

// classifierAxes picks (1/TTFT, price) when at least one candidate has TTFT
// data, otherwise falls back to (Score, price). Mixing two quality scales in
// the same Pareto frontier would be meaningless, so we pick one globally.
func classifierAxes(candidates []uiModel) paretoAxes {
	for _, m := range candidates {
		if m.TTFT > 0 {
			return func(m uiModel) (float64, float64) { return inverseTTFT(m), m.PromptPrice }
		}
	}
	return func(m uiModel) (float64, float64) { return m.Score, m.PromptPrice }
}

// paretoFrontier keeps only non-dominated models. A model is also excluded
// if its quality is 0 — Pareto would otherwise keep untested models at the
// price floor just because no one beats them on both axes. Requiring quality
// > 0 means recommendations are always based on real AA data.
func paretoFrontier(models []uiModel, axes paretoAxes) []uiModel {
	out := make([]uiModel, 0, len(models))
	for i, m := range models {
		qi, pi := axes(m)
		if qi <= 0 {
			continue
		}
		dominated := false
		for j, other := range models {
			if i == j {
				continue
			}
			qj, pj := axes(other)
			if qj <= 0 {
				continue
			}
			strictlyBetter := (qj > qi && pj <= pi) || (qj >= qi && pj < pi)
			if strictlyBetter {
				dominated = true
				break
			}
		}
		if !dominated {
			out = append(out, m)
		}
	}
	return out
}

// applyPreset returns the Pareto-optimal models for the role, sorted by
// quality descending (best first). If the role has no preset, returns nil.
// Each returned model has ValuePerDollar populated using the role's axes.
func applyPreset(all map[string]llm.Capabilities, aaModels map[string]llm.AAModelInfo, role string) []uiModel {
	preset, ok := rolePresets[role]
	if !ok {
		return nil
	}
	candidates := make([]uiModel, 0, len(all))
	for id, c := range all {
		var aa llm.AAModelInfo
		if aaModels != nil {
			if info := llm.LookupAAInfo(id, aaModels); info != nil {
				aa = *info
			}
		}
		if !preset.Filter(c, id, aa) {
			continue
		}
		m := uiModel{
			ID:              id,
			PromptPrice:     c.PromptPrice,
			CompletionPrice: c.CompletionPrice,
			ContextLength:   c.ContextLength,
			Vision:          c.Vision,
			Tools:           c.Tools,
			Reasoning:       c.Reasoning,
			Free:            c.Free(),
			Score:           c.Score,
		}
		enrichFromAA(&m, aa)
		candidates = append(candidates, m)
	}
	axes := preset.Axes
	if axes == nil && role == "classifier" {
		axes = classifierAxes(candidates)
	}
	frontier := paretoFrontier(candidates, axes)
	sort.Slice(frontier, func(i, j int) bool {
		qi, pi := axes(frontier[i])
		qj, pj := axes(frontier[j])
		if qi != qj {
			return qi > qj
		}
		if pi != pj {
			return pi < pj
		}
		return frontier[i].ID < frontier[j].ID
	})
	for i := range frontier {
		q, p := axes(frontier[i])
		if p > 0 && q > 0 {
			frontier[i].ValuePerDollar = q / p
		}
	}
	return frontier
}

// valueLeader returns the frontier model with the best quality/price ratio,
// subject to role-quality ≥ qualityFloor × topQuality. Returns nil when the
// value leader is already the top-quality model (no meaningful trade-off to
// surface) or the frontier is trivially small.
func valueLeader(frontier []uiModel, axes paretoAxes, qualityFloor float64) *uiModel {
	if len(frontier) < 2 {
		return nil
	}
	topQuality, _ := axes(frontier[0])
	floor := qualityFloor * topQuality
	bestIdx := -1
	bestVal := 0.0
	for i := range frontier {
		q, p := axes(frontier[i])
		if q < floor || p <= 0 {
			continue
		}
		if v := q / p; v > bestVal {
			bestVal = v
			bestIdx = i
		}
	}
	if bestIdx <= 0 { // not found, or it's frontier[0] itself
		return nil
	}
	return &frontier[bestIdx]
}

// presetRoles returns the list of roles that have a preset, in display order.
func presetRoles() map[string]bool {
	out := make(map[string]bool, len(rolePresets))
	for role := range rolePresets {
		out[role] = true
	}
	return out
}
