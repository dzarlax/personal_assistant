package adminapi

import (
	"regexp"
	"sort"

	"telegram-agent/internal/llm"
)

// Role-driven recommendation engine. When the user clicks "Suggest" next to a
// routing role in the admin UI, the model browser applies the matching
// preset: a set of include/exclude filters + a sort strategy. Result: a
// curated but honest list of candidates — no opaque "quality score", just
// transparent filters you can see and relax.

// multilingualRegex matches OpenRouter model id prefixes that have proven
// reasonable non-English (specifically Russian) support. Smaller English-
// primary variants (gemma-small, phi-mini, nemotron) and Cohere command-r
// small versions are intentionally NOT included.
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

// excludedVendorsRegex — providers we can reach directly elsewhere (Anthropic
// via the bridge, Google directly); going through OpenRouter for them pays a
// margin we don't need to pay.
var excludedVendorsRegex = regexp.MustCompile(`^(anthropic|openai)/`)

// specialisedCoderRegex — fine-tuned on code; poor fit for general chat.
var specialisedCoderRegex = regexp.MustCompile(`-coder(-|$|:)`)

// specialisedVisionRegex — vision-only (-vl) variants are saved for the
// multimodal preset; they're needlessly expensive for plain text.
var specialisedVisionRegex = regexp.MustCompile(`-vl-`)

// thinkingRegex matches model ids that are actually frontier reasoners — the
// `reasoning: true` capability flag alone is not enough (8B models also set
// it just because the API accepts a reasoning parameter). Match the naming
// conventions of genuine reasoning variants.
var thinkingRegex = regexp.MustCompile(
	`(-thinking|:thinking|/qwq|/deepseek-r[0-9]|-r1(-|$)|-reasoner)`,
)

// sortStrategy chooses the ordering key for a preset.
type sortStrategy int

const (
	sortByPromptPrice sortStrategy = iota
	sortByCompletionPrice
	sortByScorePerDollar // AA Intelligence Index / prompt price (higher = better value)
)

// rolePreset describes how to filter + order the OpenRouter catalog for a
// given routing role.
type rolePreset struct {
	// human-readable summary shown in the banner
	Description string
	// check each capability; models that fail any return false
	Filter func(caps llm.Capabilities, modelID string) bool
	// ordering
	Sort sortStrategy
}

// rolePresets — one entry per routing role. Absent role name = no preset
// available (all filters off).
var rolePresets = map[string]rolePreset{
	"simple": {
		Description: "tools + multilingual, ≤ $1/M prompt, no coder/vl/free variants — sorted by AA score/price",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Tools &&
				c.PromptPrice > 0 && c.PromptPrice <= 1.0
		},
		Sort: sortByScorePerDollar,
	},

	"default": {
		Description: "tools + multilingual, ≤ $2/M prompt, no coder/vl/free variants — sorted by AA score/price",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Tools &&
				c.PromptPrice > 0 && c.PromptPrice <= 2.0
		},
		Sort: sortByScorePerDollar,
	},

	"complex": {
		Description: "frontier reasoners only (models with -thinking / -r1 / qwq in name). Tools + multilingual, ≤ $5/M prompt. Claude via bridge is the preferred choice when configured.",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				thinkingRegex.MatchString(id) &&
				c.Tools && c.Reasoning &&
				c.PromptPrice > 0 && c.PromptPrice <= 5.0
		},
		Sort: sortByPromptPrice,
	},

	"multimodal": {
		Description: "vision-capable Gemini/Qwen VL; note: native audio transcription only works via direct Gemini (out of OR catalog)",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.Vision && c.Tools &&
				c.PromptPrice > 0 && c.PromptPrice <= 2.0
		},
		Sort: sortByPromptPrice,
	},

	"compaction": {
		Description: "ctx ≥ 32k, multilingual, sorted by COMPLETION price (summaries are output-heavy). No tools required.",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				!isFreeVariant(id) &&
				c.ContextLength >= 32000 &&
				c.CompletionPrice > 0 && c.CompletionPrice <= 1.0
		},
		Sort: sortByCompletionPrice,
	},

	"classifier": {
		Description: "≤ $0.1/M prompt; used for complexity rating (digit output). Local Ollama stays the primary recommendation.",
		Filter: func(c llm.Capabilities, id string) bool {
			return multilingualRegex.MatchString(id) &&
				!excludedVendorsRegex.MatchString(id) &&
				!specialisedCoderRegex.MatchString(id) &&
				!specialisedVisionRegex.MatchString(id) &&
				c.PromptPrice >= 0 && c.PromptPrice <= 0.1
		},
		Sort: sortByPromptPrice,
	},

	// "fallback" has no preset here — it should use a DIRECT provider
	// (different vendor from the default) to survive an OpenRouter outage.
	// Browsing OR candidates for it would be misleading, so we leave
	// the button off in the UI for this role.
}

func isFreeVariant(modelID string) bool {
	return len(modelID) > 5 && modelID[len(modelID)-5:] == ":free"
}

// scorePerDollar returns AA Intelligence Index per USD/1M prompt tokens.
// Models without a score fall back to sorting by price only (score=0 → value=0,
// so they appear after scored models).
func scorePerDollar(score, promptPrice float64) float64 {
	if score == 0 || promptPrice == 0 {
		return 0
	}
	return score / promptPrice
}

// applyPreset returns the models matching the preset for role, sorted per
// the preset's strategy. If the role has no preset, returns nil (caller
// should fall back to the full catalog).
func applyPreset(all map[string]llm.Capabilities, role string) []uiModel {
	preset, ok := rolePresets[role]
	if !ok {
		return nil
	}
	out := make([]uiModel, 0, len(all))
	for id, c := range all {
		if !preset.Filter(c, id) {
			continue
		}
		out = append(out, uiModel{
			ID:              id,
			PromptPrice:     c.PromptPrice,
			CompletionPrice: c.CompletionPrice,
			ContextLength:   c.ContextLength,
			Vision:          c.Vision,
			Tools:           c.Tools,
			Reasoning:       c.Reasoning,
			Free:            c.Free(),
			Score:           c.Score,
		})
	}
	switch preset.Sort {
	case sortByCompletionPrice:
		sort.Slice(out, func(i, j int) bool {
			if out[i].CompletionPrice != out[j].CompletionPrice {
				return out[i].CompletionPrice < out[j].CompletionPrice
			}
			return out[i].ID < out[j].ID
		})
	case sortByScorePerDollar:
		sort.Slice(out, func(i, j int) bool {
			vi := scorePerDollar(out[i].Score, out[i].PromptPrice)
			vj := scorePerDollar(out[j].Score, out[j].PromptPrice)
			if vi != vj {
				return vi > vj // higher value first
			}
			return out[i].ID < out[j].ID
		})
	default:
		sort.Slice(out, func(i, j int) bool {
			if out[i].PromptPrice != out[j].PromptPrice {
				return out[i].PromptPrice < out[j].PromptPrice
			}
			return out[i].ID < out[j].ID
		})
	}
	return out
}

// presetRoles returns the list of roles that have a preset, in display order.
// Used by the UI to render "Suggest" buttons only for these roles.
func presetRoles() map[string]bool {
	out := make(map[string]bool, len(rolePresets))
	for role := range rolePresets {
		out[role] = true
	}
	return out
}
