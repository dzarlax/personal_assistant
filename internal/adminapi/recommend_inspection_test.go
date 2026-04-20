package adminapi

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"telegram-agent/internal/llm"
)

// TestPresetInspection loads production capability + AA data dumps from /tmp
// and prints the top results each preset returns. Skipped if files absent.
//
// Prepare data:
//
//	ssh ... 'docker exec ... psql -c "SELECT model_id, prompt_price, ..."'
//	  > /tmp/caps.psv
//	ssh ... 'docker exec ... psql -c "SELECT value FROM kv_settings ..."'
//	  > /tmp/aa_cache.json
func TestPresetInspection(t *testing.T) {
	capsBytes, err := os.ReadFile("/tmp/caps.psv")
	if err != nil {
		t.Skipf("no /tmp/caps.psv: %v", err)
	}
	aaBytes, err := os.ReadFile("/tmp/aa_cache.json")
	if err != nil {
		t.Skipf("no /tmp/aa_cache.json: %v", err)
	}

	caps := parseCapsPSV(t, string(capsBytes))
	t.Logf("loaded %d openrouter models", len(caps))

	var cache llm.AACache
	if err := json.Unmarshal(aaBytes, &cache); err != nil {
		t.Fatalf("parse aa cache: %v", err)
	}
	t.Logf("loaded %d AA models (fetched at %s)", len(cache.Models), cache.FetchedAt)

	roles := []string{"simple", "default", "complex", "multimodal", "compaction", "classifier"}
	for _, role := range roles {
		preset := rolePresets[role]
		results := applyPreset(caps, cache.Models, role)

		var lines []string
		lines = append(lines, fmt.Sprintf("\n=== %s (%d candidates on Pareto frontier) ===", role, len(results)))
		lines = append(lines, preset.Description)

		axes := preset.Axes
		if axes == nil && role == "classifier" {
			axes = classifierAxes(results)
		}
		if vl := valueLeader(results, axes, 0.5); vl != nil {
			topQ, topP := axes(results[0])
			vQ, vP := axes(*vl)
			lines = append(lines, fmt.Sprintf("Best value: %s  (%.0f%% quality @ %.0f%% price)",
				vl.ID, 100*vQ/topQ, 100*vP/topP))
		}

		lines = append(lines, fmt.Sprintf("%-52s %8s %8s %6s %6s %6s %6s %6s %7s %8s", "model", "prompt$", "compl$", "ctx(k)", "agent", "TPS", "TTFT", "think", "markup", "value"))
		for _, m := range results {
			lines = append(lines, fmt.Sprintf(
				"%-52s %8.3f %8.3f %6d %6.1f %6.0f %6.2f %6.1f %+6.0f%% %8.0f",
				trunc(m.ID, 52),
				m.PromptPrice,
				m.CompletionPrice,
				m.ContextLength/1000,
				m.AgenticIndex,
				m.SpeedTPS,
				m.TTFT,
				m.ThinkTime,
				m.MarkupPct,
				m.ValuePerDollar,
			))
		}
		t.Log(strings.Join(lines, "\n"))
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

func parseCapsPSV(t *testing.T, data string) map[string]llm.Capabilities {
	t.Helper()
	out := make(map[string]llm.Capabilities)
	for _, line := range strings.Split(data, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, "|")
		if len(f) < 8 {
			continue
		}
		c := llm.Capabilities{
			PromptPrice:     mustFloat(f[1]),
			CompletionPrice: mustFloat(f[2]),
			ContextLength:   mustInt(f[3]),
			Vision:          f[4] == "t",
			Tools:           f[5] == "t",
			Reasoning:       f[6] == "t",
			Score:           mustFloat(f[7]),
		}
		out[f[0]] = c
	}
	return out
}

func mustFloat(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func mustInt(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0
	}
	v, _ := strconv.Atoi(s)
	return v
}
