package adminapi

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"telegram-agent/internal/llm"
)

// --- View data ---

type uiRole struct {
	Name      string // e.g. "default", "complex"
	Current   string // slot name the role currently points at
	ModelID   string // underlying model id of that slot (if OR-backed); empty otherwise
	HasPreset bool   // true when a "Suggest" preset is defined for this role
}

type uiSlot struct {
	Name    string
	ModelID string
}

type uiModel struct {
	ID              string
	PromptPrice     float64
	CompletionPrice float64
	ContextLength   int
	Vision          bool
	Tools           bool
	Reasoning       bool
	Free            bool
	Score           float64 // AA Intelligence Index
	CodingIndex     float64 // AA Coding Index
	AgenticIndex    float64 // AA Agentic Index
	SpeedTPS        float64 // median output tokens/sec
	TTFT            float64 // median time-to-first-token, seconds
	ThinkTime       float64 // TTFA - TTFT — time spent thinking before answer starts (reasoners only)
	AAPriceBlended  float64 // AA's reference blended 3:1 input/output price (USD / 1M)
	MarkupPct       float64 // (OR blended - AA blended) / AA blended × 100 — positive means OR charges more
	ValuePerDollar  float64 // quality / prompt price (role-specific in preset path; agent/$ in browse path)
}

type uiRouting struct {
	Roles    []uiRole
	AllSlots []string // all provider keys — the set of valid targets for a role
}

type uiFilters struct {
	Search            string
	Free              bool
	Vision            bool
	Tools             bool
	Reasoning         bool
	ActivePreset      string // role name when a preset is applied; empty otherwise
	PresetDescription string // human-readable summary for the banner
	ValueLeaderID     string // model id of the knee-point value leader (preset path only)
	ValueLeaderHint   string // e.g. "85% quality @ 30% price" (preset path only)
	Sort              string // active sort column: "prompt", "completion", "score", "context", "id"
	SortDir           string // "asc" or "desc"
}

type indexData struct {
	Routing uiRouting
	Slots   []uiSlot // OpenRouter-backed slots only (for per-model assign buttons)
	Models  []uiModel
	Filters uiFilters
}

// --- Handlers ---

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	data := s.buildIndexData(r)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewIndex, data); err != nil {
		s.logger.Error("render index", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	data := s.buildIndexData(r)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewModelsContent, data); err != nil {
		s.logger.Error("render models_content", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

func (s *Server) handleRouting(w http.ResponseWriter, r *http.Request) {
	data := s.buildRouting()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewRouting, data); err != nil {
		s.logger.Error("render routing", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// handleSlotAssign: POST /slots/{slot}/assign with body model_id=...
func (s *Server) handleSlotAssign(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	slot := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/slots/"), "/assign")
	if slot == "" || strings.Contains(slot, "/") {
		http.Error(w, "invalid slot", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "parse form", http.StatusBadRequest)
		return
	}
	modelID := r.FormValue("model_id")
	if modelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}

	caps := s.lookupCaps(r.Context(), modelID)
	if err := s.router.SetProviderModel(slot, modelID, caps); err != nil {
		s.logger.Warn("slot assign failed", "slot", slot, "model", modelID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.logger.Info("slot assigned", "slot", slot, "model", modelID)

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewRouting, s.buildRouting()); err != nil {
		s.logger.Error("render routing after assign", "err", err)
	}
}

// handleRoleSet: POST /routing/{role}/set with body slot=...
func (s *Server) handleRoleSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	role := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/routing/"), "/set")
	if role == "" || strings.Contains(role, "/") {
		http.Error(w, "invalid role", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "parse form", http.StatusBadRequest)
		return
	}
	slot := r.FormValue("slot")
	if slot == "" {
		http.Error(w, "slot required", http.StatusBadRequest)
		return
	}
	if err := s.router.SetRole(role, slot); err != nil {
		s.logger.Warn("role set failed", "role", role, "slot", slot, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.logger.Info("role assigned", "role", role, "slot", slot)

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewRouting, s.buildRouting()); err != nil {
		s.logger.Error("render routing after role set", "err", err)
	}
}

// handleRefresh triggers a fresh OpenRouter /models fetch (+ AA scores if configured) and re-caches caps.
func (s *Server) handleRefresh(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	apiKey := s.firstOpenRouterAPIKey()
	if apiKey == "" {
		http.Error(w, "no openrouter provider configured", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	caps, err := llm.FetchOpenRouterModels(ctx, apiKey)
	if err != nil {
		s.logger.Warn("refresh failed", "err", err)
		http.Error(w, "upstream fetch failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	// Overlay AA Intelligence Index scores if configured — always re-fetch on
	// manual Refresh (bypass cache) and update the stored cache.
	if aaKey := s.cfgRef.ArtificialAnalysisAPIKey; aaKey != "" {
		if models, aaErr := llm.FetchArtificialAnalysisData(ctx, aaKey); aaErr != nil {
			s.logger.Warn("AA data refresh failed", "err", aaErr)
		} else {
			llm.MergeAAScores(caps, models)
			s.logger.Info("AA data refreshed", "models", len(models))
			if s.settings != nil {
				if storeErr := llm.StoreAACache(ctx, s.settings, models); storeErr != nil {
					s.logger.Warn("AA cache store failed", "err", storeErr)
				}
			}
		}
	}

	if s.capStore != nil {
		for id, c := range caps {
			_ = s.capStore.PutCapabilities(ctx, "openrouter", id, c)
		}
	}
	s.logger.Info("openrouter catalog refreshed", "count", len(caps))

	data := s.buildIndexData(r)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewModelsContent, data); err != nil {
		s.logger.Error("render models after refresh", "err", err)
	}
}

// --- Data builders ---

func (s *Server) buildIndexData(r *http.Request) indexData {
	q := r.URL.Query()
	preset := q.Get("preset")

	ctx5, cancel5 := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel5()

	var allCaps map[string]llm.Capabilities
	if s.capStore != nil {
		allCaps, _ = s.capStore.GetAllCapabilities(ctx5, "openrouter")
	}

	// Load AA cache for extra columns (coding, agentic, speed, ttft).
	var aaModels map[string]llm.AAModelInfo
	if s.settings != nil {
		if cache, _ := llm.LoadAACache(ctx5, s.settings); cache != nil {
			aaModels = cache.Models
		}
	}

	// Preset path — pre-filter + pre-sort via the role's preset. Checkbox
	// filters are ignored on this path: the preset is a complete override.
	if p, ok := rolePresets[preset]; ok {
		models := applyPreset(allCaps, aaModels, preset)
		filters := uiFilters{
			ActivePreset:      preset,
			PresetDescription: p.Description,
			Tools:             requiresTools(preset),
			Vision:            preset == "multimodal",
			Reasoning:         preset == "complex",
		}
		// Knee-point value leader: best quality/price among frontier models
		// with quality ≥ 50% of top. Skipped when the top model is already
		// the value leader.
		axes := p.Axes
		if axes == nil && preset == "classifier" {
			axes = classifierAxes(models)
		}
		if axes != nil {
			if vl := valueLeader(models, axes, 0.5); vl != nil {
				topQ, topP := axes(models[0])
				vQ, vP := axes(*vl)
				if topQ > 0 && topP > 0 {
					filters.ValueLeaderID = vl.ID
					filters.ValueLeaderHint = fmt.Sprintf("%.0f%% quality @ %.0f%% price",
						100*vQ/topQ, 100*vP/topP)
				}
			}
		}
		return indexData{
			Routing: s.buildRouting(),
			Slots:   s.openRouterSlots(),
			Models:  models,
			Filters: filters,
		}
	}

	// Manual filter path — Tools defaults ON on initial page load (empty
	// query string), since the agentic loop requires tool calling. Once the
	// user submits the filters form, whatever they send wins.
	formSubmitted := len(q) > 0
	sortCol := q.Get("sort")
	if sortCol == "" {
		sortCol = "prompt"
	}
	sortDir := q.Get("dir")
	if sortDir != "asc" && sortDir != "desc" {
		if sortCol == "score" || sortCol == "context" {
			sortDir = "desc"
		} else {
			sortDir = "asc"
		}
	}
	f := uiFilters{
		Search:    strings.ToLower(strings.TrimSpace(q.Get("q"))),
		Free:      q.Get("free") != "",
		Vision:    q.Get("vision") != "",
		Tools:     q.Get("tools") != "" || !formSubmitted,
		Reasoning: q.Get("reasoning") != "",
		Sort:      sortCol,
		SortDir:   sortDir,
	}

	models := make([]uiModel, 0, len(allCaps))
	for id, c := range allCaps {
		free := c.Free()
		if f.Search != "" && !strings.Contains(strings.ToLower(id), f.Search) {
			continue
		}
		if f.Free && !free {
			continue
		}
		if f.Vision && !c.Vision {
			continue
		}
		if f.Tools && !c.Tools {
			continue
		}
		if f.Reasoning && !c.Reasoning {
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
			Free:            free,
			Score:           c.Score,
		}
		if aaModels != nil {
			if info := llm.LookupAAInfo(id, aaModels); info != nil {
				enrichFromAA(&m, *info)
			}
		}
		// Generic value metric for browse path: agentic index per $1/M prompt
		// tokens, falling back to intelligence index when agentic is absent.
		if m.PromptPrice > 0 {
			q := m.AgenticIndex
			if q == 0 {
				q = m.Score
			}
			if q > 0 {
				m.ValuePerDollar = q / m.PromptPrice
			}
		}
		models = append(models, m)
	}
	asc := sortDir == "asc"
	sort.Slice(models, func(i, j int) bool {
		var less bool
		switch sortCol {
		case "completion":
			less = models[i].CompletionPrice < models[j].CompletionPrice
		case "score":
			less = models[i].Score < models[j].Score
		case "coding":
			less = models[i].CodingIndex < models[j].CodingIndex
		case "agentic":
			less = models[i].AgenticIndex < models[j].AgenticIndex
		case "speed":
			less = models[i].SpeedTPS < models[j].SpeedTPS
		case "ttft":
			less = models[i].TTFT < models[j].TTFT
		case "think":
			less = models[i].ThinkTime < models[j].ThinkTime
		case "markup":
			less = models[i].MarkupPct < models[j].MarkupPct
		case "value":
			less = models[i].ValuePerDollar < models[j].ValuePerDollar
		case "context":
			less = models[i].ContextLength < models[j].ContextLength
		case "id":
			less = models[i].ID < models[j].ID
		default: // "prompt"
			if models[i].Free != models[j].Free {
				return models[i].Free // free always first regardless of direction
			}
			less = models[i].PromptPrice < models[j].PromptPrice
		}
		if asc {
			return less
		}
		return !less
	})

	return indexData{
		Routing: s.buildRouting(),
		Slots:   s.openRouterSlots(),
		Models:  models,
		Filters: f,
	}
}

// orBlendedPrice mirrors AA's 3:1 input/output weighting so prices are
// directly comparable across sources.
func orBlendedPrice(promptPrice, completionPrice float64) float64 {
	return (3*promptPrice + completionPrice) / 4
}

// enrichFromAA populates the AA-derived fields on a uiModel from the matched
// AAModelInfo record. Used by both the preset path and the browse path so
// they share one formula for Think time and Markup %.
func enrichFromAA(m *uiModel, aa llm.AAModelInfo) {
	m.CodingIndex = aa.CodingIndex
	m.AgenticIndex = aa.AgenticIndex
	m.SpeedTPS = aa.SpeedTPS
	m.TTFT = aa.TTFT
	if aa.TTFA > 0 && aa.TTFT > 0 && aa.TTFA >= aa.TTFT {
		m.ThinkTime = aa.TTFA - aa.TTFT
	}
	m.AAPriceBlended = aa.PriceBlended
	if aa.PriceBlended > 0 && m.PromptPrice > 0 {
		orBlended := orBlendedPrice(m.PromptPrice, m.CompletionPrice)
		m.MarkupPct = (orBlended - aa.PriceBlended) / aa.PriceBlended * 100
	}
}

// requiresTools returns true when the preset's filter requires tool-calling.
func requiresTools(preset string) bool {
	switch preset {
	case "simple", "default", "complex", "multimodal":
		return true
	}
	return false
}

func (s *Server) buildRouting() uiRouting {
	cfg := s.router.GetConfig()
	allSlots := s.router.ProviderNames()
	sort.Strings(allSlots)

	orSlots := map[string]string{} // slot → current model id
	for _, sl := range s.openRouterSlots() {
		orSlots[sl.Name] = sl.ModelID
	}

	roleCur := map[string]string{
		"simple":     cfg.Simple,
		"default":    cfg.Default,
		"complex":    cfg.Complex,
		"multimodal": cfg.Multimodal,
		"fallback":   cfg.Fallback,
		"classifier": cfg.Classifier,
		"compaction": cfg.Compaction,
	}
	order := []string{"simple", "default", "complex", "multimodal", "fallback", "classifier", "compaction"}
	presets := presetRoles()
	roles := make([]uiRole, 0, len(order))
	for _, r := range order {
		roles = append(roles, uiRole{
			Name:      r,
			Current:   roleCur[r],
			ModelID:   orSlots[roleCur[r]],
			HasPreset: presets[r],
		})
	}
	return uiRouting{Roles: roles, AllSlots: allSlots}
}

// openRouterSlots returns the subset of configured models whose provider is
// "openrouter" — these are the slots the admin UI lets you reassign per-model.
func (s *Server) openRouterSlots() []uiSlot {
	var out []uiSlot
	for name, mc := range s.cfgRef.Models {
		if mc.Provider != "openrouter" {
			continue
		}
		// Live model id may differ from config if runtime-swapped.
		modelID := mc.Model
		if p, ok := s.providerFor(name); ok {
			if cp, ok := p.(llm.ConfigurableProvider); ok {
				if cur := cp.CurrentModel(); cur != "" {
					modelID = cur
				}
			}
		}
		out = append(out, uiSlot{Name: name, ModelID: modelID})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (s *Server) providerFor(name string) (llm.Provider, bool) {
	return s.router.Provider(name)
}

func (s *Server) firstOpenRouterAPIKey() string {
	for _, mc := range s.cfgRef.Models {
		if mc.Provider == "openrouter" && mc.APIKey != "" {
			return mc.APIKey
		}
	}
	return ""
}

func (s *Server) lookupCaps(ctx context.Context, modelID string) llm.Capabilities {
	if s.capStore == nil {
		return llm.Capabilities{}
	}
	c, ok, err := s.capStore.GetCapabilities(ctx, "openrouter", modelID)
	if err != nil || !ok {
		return llm.Capabilities{}
	}
	return c
}
