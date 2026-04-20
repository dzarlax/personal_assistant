package adminapi

import (
	"context"
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
	Score           float64 // Artificial Analysis Intelligence Index (0 = unknown)
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
	data := s.buildIndexData(r) // reuses model filtering + slots
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewModelsBrowser, data); err != nil {
		s.logger.Error("render models_table", "err", err)
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

	// Overlay AA Intelligence Index scores if configured.
	if aaKey := s.cfgRef.ArtificialAnalysisAPIKey; aaKey != "" {
		if scores, aaErr := llm.FetchArtificialAnalysisScores(ctx, aaKey); aaErr != nil {
			s.logger.Warn("AA scores refresh failed", "err", aaErr)
		} else {
			llm.MergeAAScores(caps, scores)
			s.logger.Info("AA scores refreshed", "count", len(scores))
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
	if err := render(w, viewModelsBrowser, data); err != nil {
		s.logger.Error("render models after refresh", "err", err)
	}
}

// --- Data builders ---

func (s *Server) buildIndexData(r *http.Request) indexData {
	q := r.URL.Query()
	preset := q.Get("preset")

	var allCaps map[string]llm.Capabilities
	if s.capStore != nil {
		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		allCaps, _ = s.capStore.GetAllCapabilities(ctx, "openrouter")
	}

	// Preset path — pre-filter + pre-sort via the role's preset. Checkbox
	// filters are ignored on this path: the preset is a complete override.
	if p, ok := rolePresets[preset]; ok {
		return indexData{
			Routing: s.buildRouting(),
			Slots:   s.openRouterSlots(),
			Models:  applyPreset(allCaps, preset),
			Filters: uiFilters{
				ActivePreset:      preset,
				PresetDescription: p.Description,
				// reflect preset intent in the checkboxes so user can tell what's on
				Tools:     requiresTools(preset),
				Vision:    preset == "multimodal",
				Reasoning: preset == "complex",
			},
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
		models = append(models, uiModel{
			ID:              id,
			PromptPrice:     c.PromptPrice,
			CompletionPrice: c.CompletionPrice,
			ContextLength:   c.ContextLength,
			Vision:          c.Vision,
			Tools:           c.Tools,
			Reasoning:       c.Reasoning,
			Free:            free,
			Score:           c.Score,
		})
	}
	asc := sortDir == "asc"
	sort.Slice(models, func(i, j int) bool {
		var less bool
		switch sortCol {
		case "completion":
			less = models[i].CompletionPrice < models[j].CompletionPrice
		case "score":
			less = models[i].Score < models[j].Score
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
