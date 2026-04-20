package adminapi

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

// --- View data ---

type uiRole struct {
	Name           string       // e.g. "default", "complex"
	Current        string       // slot name the role currently points at
	Provider       string       // provider type backing that slot ("openrouter", "gemini", ...)
	ModelID        string       // live model id on that slot (from ConfigurableProvider)
	HasPreset      bool         // true when a "Suggest" preset is defined for this role
	AvailableSlots []uiSlotInfo // slots valid for this role (multimodal is Gemini+vision only)
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
	EffectivePrompt float64 // 0.9 × prompt + 0.1 × multimodal_slot.prompt for non-vision candidates under roles that route images elsewhere
	ValuePerDollar  float64 // quality / prompt price (role-specific in preset path; agent/$ in browse path)
}

type uiSlotInfo struct {
	Name     string // slot key in config (e.g. "default-or", "simple-gemini")
	Provider string // backend type: "openrouter", "gemini", "ollama", "claude-bridge", "local"
}

type uiRouting struct {
	Roles    []uiRole
	AllSlots []uiSlotInfo // all provider keys with their backend type
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
	ActiveTab       string // "routing" or "analytics" — drives tab highlight in layout
	Routing         uiRouting
	Slots           []uiSlot // slots backing the currently-browsed provider (for per-model assign buttons)
	Models          []uiModel
	Filters         uiFilters
	CatalogProvider string // "openrouter" | "gemini" — which catalog is shown in the models browser
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

func (s *Server) handleAnalytics(w http.ResponseWriter, r *http.Request) {
	// Analytics page is a thin frame — the usage section lazy-loads via /usage.
	// Still needs ActiveTab so the tab bar highlights the right entry.
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewAnalytics, map[string]any{"ActiveTab": "analytics"}); err != nil {
		s.logger.Error("render analytics", "err", err)
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

	// Provider type: prefer the form value (new UI — role/provider/model flow),
	// fall back to the slot's config default for the legacy model-card path.
	providerType := r.FormValue("provider")
	if providerType == "" {
		providerType = s.router.SlotProvider(slot)
	}
	if providerType == "" {
		if mc, ok := s.cfgRef.Models[slot]; ok {
			providerType = mc.Provider
		}
	}
	caps := s.lookupCapsFor(r.Context(), providerType, modelID)
	if err := s.router.SetProviderModel(slot, providerType, modelID, caps); err != nil {
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

// handleRefresh triggers fresh catalog fetches for OpenRouter (+ AA scores)
// and Gemini, then re-caches caps. Which catalog the UI shows afterwards is
// controlled by the ?provider query param (default: openrouter).
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

	// Also refresh Gemini catalog (best effort — failures don't abort the OR path).
	if geminiKey := s.firstGeminiAPIKey(); geminiKey != "" {
		if gCaps, gErr := llm.FetchGeminiModels(ctx, geminiKey); gErr != nil {
			s.logger.Warn("gemini refresh failed", "err", gErr)
		} else if s.capStore != nil {
			for id, c := range gCaps {
				_ = s.capStore.PutCapabilities(ctx, "gemini", id, c)
			}
			s.logger.Info("gemini catalog refreshed", "count", len(gCaps))
		}
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
	catalogProv := q.Get("provider")
	if catalogProv != "gemini" {
		catalogProv = "openrouter"
	}

	ctx5, cancel5 := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel5()

	var allCaps map[string]llm.Capabilities
	if s.capStore != nil {
		allCaps, _ = s.capStore.GetAllCapabilities(ctx5, catalogProv)
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
		// Find current multimodal slot's prompt price — passed to applyPreset
		// so non-vision candidates for default/simple/complex get an effective
		// price that accounts for image messages routing to multimodal.
		var visionFallbackPrompt float64
		cfg := s.router.GetConfig()
		for _, sl := range s.openRouterSlots() {
			if sl.Name == cfg.Multimodal {
				if c, ok := allCaps[sl.ModelID]; ok {
					visionFallbackPrompt = c.PromptPrice
				}
				break
			}
		}
		models := applyPreset(allCaps, aaModels, preset, visionFallbackPrompt)
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
			ActiveTab:       "routing",
			Routing:         s.buildRouting(),
			Slots:           s.allAssignableSlots(),
			Models:          models,
			Filters:         filters,
			CatalogProvider: catalogProv,
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
		case "effective":
			less = effectiveOrNominal(models[i]) < effectiveOrNominal(models[j])
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
		Routing:         s.buildRouting(),
		Slots:           s.allAssignableSlots(),
		Models:          models,
		Filters:         f,
		CatalogProvider: catalogProv,
	}
}

// orBlendedPrice mirrors AA's 3:1 input/output weighting so prices are
// directly comparable across sources.
func orBlendedPrice(promptPrice, completionPrice float64) float64 {
	return (3*promptPrice + completionPrice) / 4
}

// effectiveOrNominal returns EffectivePrompt when set, otherwise PromptPrice.
// Used by the "effective" sort case so models without the adjustment still
// rank by their nominal price.
func effectiveOrNominal(m uiModel) float64 {
	if m.EffectivePrompt > 0 {
		return m.EffectivePrompt
	}
	return m.PromptPrice
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
	allSlotNames := s.router.ProviderNames()
	sort.Strings(allSlotNames)

	// Build slot info with provider type from config.
	allSlots := make([]uiSlotInfo, 0, len(allSlotNames))
	for _, name := range allSlotNames {
		prov := ""
		if mc, ok := s.cfgRef.Models[name]; ok {
			prov = mc.Provider
		}
		allSlots = append(allSlots, uiSlotInfo{Name: name, Provider: prov})
	}

	// Model id lookup across all reconfigurable providers — used to show
	// "→ <model>" next to the current slot in the routing UI.
	slotModelID := map[string]string{}
	for _, sl := range s.allAssignableSlots() {
		slotModelID[sl.Name] = sl.ModelID
	}

	// Multimodal role: by business rule, only Gemini slots whose current
	// model supports vision are valid. Everything else would either fail or
	// silently degrade.
	multimodalAllowed := make([]uiSlotInfo, 0, len(allSlots))
	for _, sl := range allSlots {
		if sl.Provider != "gemini" {
			continue
		}
		if s.visionCapsForSlot(sl.Name).Vision {
			multimodalAllowed = append(multimodalAllowed, sl)
		}
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
		avail := allSlots
		if r == "multimodal" {
			avail = multimodalAllowed
		}
		cur := roleCur[r]
		roles = append(roles, uiRole{
			Name:           r,
			Current:        cur,
			Provider:       s.router.SlotProvider(cur),
			ModelID:        slotModelID[cur],
			HasPreset:      presets[r],
			AvailableSlots: avail,
		})
	}
	return uiRouting{Roles: roles, AllSlots: allSlots}
}

// openRouterSlots returns the subset of configured models whose provider is
// "openrouter" — kept for preset path (vision fallback lookup).
func (s *Server) openRouterSlots() []uiSlot {
	return s.slotsByProvider("openrouter")
}

// slotsByProvider returns all slots backed by the given provider type with
// their live model ids (reflecting runtime SetModel swaps).
func (s *Server) slotsByProvider(providerType string) []uiSlot {
	var out []uiSlot
	for name, mc := range s.cfgRef.Models {
		if mc.Provider != providerType {
			continue
		}
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

// allAssignableSlots returns every configured LLM slot (excluding embedding
// providers) with its LIVE model id — regardless of provider type. Used by
// the model-catalog assign buttons so users can retarget any slot to any
// backend via the cross-type swap flow.
func (s *Server) allAssignableSlots() []uiSlot {
	var out []uiSlot
	for name, mc := range s.cfgRef.Models {
		if mc.Provider == "hf-tei" || mc.Provider == "openai" || mc.Provider == "" {
			continue
		}
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

// visionCapsForSlot returns the live capabilities of the slot's current model.
// Used by buildRouting to filter the multimodal dropdown.
func (s *Server) visionCapsForSlot(slot string) llm.Capabilities {
	if p, ok := s.providerFor(slot); ok {
		if vp, ok := p.(llm.VisionProvider); ok {
			// VisionProvider only exposes a bool — we need full caps.
			_ = vp
		}
		if cp, ok := p.(interface{ Capabilities() llm.Capabilities }); ok {
			return cp.Capabilities()
		}
	}
	return llm.Capabilities{}
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

func (s *Server) firstGeminiAPIKey() string {
	for _, mc := range s.cfgRef.Models {
		if mc.Provider == "gemini" && mc.APIKey != "" {
			return mc.APIKey
		}
	}
	return ""
}

// --- MCP page ---

type uiMCPRow struct {
	Name       string
	URL        string
	Headers    string // JSON-serialised, one per line, for display/edit in a textarea
	AllowTools string // comma-separated
	DenyTools  string // comma-separated
	Type       string
}

type uiMCPData struct {
	ActiveTab    string
	Servers      []uiMCPRow
	BridgeExport string // MCP_BRIDGE_EXPORT_PATH if set, shown as a banner note
	SavedName    string // non-empty after a successful save — used by template to flash a success message
}

// loadMCPForUI returns the current MCP list flattened into UI rows. Prefers
// the DB list; falls back to the legacy mcp.json file so users landing on
// the page for the first time see their existing config instead of an
// empty table.
func (s *Server) loadMCPForUI(ctx context.Context) map[string]config.MCPServerConfig {
	if servers, found, _ := LoadMCPServersFromSettings(ctx, s.settings); found {
		return servers
	}
	// File fallback — best effort.
	if servers, err := config.LoadMCPServers("config/mcp.json"); err == nil {
		return servers
	}
	return nil
}

func mcpToRows(servers map[string]config.MCPServerConfig) []uiMCPRow {
	names := sortedMCPServerNames(servers)
	out := make([]uiMCPRow, 0, len(names))
	for _, n := range names {
		sv := servers[n]
		hdrLines := make([]string, 0, len(sv.Headers))
		hdrKeys := make([]string, 0, len(sv.Headers))
		for k := range sv.Headers {
			hdrKeys = append(hdrKeys, k)
		}
		sort.Strings(hdrKeys)
		for _, k := range hdrKeys {
			hdrLines = append(hdrLines, k+": "+sv.Headers[k])
		}
		out = append(out, uiMCPRow{
			Name:       n,
			URL:        sv.URL,
			Type:       sv.Type,
			Headers:    strings.Join(hdrLines, "\n"),
			AllowTools: strings.Join(sv.AllowTools, ","),
			DenyTools:  strings.Join(sv.DenyTools, ","),
		})
	}
	return out
}

func (s *Server) handleMCP(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	data := uiMCPData{
		ActiveTab:    "mcp",
		Servers:      mcpToRows(s.loadMCPForUI(ctx)),
		BridgeExport: os.Getenv("MCP_BRIDGE_EXPORT_PATH"),
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewMCP, data); err != nil {
		s.logger.Error("render mcp", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// parseHeadersText splits "Key: value" lines into a map. Blank lines are
// ignored; malformed lines are silently dropped so a UI save with partial
// typing doesn't wipe the entry.
func parseHeadersText(text string) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		if k == "" {
			continue
		}
		out[k] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// splitCSV turns "a,b, c" into ["a","b","c"], trimming and dropping blanks.
func splitCSV(s string) []string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// handleMCPSet: POST /mcp/{name}/set with body url, headers, allow_tools,
// deny_tools, type. Upserts the named server in the DB-backed list. Writes
// the bridge mirror file on success.
func (s *Server) handleMCPSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/mcp/"), "/set")
	name = strings.TrimSpace(name)
	if name == "" || strings.ContainsAny(name, "/ \t\n") {
		http.Error(w, "invalid server name", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "parse form", http.StatusBadRequest)
		return
	}
	if s.settings == nil {
		http.Error(w, "settings store not available", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	current := s.loadMCPForUI(ctx)
	if current == nil {
		current = map[string]config.MCPServerConfig{}
	}
	url := strings.TrimSpace(r.FormValue("url"))
	if url == "" {
		http.Error(w, "url required", http.StatusBadRequest)
		return
	}
	current[name] = config.MCPServerConfig{
		Type:       strings.TrimSpace(r.FormValue("type")),
		URL:        url,
		Headers:    parseHeadersText(r.FormValue("headers")),
		AllowTools: splitCSV(r.FormValue("allow_tools")),
		DenyTools:  splitCSV(r.FormValue("deny_tools")),
	}
	if err := saveMCPServers(ctx, s.settings, current); err != nil {
		s.logger.Warn("mcp save failed", "name", name, "err", err)
		http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	s.logger.Info("mcp server updated", "name", name, "url", url)
	// Re-render the page so the new row shows up.
	data := uiMCPData{
		ActiveTab:    "mcp",
		Servers:      mcpToRows(current),
		BridgeExport: os.Getenv("MCP_BRIDGE_EXPORT_PATH"),
		SavedName:    name,
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewMCP, data); err != nil {
		s.logger.Error("render mcp after save", "err", err)
	}
}

// handleMCPDelete: POST /mcp/{name}/delete. Removes the named server.
func (s *Server) handleMCPDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	name := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/mcp/"), "/delete")
	name = strings.TrimSpace(name)
	if name == "" {
		http.Error(w, "name required", http.StatusBadRequest)
		return
	}
	if s.settings == nil {
		http.Error(w, "settings store not available", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	current := s.loadMCPForUI(ctx)
	if _, ok := current[name]; !ok {
		http.Error(w, "unknown server: "+name, http.StatusNotFound)
		return
	}
	delete(current, name)
	if err := saveMCPServers(ctx, s.settings, current); err != nil {
		s.logger.Warn("mcp delete failed", "name", name, "err", err)
		http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	s.logger.Info("mcp server deleted", "name", name)
	data := uiMCPData{
		ActiveTab:    "mcp",
		Servers:      mcpToRows(current),
		BridgeExport: os.Getenv("MCP_BRIDGE_EXPORT_PATH"),
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = render(w, viewMCP, data)
}

// --- Settings page ---

// settingSpec describes one user-editable scalar for the Settings tab.
// Stays in the handlers file so adding a new setting is a one-liner.
type settingSpec struct {
	Key         string // kv_settings key (e.g. llm.SettingKeyClassifierTimeout)
	Label       string
	Description string
	Default     string // shown in UI placeholder + as "default: X" hint
	InputType   string // HTML <input type>; "number" for ints, "text" otherwise
	Value       string // current DB value, empty if unset
}

// settingSpecs returns the list of scalars exposed in the Settings tab, in
// display order. Extend this slice to surface more config keys.
func (s *Server) settingSpecs() []settingSpec {
	boolDefault := func(v bool) string {
		if v {
			return "true"
		}
		return "false"
	}
	defs := []settingSpec{
		{
			Key:         llm.SettingKeyClassifierTimeout,
			Label:       "Classifier timeout (seconds)",
			Description: "Max time to wait for the classifier model per request before defaulting to level 2.",
			Default:     fmt.Sprintf("%d", s.cfgRef.Routing.ClassifierTimeout),
			InputType:   "number",
		},
		{
			Key:         llm.SettingKeyToolFilterTopK,
			Label:       "Tool filter top-K",
			Description: "Number of most-relevant MCP tools to include per request (0 = disabled). Applies on next restart.",
			Default:     fmt.Sprintf("%d", s.cfgRef.ToolFilter.TopK),
			InputType:   "number",
		},
		{
			Key:         llm.SettingKeyToolFilterAlwaysIncludeKeywords,
			Label:       "Tool filter always-include keywords",
			Description: "Comma-separated substrings. Any MCP tool whose name contains one (case-insensitive) is always included, bypassing top-K. Example: get, list, recall, search, read. Applies on next restart.",
			Default:     strings.Join(s.cfgRef.ToolFilter.AlwaysIncludeKeywords, ","),
			InputType:   "text",
		},
		// Feature flags — changes apply on next restart.
		{
			Key:         llm.SettingKeyWebSearchEnabled,
			Label:       "Web search enabled",
			Description: "Enables the web_search tool. Use 'true' or 'false'. Applies on next restart.",
			Default:     boolDefault(s.cfgRef.WebSearch.Enabled),
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyWebSearchProvider,
			Label:       "Web search provider",
			Description: "Backend for web_search: 'tavily' (free 1000/mo) or 'ollama' (legacy). Applies on next restart.",
			Default:     s.cfgRef.WebSearch.Provider,
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyWebFetchEnabled,
			Label:       "Web fetch enabled",
			Description: "Enables the web_fetch tool. Use 'true' or 'false'. Applies on next restart.",
			Default:     boolDefault(s.cfgRef.WebFetch.Enabled),
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyFilesystemEnabled,
			Label:       "Filesystem tools enabled",
			Description: "Enables read/write access to the mounted /assistant_context volume. Applies on next restart.",
			Default:     boolDefault(s.cfgRef.Filesystem.Enabled),
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyTTSEnabled,
			Label:       "TTS enabled",
			Description: "Enables Edge TTS voice replies. Applies on next restart.",
			Default:     boolDefault(s.cfgRef.TTS.Enabled),
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyTTSVoice,
			Label:       "TTS voice",
			Description: "Edge TTS voice identifier (e.g. ru-RU-DmitryNeural, en-US-EmmaMultilingualNeural). Applies on next restart.",
			Default:     s.cfgRef.TTS.Voice,
			InputType:   "text",
		},
		{
			Key:         llm.SettingKeyVoiceAPIChatID,
			Label:       "Voice API chat_id",
			Description: "Telegram chat_id for voice API (Atom Echo etc). 0 keeps the config default.",
			Default:     fmt.Sprintf("%d", s.cfgRef.VoiceAPI.ChatID),
			InputType:   "number",
		},
		{
			Key:         llm.SettingKeyTrustForwardAuth,
			Label:       "Trust forward-auth header",
			Description: "Admin API trusts X-authentik-username (via Traefik). Disable for local dev. Applies on next restart.",
			Default:     boolDefault(s.cfgRef.AdminAPI.TrustForwardAuth),
			InputType:   "text",
		},
	}
	if s.settings != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		for i := range defs {
			if v, ok, _ := s.settings.GetSetting(ctx, defs[i].Key); ok {
				defs[i].Value = v
			}
		}
	}
	return defs
}

func (s *Server) handleSettings(w http.ResponseWriter, r *http.Request) {
	data := map[string]any{
		"ActiveTab": "settings",
		"Settings":  s.settingSpecs(),
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewSettings, data); err != nil {
		s.logger.Error("render settings", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// handleSettingSet: POST /settings/{key}/set with body value=...
// Validates the key against the allowlist derived from settingSpecs so an
// attacker can't smuggle arbitrary kv_settings keys through this endpoint.
func (s *Server) handleSettingSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	key := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/settings/"), "/set")
	allowed := false
	for _, sp := range s.settingSpecs() {
		if sp.Key == key {
			allowed = true
			break
		}
	}
	if !allowed {
		http.Error(w, "unknown setting", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "parse form", http.StatusBadRequest)
		return
	}
	value := strings.TrimSpace(r.FormValue("value"))
	if s.settings == nil {
		http.Error(w, "settings store not available", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if value == "" {
		// Empty string = revert to config default. Delete the DB key so the
		// GetIntSetting fallback kicks in.
		if err := s.settings.PutSetting(ctx, key, ""); err != nil {
			s.logger.Warn("setting delete failed", "key", key, "err", err)
			http.Error(w, "save failed", http.StatusInternalServerError)
			return
		}
	} else if err := s.settings.PutSetting(ctx, key, value); err != nil {
		s.logger.Warn("setting save failed", "key", key, "err", err)
		http.Error(w, "save failed", http.StatusInternalServerError)
		return
	}
	s.logger.Info("setting updated", "key", key, "value", value)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(`<span class="save-status ok">Saved</span>`)) //nolint:errcheck
}

// handlePrompts: GET /prompts — full page with current prompt values from DB.
func (s *Server) handlePrompts(w http.ResponseWriter, r *http.Request) {
	type promptsData struct {
		ActiveTab        string
		SystemPrompt     string
		ClassifierPrompt string
	}
	data := promptsData{ActiveTab: "prompts"}
	if s.settings != nil {
		if v, ok, _ := s.settings.GetSetting(r.Context(), "prompts.system"); ok {
			data.SystemPrompt = v
		}
		if v, ok, _ := s.settings.GetSetting(r.Context(), "prompts.classifier"); ok {
			data.ClassifierPrompt = v
		}
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewPrompts, data); err != nil {
		s.logger.Error("render prompts", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

// handlePromptSet: POST /prompts/{key}/set with body value=...
// key must be "system" or "classifier".
func (s *Server) handlePromptSet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	key := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/prompts/"), "/set")
	if key != "system" && key != "classifier" {
		http.Error(w, "unknown prompt key", http.StatusBadRequest)
		return
	}
	if err := r.ParseForm(); err != nil {
		http.Error(w, "parse form", http.StatusBadRequest)
		return
	}
	value := r.FormValue("value")
	if s.settings == nil {
		http.Error(w, "settings store not available", http.StatusServiceUnavailable)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	if err := s.settings.PutSetting(ctx, "prompts."+key, value); err != nil {
		s.logger.Warn("prompt save failed", "key", key, "err", err)
		http.Error(w, "save failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	s.logger.Info("prompt updated", "key", key, "len", len(value))
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(`<span class="save-status ok">Saved</span>`)) //nolint:errcheck
}

func (s *Server) lookupCaps(ctx context.Context, modelID string) llm.Capabilities {
	return s.lookupCapsFor(ctx, "openrouter", modelID)
}

func (s *Server) lookupCapsFor(ctx context.Context, providerType, modelID string) llm.Capabilities {
	if s.capStore == nil || providerType == "" {
		return llm.Capabilities{}
	}
	c, ok, err := s.capStore.GetCapabilities(ctx, providerType, modelID)
	if err != nil || !ok {
		return llm.Capabilities{}
	}
	return c
}
