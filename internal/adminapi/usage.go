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

// usageView is the data passed to the usage template. Populated by handleUsage
// from llm.UsageStore aggregations over three time windows.
type usageView struct {
	Period         string // "24h" / "7d" / "30d" — echoed back to keep the UI in sync
	Since          time.Time
	Totals         llm.UsageTotals
	TotalsToday    llm.UsageTotals // always computed for the forecast card, independent of Period
	Totals7d       llm.UsageTotals // always computed for the forecast card
	Totals30d      llm.UsageTotals
	CacheHitPct    float64 // 100 * cached_prompt / prompt
	ReasoningPct   float64 // 100 * reasoning / completion
	ForecastUSD    float64 // 30-day run-rate projected from 7d trailing
	ByModel        []llm.UsageModelRow
	ByRole         []llm.UsageRoleRow
	ExpensiveTurns []expensiveTurnView
	DailyChart     dailyChart
}

// expensiveTurnView trims the raw ExpensiveTurn fields for display — questions
// and answers get truncated to keep the table readable.
type expensiveTurnView struct {
	Ts         string
	ChatID     int64
	Role       string
	ModelID    string
	CostUSD    float64
	Tokens     int
	Question   string // first line, up to 200 chars
	Answer     string // first line, up to 200 chars
}

// dailyChart describes a single combined SVG with two panels (calls on top,
// cost below) sharing the same X-axis slot layout. All coordinates are
// pre-computed so the template stays math-free. The SVG viewBox is fixed-width
// (520 units) but renders at 100% CSS width so it scales responsively.
type dailyChart struct {
	Width       int
	TotalHeight int // combined SVG height covering both panels + x-axis row
	// Panel 1 — calls (top)
	P1Top    int
	P1Bot    int
	// Panel 2 — cost (bottom)
	P2Top    int
	P2Bot    int
	// Separator between panels
	SepY int
	PlotLeft  int // x-coordinate of the Y-axis line (shared by both panels)
	PlotRight int
	Days      int
	// Calls series
	CallsBars  []chartBar
	CallsYAxis []yTick
	MaxCalls   int
	// Cost series
	CostBars  []chartBar
	CostYAxis []yTick
	MaxCost   float64
	// X-axis labels at the very bottom (shared)
	XAxisTags []xAxisTag
}

type chartBar struct {
	X        int
	W        int
	Y        int
	H        int
	DayLabel string
	// Tooltip content — both metrics shown on hover so users don't need to
	// cross-reference.
	Calls   int
	CostUSD float64
}

type yTick struct {
	Y     int
	Label string
}

type xAxisTag struct {
	X     int
	Label string
}

func (s *Server) handleUsage(w http.ResponseWriter, r *http.Request) {
	if s.usageStore == nil {
		http.Error(w, "usage store not configured", http.StatusServiceUnavailable)
		return
	}
	period := r.URL.Query().Get("period")
	since, label := resolvePeriod(period)

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	view, err := s.buildUsageView(ctx, since, label)
	if err != nil {
		s.logger.Error("usage: aggregation failed", "err", err)
		http.Error(w, "aggregation failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := render(w, viewUsage, view); err != nil {
		s.logger.Error("usage: render failed", "err", err)
		http.Error(w, "render error", http.StatusInternalServerError)
	}
}

func (s *Server) buildUsageView(ctx context.Context, since time.Time, label string) (usageView, error) {
	now := time.Now()
	v := usageView{Period: label, Since: since}

	// Primary period — drives the main table view.
	totals, err := s.usageStore.UsageTotals(ctx, since)
	if err != nil {
		return v, err
	}
	v.Totals = totals

	// Additional windows for the forecast/summary cards. Always the same three
	// windows regardless of active period, so the user gets a consistent glance.
	if v.TotalsToday, err = s.usageStore.UsageTotals(ctx, now.Add(-24*time.Hour)); err != nil {
		return v, err
	}
	if v.Totals7d, err = s.usageStore.UsageTotals(ctx, now.Add(-7*24*time.Hour)); err != nil {
		return v, err
	}
	if v.Totals30d, err = s.usageStore.UsageTotals(ctx, now.Add(-30*24*time.Hour)); err != nil {
		return v, err
	}
	// Monthly run-rate = 7d cost × (30/7).
	v.ForecastUSD = v.Totals7d.CostUSD * 30.0 / 7.0

	if v.Totals.PromptTokens > 0 {
		v.CacheHitPct = 100 * float64(v.Totals.CachedPromptTokens) / float64(v.Totals.PromptTokens)
	}
	if v.Totals.CompletionTokens > 0 {
		v.ReasoningPct = 100 * float64(v.Totals.ReasoningTokens) / float64(v.Totals.CompletionTokens)
	}

	byModel, err := s.usageStore.UsageByModel(ctx, since, 15)
	if err != nil {
		return v, err
	}
	v.ByModel = byModel

	byRole, err := s.usageStore.UsageByRole(ctx, since)
	if err != nil {
		return v, err
	}
	v.ByRole = byRole

	// Expensive turns — always from the primary period.
	turns, err := s.usageStore.ExpensiveTurns(ctx, since, 10)
	if err != nil {
		return v, err
	}
	for _, t := range turns {
		v.ExpensiveTurns = append(v.ExpensiveTurns, expensiveTurnView{
			Ts:       t.Ts.Format("2006-01-02 15:04"),
			ChatID:   t.ChatID,
			Role:     t.Role,
			ModelID:  t.ModelID,
			CostUSD:  t.CostUSD,
			Tokens:   t.Tokens,
			Question: truncText(firstLine(t.Question), 200),
			Answer:   truncText(firstLine(t.Answer), 200),
		})
	}

	// Daily chart — always from 30 days regardless of active period, so the
	// shape is comparable across period toggles.
	buckets, err := s.usageStore.UsageByDay(ctx, now.Add(-30*24*time.Hour))
	if err != nil {
		return v, err
	}
	v.DailyChart = buildDailyChart(buckets)

	return v, nil
}

// resolvePeriod converts "24h" / "7d" / "30d" (default "7d") into a time
// window + the label echoed in the UI.
func resolvePeriod(period string) (time.Time, string) {
	now := time.Now()
	switch period {
	case "24h":
		return now.Add(-24 * time.Hour), "24h"
	case "30d":
		return now.Add(-30 * 24 * time.Hour), "30d"
	default:
		return now.Add(-7 * 24 * time.Hour), "7d"
	}
}

// buildDailyChart produces a single combined SVG with two panels (calls on
// top, cost below) sharing the same X-axis slot layout. The SVG viewBox is
// 520 units wide and scales to 100% CSS width for responsiveness.
func buildDailyChart(buckets []llm.UsageDayBucket) dailyChart {
	const (
		width      = 520
		padLeft    = 52 // room for Y-axis labels
		padRight   = 10
		p1Top      = 12  // panel 1 plot area top
		p1Bot      = 76  // panel 1 plot area bottom
		sepY       = 90  // separator line between panels
		p2Top      = 102 // panel 2 plot area top
		p2Bot      = 166 // panel 2 plot area bottom
		totalH     = 184 // total SVG height (includes x-axis labels row)
	)
	c := dailyChart{
		Width:       width,
		TotalHeight: totalH,
		P1Top:       p1Top,
		P1Bot:       p1Bot,
		SepY:        sepY,
		P2Top:       p2Top,
		P2Bot:       p2Bot,
		PlotLeft:    padLeft,
		PlotRight:   width - padRight,
		Days:        len(buckets),
	}
	if len(buckets) == 0 {
		return c
	}
	sort.Slice(buckets, func(i, j int) bool { return buckets[i].Day.Before(buckets[j].Day) })

	for _, b := range buckets {
		if b.Calls > c.MaxCalls {
			c.MaxCalls = b.Calls
		}
		if b.CostUSD > c.MaxCost {
			c.MaxCost = b.CostUSD
		}
	}
	if c.MaxCalls == 0 {
		c.MaxCalls = 1
	}
	if c.MaxCost == 0 {
		c.MaxCost = 0.0001
	}

	plotW := c.PlotRight - c.PlotLeft
	p1H := p1Bot - p1Top
	p2H := p2Bot - p2Top
	slot := float64(plotW) / float64(len(buckets))
	barW := int(slot * 0.6)
	if barW < 2 {
		barW = 2
	}
	if barW > 32 {
		barW = 32
	}

	for i, b := range buckets {
		slotCenter := c.PlotLeft + int((float64(i)+0.5)*slot)
		barX := slotCenter - barW/2
		dayLabel := b.Day.Format("01-02")

		callsH := int(float64(b.Calls) / float64(c.MaxCalls) * float64(p1H))
		if callsH < 0 {
			callsH = 0
		}
		c.CallsBars = append(c.CallsBars, chartBar{
			X: barX, W: barW,
			Y: p1Bot - callsH, H: callsH,
			DayLabel: dayLabel, Calls: b.Calls, CostUSD: b.CostUSD,
		})

		costH := int(b.CostUSD / c.MaxCost * float64(p2H))
		if costH < 0 {
			costH = 0
		}
		c.CostBars = append(c.CostBars, chartBar{
			X: barX, W: barW,
			Y: p2Bot - costH, H: costH,
			DayLabel: dayLabel, Calls: b.Calls, CostUSD: b.CostUSD,
		})
	}

	c.CallsYAxis = []yTick{
		{Y: p1Bot, Label: "0"},
		{Y: p1Top + p1H/2, Label: intFmt(c.MaxCalls / 2)},
		{Y: p1Top, Label: intFmt(c.MaxCalls)},
	}
	c.CostYAxis = []yTick{
		{Y: p2Bot, Label: "$0"},
		{Y: p2Top + p2H/2, Label: priceFmt(c.MaxCost / 2)},
		{Y: p2Top, Label: priceFmt(c.MaxCost)},
	}

	// X-axis: label every day up to 15 days; above that switch to sparse
	// (first, every 5th, last) to avoid crowding.
	n := len(buckets)
	xLabel := func(i int) {
		sc := c.PlotLeft + int((float64(i)+0.5)*slot)
		c.XAxisTags = append(c.XAxisTags, xAxisTag{X: sc, Label: buckets[i].Day.Format("01/02")})
	}
	if n <= 15 {
		for i := range buckets {
			xLabel(i)
		}
	} else {
		xLabel(0)
		for i := 4; i < n-1; i += 5 {
			xLabel(i)
		}
		xLabel(n - 1)
	}
	return c
}

// firstLine returns the first non-empty line of s, trimmed.
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			return t
		}
	}
	return strings.TrimSpace(s)
}

// truncText caps a string at n runes, appending an ellipsis when truncated.
func truncText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// priceFmt formats a USD cost with more precision when small.
func priceFmt(v float64) string {
	switch {
	case v == 0:
		return "—"
	case v < 0.01:
		return fmt.Sprintf("$%.4f", v)
	case v < 1:
		return fmt.Sprintf("$%.3f", v)
	default:
		return fmt.Sprintf("$%.2f", v)
	}
}

// intFmt adds thousand separators for large numbers.
func intFmt(n int) string {
	s := fmt.Sprintf("%d", n)
	if n < 1000 {
		return s
	}
	var b strings.Builder
	rem := len(s) % 3
	if rem > 0 {
		b.WriteString(s[:rem])
		if len(s) > rem {
			b.WriteString(",")
		}
	}
	for i := rem; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteString(",")
		}
	}
	return b.String()
}
