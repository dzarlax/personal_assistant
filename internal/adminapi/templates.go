package adminapi

import (
	"embed"
	"fmt"
	"html/template"
	"io"
)

//go:embed templates
var templatesFS embed.FS

// View names — must match template file basenames under templates/.
const (
	viewIndex         = "index"
	viewRouting       = "routing"
	viewModelsBrowser = "models_browser"
	viewModelsContent = "models_content"
	viewUsage         = "usage"
	viewAnalytics     = "analytics"
)

// tmpls is the parsed template set. Entries are keyed by view name and
// render the full payload for that view (using shared partials).
var tmpls = func() map[string]*template.Template {
	out := map[string]*template.Template{}
	// Shared partials are parsed into each view so {{template "..."}} calls work.
	partials := []string{
		"templates/partials_layout.html",
		"templates/partials_routing.html",
		"templates/partials_models_row.html",
		"templates/partials_models_browser.html",
	}
	// Each view's "main" template name must match a {{define "<name>"}} block
	// inside the listed file. models_browser is defined inside the partial,
	// so no extra file is needed.
	views := map[string]string{
		viewIndex:         "templates/index.html",
		viewRouting:       "templates/routing.html",
		viewModelsBrowser: "templates/partials_models_browser.html",
		viewModelsContent: "templates/partials_models_browser.html",
		viewUsage:         "templates/usage.html",
		viewAnalytics:     "templates/analytics.html",
	}
	funcs := template.FuncMap{
		"priceUSD": func(v float64) string {
			if v == 0 {
				return "free"
			}
			if v < 0.01 {
				return fmt.Sprintf("$%.4f/M", v)
			}
			return fmt.Sprintf("$%.2f/M", v)
		},
		"ctxShort": func(n int) string {
			switch {
			case n >= 1_000_000:
				return fmt.Sprintf("%dM", n/1_000_000)
			case n >= 1000:
				return fmt.Sprintf("%dk", n/1000)
			case n > 0:
				return fmt.Sprintf("%d", n)
			}
			return "—"
		},
		"priceFmt": priceFmt,
		"intFmt":   intFmt,
		"add":      func(a, b int) int { return a + b },
		"sub":      func(a, b int) int { return a - b },
		"tokenShort": func(n int) string {
			switch {
			case n >= 1_000_000:
				return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
			case n >= 1000:
				return fmt.Sprintf("%.1fk", float64(n)/1000)
			case n > 0:
				return fmt.Sprintf("%d", n)
			}
			return "—"
		},
	}
	for name, main := range views {
		files := append([]string{main}, partials...)
		t := template.Must(template.New(name).Funcs(funcs).ParseFS(templatesFS, files...))
		out[name] = t
	}
	return out
}()

// render renders the named view to w. Returns the error instead of logging so
// handlers can pick the right response code.
func render(w io.Writer, view string, data any) error {
	t, ok := tmpls[view]
	if !ok {
		return fmt.Errorf("unknown view: %s", view)
	}
	return t.Execute(w, data)
}
