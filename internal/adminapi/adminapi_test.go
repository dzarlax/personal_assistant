package adminapi

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

func newTestServer(t *testing.T) *Server {
	t.Helper()
	providers := map[string]llm.Provider{}
	router := llm.NewRouter(providers, llm.RouterConfig{})
	cfg := &config.Config{
		Models: config.ModelsConfig{},
	}
	return New(config.AdminAPIConfig{
		Enabled:          true,
		Listen:           ":0",
		Token:            "secret-token",
		TrustForwardAuth: false,
	}, router, nil, cfg, slog.Default())
}

// TestTemplatesParse verifies every registered view renders without panicking
// against minimal data. Catches template syntax errors at test time.
func TestTemplatesParse(t *testing.T) {
	data := indexData{
		Routing: uiRouting{
			Roles:    []uiRole{{Name: "default", Current: "workhorse", ModelID: "deepseek/v3.1"}},
			AllSlots: []string{"workhorse", "gemini-flash-lite"},
		},
		Slots:  []uiSlot{{Name: "workhorse", ModelID: "deepseek/v3.1"}},
		Models: []uiModel{{ID: "anthropic/claude", PromptPrice: 3.0, CompletionPrice: 15.0, ContextLength: 200000, Vision: true, Tools: true}},
	}
	cases := map[string]any{
		viewIndex:       data,
		viewRouting:     data.Routing, // routing view takes uiRouting directly
		viewModelsTable: data,
	}
	for v, d := range cases {
		t.Run(v, func(t *testing.T) {
			var buf bytes.Buffer
			if err := render(&buf, v, d); err != nil {
				t.Fatalf("render %s: %v", v, err)
			}
			if buf.Len() == 0 {
				t.Errorf("view %s rendered empty", v)
			}
		})
	}
}

// TestAuthRequired covers the four auth paths + the 401 default.
func TestAuthRequired(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	s.registerRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	tests := []struct {
		name     string
		prepare  func(*http.Request)
		wantCode int
	}{
		{"no auth", func(r *http.Request) {}, http.StatusUnauthorized},
		{"wrong bearer", func(r *http.Request) { r.Header.Set("Authorization", "Bearer wrong") }, http.StatusUnauthorized},
		{"good bearer", func(r *http.Request) { r.Header.Set("Authorization", "Bearer secret-token") }, http.StatusOK},
		{"good cookie", func(r *http.Request) { r.AddCookie(&http.Cookie{Name: authCookieName, Value: "secret-token"}) }, http.StatusOK},
		{"wrong cookie", func(r *http.Request) { r.AddCookie(&http.Cookie{Name: authCookieName, Value: "wrong"}) }, http.StatusUnauthorized},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", srv.URL+"/", nil)
			tc.prepare(req)
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			if resp.StatusCode != tc.wantCode {
				t.Errorf("got %d, want %d", resp.StatusCode, tc.wantCode)
			}
		})
	}
}

// TestAuthTokenQueryBootstrap: ?token=... sets cookie and redirects.
func TestAuthTokenQueryBootstrap(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	s.registerRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	client := &http.Client{
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	resp, err := client.Get(srv.URL + "/?token=secret-token")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusSeeOther {
		t.Errorf("expected 303, got %d", resp.StatusCode)
	}
	var found bool
	for _, c := range resp.Cookies() {
		if c.Name == authCookieName && c.Value == "secret-token" {
			found = true
			if !c.HttpOnly {
				t.Error("cookie should be HttpOnly")
			}
		}
	}
	if !found {
		t.Error("auth cookie not set")
	}
}

// TestAuthForwardAuth: when TrustForwardAuth and the header is set, request passes.
func TestAuthForwardAuth(t *testing.T) {
	s := newTestServer(t)
	s.cfg.TrustForwardAuth = true
	s.cfg.ForwardAuthHeader = "X-authentik-username"
	mux := http.NewServeMux()
	s.registerRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	// With header → 200.
	req, _ := http.NewRequest("GET", srv.URL+"/healthz", nil)
	req.Header.Set("X-authentik-username", "alice")
	// healthz is unauthenticated anyway. Hit /routing instead (authed).
	req.URL.Path = "/routing"
	resp, _ := srv.Client().Do(req)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := bytes.NewBuffer(nil), resp.Body
		t.Errorf("expected 200 with forward-auth header, got %d", resp.StatusCode)
		_ = body
	}

	// Without header → 401.
	req2, _ := http.NewRequest("GET", srv.URL+"/routing", nil)
	resp2, _ := srv.Client().Do(req2)
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401 without forward-auth header, got %d", resp2.StatusCode)
	}
}

// TestHealthz: always reachable.
func TestHealthz(t *testing.T) {
	s := newTestServer(t)
	mux := http.NewServeMux()
	s.registerRoutes(mux)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("healthz: %d", resp.StatusCode)
	}
	body := make([]byte, 64)
	n, _ := resp.Body.Read(body)
	if !strings.Contains(string(body[:n]), `"ok":true`) {
		t.Errorf("healthz body: %s", string(body[:n]))
	}
}
