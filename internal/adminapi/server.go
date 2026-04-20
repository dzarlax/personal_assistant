// Package adminapi serves the web admin interface — an htmx-driven page for
// browsing OpenRouter models, reassigning them to slots, and editing routing
// roles. Designed to run behind a reverse proxy (Traefik) with optional
// Authentik forward-auth; the package itself only TRUSTS those headers when
// explicitly configured.
package adminapi

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

// Server wraps an http.Server + the upstream dependencies it needs to render
// and mutate routing state.
type Server struct {
	cfg        config.AdminAPIConfig
	router     *llm.Router
	capStore   llm.CapabilityStore
	settings   llm.SettingsStore // for AA cache persistence; may be nil
	usageStore llm.UsageStore    // for Usage/Cost section; may be nil
	cfgRef     *config.Config    // needed for enumerating OpenRouter slots
	logger     *slog.Logger

	httpSrv *http.Server
}

// New constructs the admin API server but does not start it. Call Start to
// bind the listener.
func New(cfg config.AdminAPIConfig, router *llm.Router, capStore llm.CapabilityStore, settings llm.SettingsStore, usageStore llm.UsageStore, cfgRef *config.Config, logger *slog.Logger) *Server {
	if cfg.ForwardAuthHeader == "" {
		cfg.ForwardAuthHeader = "X-authentik-username"
	}
	return &Server{
		cfg:        cfg,
		router:     router,
		capStore:   capStore,
		settings:   settings,
		usageStore: usageStore,
		cfgRef:     cfgRef,
		logger:     logger,
	}
}

// Start binds the listener and serves in a goroutine. Non-blocking.
func (s *Server) Start() error {
	if !s.cfg.Enabled {
		return nil
	}
	if s.cfg.Listen == "" {
		return errors.New("adminapi: listen address is required when enabled")
	}
	if s.cfg.Token == "" && !s.cfg.TrustForwardAuth {
		return errors.New("adminapi: either token or trust_forward_auth must be set")
	}

	mux := http.NewServeMux()
	s.registerRoutes(mux)

	s.httpSrv = &http.Server{
		Addr:              s.cfg.Listen,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		s.logger.Info("admin API listening",
			"addr", s.cfg.Listen,
			"trust_forward_auth", s.cfg.TrustForwardAuth)
		if err := s.httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("admin API server stopped", "err", err)
		}
	}()
	return nil
}

// Shutdown gracefully stops the server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.httpSrv == nil {
		return nil
	}
	return s.httpSrv.Shutdown(ctx)
}

// registerRoutes wires the public surface. Kept separate so tests can assemble
// a mux without starting a real listener.
func (s *Server) registerRoutes(mux *http.ServeMux) {
	// Static assets (design system + htmx). Served unauthenticated so the
	// page can load its own CSS/JS after auth succeeds on /.
	mux.Handle("/static/", http.StripPrefix("/static/", staticFileServer()))

	// Unauthenticated probe.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"ok":true}`)) //nolint:errcheck
	})

	// Authenticated routes.
	authed := s.requireAuth
	mux.Handle("/", authed(http.HandlerFunc(s.handleIndex)))
	mux.Handle("/models", authed(http.HandlerFunc(s.handleModels)))
	mux.Handle("/routing", authed(http.HandlerFunc(s.handleRouting)))
	mux.Handle("/slots/", authed(http.HandlerFunc(s.handleSlotAssign))) // POST /slots/{slot}/assign
	mux.Handle("/routing/", authed(http.HandlerFunc(s.handleRoleSet))) // POST /routing/{role}/set
	mux.Handle("/refresh", authed(http.HandlerFunc(s.handleRefresh)))
	mux.Handle("/usage", authed(http.HandlerFunc(s.handleUsage)))
	mux.Handle("/analytics", authed(http.HandlerFunc(s.handleAnalytics)))
}
