package voiceapi

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

const (
	maxBodySize    = 256 * 1024 // 256 KB — ~4 sec at 16kHz 16bit mono
	requestTimeout = 60 * time.Second
)

// Server serves the HTTP voice API.
type Server struct {
	agent  *agent.Agent
	cfg    config.VoiceAPIConfig
	logger *slog.Logger
	srv    *http.Server
}

// New creates a voice API server.
func New(ag *agent.Agent, cfg config.VoiceAPIConfig, logger *slog.Logger) *Server {
	s := &Server{
		agent:  ag,
		cfg:    cfg,
		logger: logger,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/voice", s.auth(s.handleVoice))
	mux.HandleFunc("/voice/health", s.handleHealth)

	listen := cfg.Listen
	if listen == "" {
		listen = ":8086"
	}

	s.srv = &http.Server{
		Addr:         listen,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: requestTimeout + 10*time.Second,
	}
	return s
}

// ListenAndServe starts the HTTP server. Blocks until the server stops.
func (s *Server) ListenAndServe() error {
	s.logger.Info("voice API listening", "addr", s.srv.Addr)
	if err := s.srv.ListenAndServe(); err != http.ErrServerClosed {
		return fmt.Errorf("voice API: %w", err)
	}
	return nil
}

// Shutdown gracefully stops the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}

// auth is a middleware that checks the Bearer token.
func (s *Server) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Token == "" {
			next(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") || auth[7:] != s.cfg.Token {
			writeError(w, http.StatusUnauthorized, "invalid token")
			return
		}
		next(w, r)
	}
}

// handleVoice processes a voice request: STT → LLM → TTS.
func (s *Server) handleVoice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "POST required")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()

	// Read audio body.
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBodySize+1))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	if len(body) > maxBodySize {
		writeError(w, http.StatusRequestEntityTooLarge, "audio too large (max 256 KB)")
		return
	}
	if len(body) == 0 {
		writeError(w, http.StatusBadRequest, "empty body")
		return
	}

	contentType := r.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "audio/wav"
	}

	s.logger.Info("voice request", "content_type", contentType, "body_size", len(body))

	// Step 1: Transcribe audio to text.
	text, err := s.agent.TranscribeAudio(ctx, body, contentType)
	if err != nil {
		s.logger.Error("STT failed", "err", err)
		writeError(w, http.StatusInternalServerError, "transcription failed: "+err.Error())
		return
	}
	if text == "" {
		writeError(w, http.StatusBadRequest, "no speech detected")
		return
	}
	s.logger.Info("voice transcribed", "text_len", len(text), "text", truncate(text, 100))

	// Step 2: Process through agent.
	chatID := s.cfg.ChatID
	if chatID == 0 {
		chatID = 9999
	}
	userMsg := llm.Message{Role: "user", Content: text}
	response, err := s.agent.Process(ctx, chatID, userMsg, nil)
	if err != nil {
		s.logger.Error("agent failed", "err", err)
		writeError(w, http.StatusInternalServerError, "agent error: "+err.Error())
		return
	}
	s.logger.Info("agent response", "response_len", len(response))

	// Step 3: Synthesize speech.
	audio, err := s.agent.SynthesizeSpeech(ctx, response)
	if err != nil {
		s.logger.Error("TTS failed", "err", err)
		// Return text response even if TTS fails.
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Transcription", truncate(text, 200))
		json.NewEncoder(w).Encode(map[string]string{
			"transcription": text,
			"response":      response,
			"error":         "TTS failed: " + err.Error(),
		})
		return
	}

	// Return MP3 audio.
	w.Header().Set("Content-Type", "audio/mpeg")
	w.Header().Set("X-Transcription", truncate(text, 200))
	w.Header().Set("X-Response", truncate(response, 500))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(audio)))
	w.Write(audio)

	s.logger.Info("voice reply sent", "audio_bytes", len(audio))
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
