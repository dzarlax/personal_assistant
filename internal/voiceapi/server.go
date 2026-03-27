package voiceapi

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/hajimehoshi/go-mp3"

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

	// Step 3: Synthesize speech — strip markdown for cleaner voice output.
	spokenText := stripMarkdown(response)
	audio, err := s.agent.SynthesizeSpeech(ctx, spokenText)
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

	// Check if client wants WAV (ESP32 speakers need raw PCM).
	accept := r.Header.Get("Accept")
	if strings.Contains(accept, "audio/wav") {
		wavData, convErr := mp3ToWAV(audio)
		if convErr != nil {
			s.logger.Error("MP3→WAV conversion failed", "err", convErr)
			// Fall back to MP3.
		} else {
			audio = wavData
			s.logger.Info("converted to WAV", "wav_bytes", len(wavData))
		}
	}

	respContentType := "audio/mpeg"
	if strings.Contains(accept, "audio/wav") {
		respContentType = "audio/wav"
	}

	w.Header().Set("Content-Type", respContentType)
	w.Header().Set("X-Transcription", truncate(text, 200))
	w.Header().Set("X-Response", truncate(response, 500))
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(audio)))
	w.Write(audio)

	s.logger.Info("voice reply sent", "audio_bytes", len(audio), "format", respContentType)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// mp3ToWAV decodes MP3 to 16kHz 16-bit mono PCM WAV.
// go-mp3 outputs stereo 24kHz 16-bit PCM; we downsample and convert to mono.
func mp3ToWAV(mp3Data []byte) ([]byte, error) {
	decoder, err := mp3.NewDecoder(bytes.NewReader(mp3Data))
	if err != nil {
		return nil, fmt.Errorf("mp3 decode: %w", err)
	}

	// Read all decoded PCM (stereo, 16-bit LE, source sample rate).
	pcm, err := io.ReadAll(decoder)
	if err != nil {
		return nil, fmt.Errorf("mp3 read: %w", err)
	}

	srcRate := decoder.SampleRate()

	// Convert stereo to mono: take left channel only (every other int16).
	monoSamples := len(pcm) / 4 // 4 bytes per stereo sample (2 channels × 2 bytes)
	mono := make([]int16, monoSamples)
	for i := 0; i < monoSamples; i++ {
		mono[i] = int16(binary.LittleEndian.Uint16(pcm[i*4 : i*4+2]))
	}

	// Resample to 16kHz using linear interpolation.
	const dstRate = 16000
	ratio := float64(srcRate) / float64(dstRate)
	dstLen := int(float64(len(mono)) / ratio)
	resampled := make([]int16, dstLen)
	for i := 0; i < dstLen; i++ {
		srcIdx := float64(i) * ratio
		idx := int(srcIdx)
		frac := srcIdx - float64(idx)
		if idx+1 < len(mono) {
			resampled[i] = int16(float64(mono[idx])*(1-frac) + float64(mono[idx+1])*frac)
		} else if idx < len(mono) {
			resampled[i] = mono[idx]
		}
	}

	// Build WAV file.
	dataSize := len(resampled) * 2
	buf := make([]byte, 44+dataSize)

	// WAV header.
	copy(buf[0:], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:], uint32(36+dataSize))
	copy(buf[8:], "WAVE")
	copy(buf[12:], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:], 16) // chunk size
	binary.LittleEndian.PutUint16(buf[20:], 1)  // PCM
	binary.LittleEndian.PutUint16(buf[22:], 1)  // mono
	binary.LittleEndian.PutUint32(buf[24:], dstRate)
	binary.LittleEndian.PutUint32(buf[28:], dstRate*2) // byte rate
	binary.LittleEndian.PutUint16(buf[32:], 2)         // block align
	binary.LittleEndian.PutUint16(buf[34:], 16)        // bits per sample
	copy(buf[36:], "data")
	binary.LittleEndian.PutUint32(buf[40:], uint32(dataSize))

	// Write PCM samples.
	for i, s := range resampled {
		binary.LittleEndian.PutUint16(buf[44+i*2:], uint16(s))
	}

	return buf, nil
}

// stripMarkdown removes markdown formatting that sounds bad when spoken.
func stripMarkdown(s string) string {
	// Remove code blocks.
	for {
		start := strings.Index(s, "```")
		if start == -1 {
			break
		}
		end := strings.Index(s[start+3:], "```")
		if end == -1 {
			s = s[:start]
			break
		}
		s = s[:start] + s[start+3+end+3:]
	}
	// Remove markdown markers.
	r := strings.NewReplacer(
		"**", "",
		"__", "",
		"~~", "",
		"`", "",
		"* ", "- ",  // bullet points stay readable
		"# ", "",
		"## ", "",
		"### ", "",
		"#### ", "",
	)
	s = r.Replace(s)
	return strings.TrimSpace(s)
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
