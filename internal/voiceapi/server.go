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

	"github.com/gorilla/websocket"
	"github.com/hajimehoshi/go-mp3"

	"telegram-agent/internal/agent"
	"telegram-agent/internal/config"
	"telegram-agent/internal/llm"
)

var wsUpgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

const (
	maxBodySize    = 640 * 1024 // 640 KB — ~10 sec at 16kHz 16bit mono + margin
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
	mux.HandleFunc("/voice/ws", s.authWS(s.handleWebSocket))
	mux.HandleFunc("/voice/health", s.handleHealth)

	listen := cfg.Listen
	if listen == "" {
		listen = ":8086"
	}

	s.srv = &http.Server{
		Addr:         listen,
		Handler:      mux,
		ReadTimeout:  0, // no read timeout — clients stream audio over seconds
		WriteTimeout: requestTimeout + 30*time.Second,
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

// authWS checks token from query parameter (WebSocket can't send custom headers in ESP-IDF).
func (s *Server) authWS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Token == "" {
			next(w, r)
			return
		}
		token := r.URL.Query().Get("token")
		if token != s.cfg.Token {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
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

	// Read audio body — supports both chunked and regular transfer.
	s.logger.Info("reading body", "content_length", r.ContentLength, "transfer_encoding", r.TransferEncoding)
	var body []byte
	buf := make([]byte, 4096)
	for {
		n, readErr := r.Body.Read(buf)
		if n > 0 {
			body = append(body, buf[:n]...)
			if len(body) > maxBodySize {
				writeError(w, http.StatusRequestEntityTooLarge, "audio too large")
				return
			}
		}
		if readErr != nil {
			break // EOF or error — either way we have what we got
		}
	}
	if len(body) == 0 {
		writeError(w, http.StatusBadRequest, "empty body")
		return
	}
	s.logger.Info("body read complete", "size", len(body))

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

// handleWebSocket handles streaming voice interaction over WebSocket.
// Protocol:
//   Client → Server: binary frames with raw PCM audio (16kHz 16bit mono, no WAV header)
//   Client → Server: text frame {"action":"stop"} signals end of recording
//   Server → Client: text frame {"status":"processing","transcription":"..."} after STT
//   Server → Client: binary frames with WAV audio response (streamed)
//   Server → Client: text frame {"status":"done","response":"..."} when complete
func (s *Server) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := wsUpgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Error("websocket upgrade failed", "err", err)
		return
	}
	defer conn.Close()

	s.logger.Info("websocket connected", "remote", r.RemoteAddr)

	// Phase 1: Receive audio frames until "stop" signal.
	var audioData []byte
	conn.SetReadDeadline(time.Now().Add(30 * time.Second)) // max recording time

	for {
		msgType, data, readErr := conn.ReadMessage()
		if readErr != nil {
			s.logger.Warn("websocket read error", "err", readErr)
			break
		}

		if msgType == websocket.TextMessage {
			var msg struct {
				Action string `json:"action"`
			}
			if json.Unmarshal(data, &msg) == nil && msg.Action == "stop" {
				s.logger.Info("recording stop received", "audio_bytes", len(audioData))
				break
			}
		} else if msgType == websocket.BinaryMessage {
			audioData = append(audioData, data...)
			if len(audioData) > maxBodySize {
				s.logger.Warn("audio too large, truncating")
				break
			}
		}
	}

	if len(audioData) == 0 {
		conn.WriteJSON(map[string]string{"status": "error", "error": "no audio received"})
		return
	}

	// Build WAV from raw PCM for Gemini STT.
	wavData := buildWAVFromPCM(audioData, 16000)
	s.logger.Info("audio received", "pcm_bytes", len(audioData), "duration_ms", len(audioData)*1000/(16000*2))

	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	// Phase 2: Transcribe.
	text, err := s.agent.TranscribeAudio(ctx, wavData, "audio/wav")
	if err != nil {
		s.logger.Error("STT failed", "err", err)
		conn.WriteJSON(map[string]string{"status": "error", "error": "transcription failed"})
		return
	}
	if text == "" {
		conn.WriteJSON(map[string]string{"status": "error", "error": "no speech detected"})
		return
	}
	s.logger.Info("transcribed", "text", truncate(text, 100))
	conn.WriteJSON(map[string]string{"status": "processing", "transcription": text})

	// Phase 3: Process through agent.
	chatID := s.cfg.ChatID
	if chatID == 0 {
		chatID = 9999
	}
	response, err := s.agent.Process(ctx, chatID, llm.Message{Role: "user", Content: text}, nil)
	if err != nil {
		s.logger.Error("agent failed", "err", err)
		conn.WriteJSON(map[string]string{"status": "error", "error": "agent error"})
		return
	}
	s.logger.Info("agent response", "len", len(response))

	// Phase 4: Synthesize and send audio.
	spokenText := stripMarkdown(response)
	mp3Audio, err := s.agent.SynthesizeSpeech(ctx, spokenText)
	if err != nil {
		s.logger.Error("TTS failed", "err", err)
		conn.WriteJSON(map[string]string{"status": "done", "response": response, "error": "TTS failed"})
		return
	}

	// Convert to WAV for ESP32.
	wavResponse, err := mp3ToWAV(mp3Audio)
	if err != nil {
		s.logger.Error("MP3→WAV failed", "err", err)
		conn.WriteJSON(map[string]string{"status": "done", "response": response, "error": "audio conversion failed"})
		return
	}

	// Send WAV in chunks over websocket.
	const chunkSize = 4096
	for offset := 0; offset < len(wavResponse); offset += chunkSize {
		end := offset + chunkSize
		if end > len(wavResponse) {
			end = len(wavResponse)
		}
		if writeErr := conn.WriteMessage(websocket.BinaryMessage, wavResponse[offset:end]); writeErr != nil {
			s.logger.Warn("websocket write error", "err", writeErr)
			return
		}
	}

	conn.WriteJSON(map[string]string{"status": "done", "response": truncate(response, 500)})
	s.logger.Info("websocket response sent", "audio_bytes", len(wavResponse))
}

// buildWAVFromPCM wraps raw PCM data with a WAV header.
func buildWAVFromPCM(pcm []byte, sampleRate uint32) []byte {
	dataSize := uint32(len(pcm))
	buf := make([]byte, 44+len(pcm))

	copy(buf[0:], "RIFF")
	binary.LittleEndian.PutUint32(buf[4:], 36+dataSize)
	copy(buf[8:], "WAVE")
	copy(buf[12:], "fmt ")
	binary.LittleEndian.PutUint32(buf[16:], 16)
	binary.LittleEndian.PutUint16(buf[20:], 1) // PCM
	binary.LittleEndian.PutUint16(buf[22:], 1) // mono
	binary.LittleEndian.PutUint32(buf[24:], sampleRate)
	binary.LittleEndian.PutUint32(buf[28:], sampleRate*2) // byte rate
	binary.LittleEndian.PutUint16(buf[32:], 2)            // block align
	binary.LittleEndian.PutUint16(buf[34:], 16)           // bits per sample
	copy(buf[36:], "data")
	binary.LittleEndian.PutUint32(buf[40:], dataSize)
	copy(buf[44:], pcm)

	return buf
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
