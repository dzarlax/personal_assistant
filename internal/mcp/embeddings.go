package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"telegram-agent/internal/config"
)

const geminiEmbeddingBaseURL = "https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent"

// embed dispatches to the correct embedding provider based on cfg.Provider.
// Supported: "gemini" (default), "hf-tei", "openai".
func embed(ctx context.Context, cfg config.ModelConfig, text string) ([]float32, error) {
	switch cfg.Provider {
	case "hf-tei":
		return embedHFTEI(ctx, cfg, text)
	case "openai":
		return embedOpenAI(ctx, cfg, text)
	default: // "gemini" or empty
		return embedGemini(ctx, cfg, text)
	}
}

// embedGemini calls the Gemini embedContent API.
func embedGemini(ctx context.Context, cfg config.ModelConfig, text string) ([]float32, error) {
	reqBody := map[string]any{
		"model": "models/" + cfg.Model,
		"content": map[string]any{
			"parts": []map[string]any{{"text": text}},
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf(geminiEmbeddingBaseURL, cfg.Model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", cfg.APIKey)

	return doEmbedRequest(req, func(body []byte) ([]float32, error) {
		var result struct {
			Embedding struct {
				Values []float32 `json:"values"`
			} `json:"embedding"`
		}
		if err := json.Unmarshal(body, &result); err != nil {
			return nil, fmt.Errorf("decode gemini response: %w", err)
		}
		return result.Embedding.Values, nil
	})
}

// embedHFTEI calls a HuggingFace Text Embeddings Inference server.
// Format: POST {base_url}/embed, {"inputs": text} → [[float, ...]]
// Auth: Basic Auth — api_key as "user:password" or just password (empty user).
func embedHFTEI(ctx context.Context, cfg config.ModelConfig, text string) ([]float32, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("hf-tei embedding requires base_url")
	}
	reqBody := map[string]any{"inputs": text}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(cfg.BaseURL, "/") + "/embed"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		user, pass, _ := strings.Cut(cfg.APIKey, ":")
		req.SetBasicAuth(user, pass)
	}

	return doEmbedRequest(req, func(body []byte) ([]float32, error) {
		// Response: [[float, ...]] — outer array is batch, take first element
		var result [][]float32
		if err := json.Unmarshal(body, &result); err != nil {
			return nil, fmt.Errorf("decode hf-tei response: %w", err)
		}
		if len(result) == 0 {
			return nil, fmt.Errorf("hf-tei returned empty embedding")
		}
		return result[0], nil
	})
}

// embedOpenAI calls an OpenAI-compatible embeddings endpoint.
// Format: POST {base_url}/v1/embeddings, Bearer auth.
func embedOpenAI(ctx context.Context, cfg config.ModelConfig, text string) ([]float32, error) {
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("openai embedding requires base_url")
	}
	reqBody := map[string]any{
		"input": text,
		"model": cfg.Model,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(cfg.BaseURL, "/") + "/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	return doEmbedRequest(req, func(body []byte) ([]float32, error) {
		var result struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(body, &result); err != nil {
			return nil, fmt.Errorf("decode openai response: %w", err)
		}
		if len(result.Data) == 0 {
			return nil, fmt.Errorf("openai returned empty embedding")
		}
		return result.Data[0].Embedding, nil
	})
}

var embedHTTPClient = &http.Client{Timeout: 15 * time.Second}

// doEmbedRequest executes an HTTP request and parses the response with the given decoder.
func doEmbedRequest(req *http.Request, decode func([]byte) ([]float32, error)) ([]float32, error) {
	resp, err := embedHTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("embedding API returned %d: %s", resp.StatusCode, string(body))
	}
	return decode(body)
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
