package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestClassifyErrorClass(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"nil", nil, ""},
		{"rate_limit", &APIError{StatusCode: 429, Message: "limit"}, "rate_limit"},
		{"5xx", &APIError{StatusCode: 502, Message: "bad"}, "5xx"},
		{"4xx", &APIError{StatusCode: 400, Message: "bad"}, "4xx"},
		{"network-conn-refused", errors.New("dial tcp: connection refused"), "network"},
		{"network-dns", errors.New("lookup bogus: no such host"), "network"},
		{"timeout", errors.New("context deadline exceeded"), "timeout"},
		{"other", errors.New("kaboom"), "other"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyErrorClass(tc.err)
			if got != tc.want {
				t.Fatalf("ClassifyErrorClass(%v) = %q, want %q", tc.err, got, tc.want)
			}
		})
	}
}

func TestTurnMetaRoundTrip(t *testing.T) {
	ctx := context.Background()
	meta := TurnMeta{ChatID: 42, UserMessageID: 777, RoleHint: "classifier"}
	ctx = WithTurnMeta(ctx, meta)

	got, ok := TurnMetaFrom(ctx)
	if !ok {
		t.Fatal("TurnMetaFrom should find the value")
	}
	if got != meta {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", got, meta)
	}

	if _, ok := TurnMetaFrom(context.Background()); ok {
		t.Fatal("empty ctx must not yield meta")
	}
}

func TestParseUsageFromOpenAICompat(t *testing.T) {
	// A trimmed OpenRouter-style response with usage block + cached + reasoning.
	body := []byte(`{
	  "id": "gen-123abc",
	  "choices": [{"message": {"content": "hi", "tool_calls": null}}],
	  "usage": {
	    "prompt_tokens": 100,
	    "completion_tokens": 50,
	    "total_tokens": 150,
	    "prompt_tokens_details": {"cached_tokens": 60},
	    "completion_tokens_details": {"reasoning_tokens": 20}
	  }
	}`)
	var cr rawChatResponse
	if err := json.Unmarshal(body, &cr); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if cr.Usage == nil {
		t.Fatal("usage should be present")
	}
	if cr.Usage.PromptTokens != 100 || cr.Usage.CompletionTokens != 50 {
		t.Fatalf("wrong counts: %+v", cr.Usage)
	}
	if cr.Usage.PromptTokensDetails == nil || cr.Usage.PromptTokensDetails.CachedTokens != 60 {
		t.Fatalf("cached tokens not parsed: %+v", cr.Usage)
	}
	if cr.Usage.CompletionTokensDetails == nil || cr.Usage.CompletionTokensDetails.ReasoningTokens != 20 {
		t.Fatalf("reasoning tokens not parsed: %+v", cr.Usage)
	}
	if cr.ID != "gen-123abc" {
		t.Fatalf("request id not parsed: %q", cr.ID)
	}
}
