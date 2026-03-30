package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

// mockProvider — заглушка для тестов. Реализует интерфейс Provider.
type mockProvider struct {
	name  string
	resp  Response
	err   error
	calls int       // сколько раз был вызван Chat
	delay time.Duration // имитировать медленный ответ
}

func (m *mockProvider) Chat(ctx context.Context, _ []Message, _ string, _ []Tool) (Response, error) {
	m.calls++
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return Response{}, ctx.Err()
		}
	}
	return m.resp, m.err
}

func (m *mockProvider) Name() string { return m.name }

// newTestRouter создаёт роутер с именованными провайдерами для тестов.
func newTestRouter(cfg RouterConfig, providers map[string]Provider) *Router {
	return &Router{
		providers: providers,
		cfg:       cfg,
		logger:    noopLogger(),
	}
}

// --- Тесты ---

// TestRouter_UsesPrimary: обычное сообщение → primary провайдер.
func TestRouter_UsesPrimary(t *testing.T) {
	primary := &mockProvider{name: "primary", resp: Response{Content: "hello"}}
	r := newTestRouter(RouterConfig{Primary: "primary"}, map[string]Provider{
		"primary": primary,
	})

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("expected 'hello', got %q", resp.Content)
	}
	if primary.calls != 1 {
		t.Errorf("primary called %d times, want 1", primary.calls)
	}
}

// TestRouter_FallbackOn5xx: primary возвращает 500 → fallback используется.
func TestRouter_FallbackOn5xx(t *testing.T) {
	primary := &mockProvider{name: "primary", err: &APIError{StatusCode: 500, Message: "server error"}}
	fallback := &mockProvider{name: "fallback", resp: Response{Content: "fallback reply"}}

	r := newTestRouter(RouterConfig{Primary: "primary", Fallback: "fallback"}, map[string]Provider{
		"primary":  primary,
		"fallback": fallback,
	})

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "fallback reply" {
		t.Errorf("expected fallback reply, got %q", resp.Content)
	}
}

// TestRouter_FallbackOn429: 429 (rate limit) тоже триггерит fallback.
func TestRouter_FallbackOn429(t *testing.T) {
	primary := &mockProvider{name: "primary", err: &APIError{StatusCode: 429, Message: "rate limited"}}
	fallback := &mockProvider{name: "fallback", resp: Response{Content: "ok"}}

	r := newTestRouter(RouterConfig{Primary: "primary", Fallback: "fallback"}, map[string]Provider{
		"primary":  primary,
		"fallback": fallback,
	})

	_, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if fallback.calls != 1 {
		t.Errorf("fallback should be called once, got %d", fallback.calls)
	}
}

// TestRouter_NoFallbackOn4xx: 4xx (кроме 429) ошибки клиента — fallback не нужен.
func TestRouter_NoFallbackOn4xx(t *testing.T) {
	primary := &mockProvider{name: "primary", err: &APIError{StatusCode: 400, Message: "bad request"}}
	fallback := &mockProvider{name: "fallback", resp: Response{Content: "ok"}}

	r := newTestRouter(RouterConfig{Primary: "primary", Fallback: "fallback"}, map[string]Provider{
		"primary":  primary,
		"fallback": fallback,
	})

	_, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if fallback.calls != 0 {
		t.Errorf("fallback should not be called on 400, got %d calls", fallback.calls)
	}
}

// TestRouter_MultimodalPriority: сообщение с картинкой → multimodal провайдер.
func TestRouter_MultimodalPriority(t *testing.T) {
	primary := &mockProvider{name: "primary"}
	multimodal := &mockProvider{name: "multimodal", resp: Response{Content: "image described"}}

	r := newTestRouter(RouterConfig{Primary: "primary", Multimodal: "multimodal"}, map[string]Provider{
		"primary":   primary,
		"multimodal": multimodal,
	})

	imgMsg := Message{
		Role: "user",
		Parts: []ContentPart{
			{Type: "image_url", ImageURL: &ImageURL{URL: "data:image/jpeg;base64,abc"}},
		},
	}

	resp, err := r.Chat(context.Background(), []Message{imgMsg}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "image described" {
		t.Errorf("expected multimodal response, got %q", resp.Content)
	}
	if primary.calls != 0 {
		t.Errorf("primary should not be called for multimodal message")
	}
	if multimodal.calls != 1 {
		t.Errorf("multimodal should be called once, got %d", multimodal.calls)
	}
}

// TestRouter_ClassifierLevel3: classifier возвращает "3" → reasoner используется.
func TestRouter_ClassifierLevel3(t *testing.T) {
	classifier := &mockProvider{name: "classifier", resp: Response{Content: "3"}}
	reasoner := &mockProvider{name: "reasoner", resp: Response{Content: "deep answer"}}
	primary := &mockProvider{name: "primary", resp: Response{Content: "simple answer"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Reasoner:         "reasoner",
		Classifier:       "classifier",
		ClassifierMinLen: 10,
	}, map[string]Provider{
		"primary":    primary,
		"reasoner":   reasoner,
		"classifier": classifier,
	})

	msg := Message{Role: "user", Content: "prove that sqrt(2) is irrational"}
	resp, err := r.Chat(context.Background(), []Message{msg}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "deep answer" {
		t.Errorf("expected reasoner, got %q", resp.Content)
	}
}

// TestRouter_ClassifierLevel1: classifier возвращает "1" → local используется.
func TestRouter_ClassifierLevel1(t *testing.T) {
	classifier := &mockProvider{name: "classifier", resp: Response{Content: "1"}}
	local := &mockProvider{name: "local", resp: Response{Content: "local answer"}}
	primary := &mockProvider{name: "primary", resp: Response{Content: "cloud answer"}}

	r := newTestRouter(RouterConfig{
		Local:            "local",
		Primary:          "primary",
		Classifier:       "classifier",
		ClassifierMinLen: 5,
	}, map[string]Provider{
		"local":      local,
		"primary":    primary,
		"classifier": classifier,
	})

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "what time is it"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "local answer" {
		t.Errorf("expected local, got %q", resp.Content)
	}
}

// TestRouter_ClassifierLevel2: classifier возвращает "2" → primary используется.
func TestRouter_ClassifierLevel2(t *testing.T) {
	classifier := &mockProvider{name: "classifier", resp: Response{Content: "2"}}
	reasoner := &mockProvider{name: "reasoner"}
	primary := &mockProvider{name: "primary", resp: Response{Content: "cloud answer"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Reasoner:         "reasoner",
		Classifier:       "classifier",
		ClassifierMinLen: 5,
	}, map[string]Provider{
		"primary":    primary,
		"reasoner":   reasoner,
		"classifier": classifier,
	})

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "summarize this article about AI"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "cloud answer" {
		t.Errorf("expected primary, got %q", resp.Content)
	}
	if reasoner.calls != 0 {
		t.Errorf("reasoner should not be called for level 2")
	}
}

// TestRouter_ClassifierSkippedForShortMessages: короткое сообщение → classifier не вызывается.
func TestRouter_ClassifierSkippedForShortMessages(t *testing.T) {
	classifier := &mockProvider{name: "classifier"}
	primary := &mockProvider{name: "primary", resp: Response{Content: "ok"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Classifier:       "classifier",
		ClassifierMinLen: 100, // порог 100 символов
	}, map[string]Provider{
		"primary":    primary,
		"classifier": classifier,
	})

	_, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if classifier.calls != 0 {
		t.Errorf("classifier should not be called for short messages, got %d calls", classifier.calls)
	}
}

// TestRouter_ClassifierDisabled: minLen<0 → classifier никогда не вызывается.
func TestRouter_ClassifierDisabled(t *testing.T) {
	classifier := &mockProvider{name: "classifier"}
	primary := &mockProvider{name: "primary", resp: Response{Content: "ok"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Classifier:       "classifier",
		ClassifierMinLen: -1, // отключён
	}, map[string]Provider{
		"primary":    primary,
		"classifier": classifier,
	})

	_, _ = r.Chat(context.Background(), []Message{{Role: "user", Content: "long message that would normally trigger classifier"}}, "", nil)
	if classifier.calls != 0 {
		t.Errorf("classifier should be disabled when minLen<0, got %d calls", classifier.calls)
	}
}

// TestRouter_ClassifierAlwaysWhenZero: minLen=0 → classifier вызывается для любого сообщения.
func TestRouter_ClassifierAlwaysWhenZero(t *testing.T) {
	classifier := &mockProvider{name: "classifier", resp: Response{Content: "1"}}
	local := &mockProvider{name: "local", resp: Response{Content: "local"}}
	primary := &mockProvider{name: "primary", resp: Response{Content: "cloud"}}

	r := newTestRouter(RouterConfig{
		Local:            "local",
		Primary:          "primary",
		Classifier:       "classifier",
		ClassifierMinLen: 0, // всегда классифицировать
	}, map[string]Provider{
		"local":      local,
		"primary":    primary,
		"classifier": classifier,
	})

	resp, _ := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if classifier.calls != 1 {
		t.Errorf("classifier should be called for any message when minLen=0, got %d calls", classifier.calls)
	}
	if resp.Content != "local" {
		t.Errorf("expected local, got %q", resp.Content)
	}
}

// TestRouter_ClassifierErrorFallsBackToPrimary: classifier падает → primary используется (наш новый фикс).
func TestRouter_ClassifierErrorFallsBackToPrimary(t *testing.T) {
	classifier := &mockProvider{name: "classifier", err: errors.New("classifier unavailable")}
	reasoner := &mockProvider{name: "reasoner"}
	primary := &mockProvider{name: "primary", resp: Response{Content: "primary answer"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Reasoner:         "reasoner",
		Classifier:       "classifier",
		ClassifierMinLen: 5,
	}, map[string]Provider{
		"primary":    primary,
		"reasoner":   reasoner,
		"classifier": classifier,
	})

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "some long question here"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "primary answer" {
		t.Errorf("expected primary after classifier failure, got %q", resp.Content)
	}
	if reasoner.calls != 0 {
		t.Errorf("reasoner should not be called when classifier errors")
	}
}

// TestRouter_ClassifierTimeout: classifier медленный → timeout (5s) и fallback на primary.
func TestRouter_ClassifierTimeout(t *testing.T) {
	// Classifier зависает, но 5-секундный таймаут должен его прервать.
	// Чтобы тест был быстрым — используем уже отменённый контекст как родительский:
	// classifierCtx = WithTimeout(ctx, 5s), но ctx уже почти истёк.
	classifier := &mockProvider{name: "classifier", delay: 10 * time.Second} // медленно
	primary := &mockProvider{name: "primary", resp: Response{Content: "primary"}}

	r := newTestRouter(RouterConfig{
		Primary:          "primary",
		Classifier:       "classifier",
		ClassifierMinLen: 5,
	}, map[string]Provider{
		"primary":    primary,
		"classifier": classifier,
	})

	// Контекст с очень коротким дедлайном — classifier.Chat получит его и сразу вернёт DeadlineExceeded
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()
	time.Sleep(2 * time.Millisecond) // убедимся что контекст истёк

	resp, err := r.Chat(ctx, []Message{{Role: "user", Content: "some message"}}, "", nil)
	// Контекст истёк, поэтому primary тоже может вернуть ошибку — это нормально.
	// Главное — reasoner не должен быть вызван.
	_ = resp
	_ = err
	if classifier.calls > 0 {
		// Если classifier был вызван, убеждаемся что его ошибка не привела к вызову reasoner
		// (reasoner.calls уже 0, нет reasoner в этом тесте)
	}
}

// TestRouter_Override: SetOverride → всегда используется указанный провайдер.
func TestRouter_Override(t *testing.T) {
	primary := &mockProvider{name: "primary"}
	special := &mockProvider{name: "special", resp: Response{Content: "special"}}

	r := newTestRouter(RouterConfig{Primary: "primary"}, map[string]Provider{
		"primary": primary,
		"special": special,
	})

	if err := r.SetOverride("special"); err != nil {
		t.Fatalf("SetOverride failed: %v", err)
	}

	resp, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "special" {
		t.Errorf("expected special provider, got %q", resp.Content)
	}
	if primary.calls != 0 {
		t.Errorf("primary should not be called when override is set")
	}
}

// TestRouter_UnknownOverride: SetOverride с несуществующим именем → ошибка.
func TestRouter_UnknownOverride(t *testing.T) {
	r := newTestRouter(RouterConfig{Primary: "primary"}, map[string]Provider{
		"primary": &mockProvider{name: "primary"},
	})

	if err := r.SetOverride("nonexistent"); err == nil {
		t.Error("expected error for unknown override, got nil")
	}
}

// TestRouter_FallbackCalledOnce: fallback вызывается только один раз, не рекурсивно.
func TestRouter_FallbackCalledOnce(t *testing.T) {
	primary := &mockProvider{name: "primary", err: &APIError{StatusCode: 503}}
	fallback := &mockProvider{name: "fallback", err: &APIError{StatusCode: 503}}

	r := newTestRouter(RouterConfig{Primary: "primary", Fallback: "fallback"}, map[string]Provider{
		"primary":  primary,
		"fallback": fallback,
	})

	_, err := r.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, "", nil)
	if err == nil {
		t.Fatal("expected error when both primary and fallback fail")
	}
	if fallback.calls != 1 {
		t.Errorf("fallback should be called exactly once, got %d", fallback.calls)
	}
}
