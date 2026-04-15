package agent

import (
	"fmt"
	"log/slog"
	"time"
)

// requestTrace records durations for named phases of a single agent request.
// Usage:
//
//	tr := newRequestTrace()
//	done := tr.begin("embed")
//	... work ...
//	done()
//	tr.log(logger, chatID, "ok")
type requestTrace struct {
	spans []traceSpan
	start time.Time
}

type traceSpan struct {
	name string
	dur  time.Duration
}

func newRequestTrace() *requestTrace {
	return &requestTrace{start: time.Now()}
}

// begin starts a named span and returns a function that records its duration.
func (t *requestTrace) begin(name string) func() {
	s := time.Now()
	return func() {
		t.spans = append(t.spans, traceSpan{name: name, dur: time.Since(s)})
	}
}

// addSpan records a pre-measured span (e.g. extracted from context timings).
func (t *requestTrace) addSpan(name string, dur time.Duration) {
	t.spans = append(t.spans, traceSpan{name: name, dur: dur})
}

// log emits a single structured log line with all span durations in milliseconds.
func (t *requestTrace) log(logger *slog.Logger, chatID int64, status string) {
	args := make([]any, 0, 3+len(t.spans)*2)
	args = append(args, "chat_id", chatID, "status", status, "total_ms", time.Since(t.start).Milliseconds())
	for _, s := range t.spans {
		args = append(args, fmt.Sprintf("%s_ms", s.name), s.dur.Milliseconds())
	}
	logger.Info("request_trace", args...)
}
