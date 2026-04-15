package llm

import (
	"context"
	"time"
)

// RequestTimings collects per-phase durations for a single router.Chat call.
// Attach to a context via WithTimings; the router populates it during the call.
type RequestTimings struct {
	ClassifyDur time.Duration
	ClassifyRan bool
}

type timingCtxKey struct{}

// WithTimings returns a child context carrying a *RequestTimings that the router
// will populate. The caller reads timings after Chat/ChatStream returns.
func WithTimings(ctx context.Context) (context.Context, *RequestTimings) {
	t := &RequestTimings{}
	return context.WithValue(ctx, timingCtxKey{}, t), t
}

func timingsFrom(ctx context.Context) *RequestTimings {
	t, _ := ctx.Value(timingCtxKey{}).(*RequestTimings)
	return t
}
