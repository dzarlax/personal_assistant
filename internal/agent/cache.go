package agent

import (
	"sync"
	"time"
)

const (
	cacheHitThreshold   = 0.92
	defaultCacheTTL     = 4 * time.Hour
	defaultCacheMaxSize = 200
)

type cacheEntry struct {
	chatID    int64
	emb       []float32
	response  string
	expiresAt time.Time
}

// ResponseCache is a thread-safe in-memory cache of LLM responses keyed by
// query embedding. Only direct responses (no tool calls) are stored.
// Entries expire after TTL; the oldest entry is evicted when capacity is full.
type ResponseCache struct {
	mu      sync.RWMutex
	entries []cacheEntry
	ttl     time.Duration
	maxSize int
}

func newResponseCache() *ResponseCache {
	return &ResponseCache{ttl: defaultCacheTTL, maxSize: defaultCacheMaxSize}
}

// Get returns a cached response for chatID if a stored embedding has cosine
// similarity ≥ cacheHitThreshold with emb and has not expired.
func (c *ResponseCache) Get(chatID int64, emb []float32) (string, bool) {
	if len(emb) == 0 {
		return "", false
	}
	now := time.Now()
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, e := range c.entries {
		if e.chatID != chatID || now.After(e.expiresAt) {
			continue
		}
		if compactCosine(e.emb, emb) >= cacheHitThreshold {
			return e.response, true
		}
	}
	return "", false
}

// Set stores response for chatID under emb. Evicts expired entries first;
// if still at capacity, removes the oldest entry.
func (c *ResponseCache) Set(chatID int64, emb []float32, response string) {
	if len(emb) == 0 {
		return
	}
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()
	// Evict expired.
	live := c.entries[:0]
	for _, e := range c.entries {
		if now.Before(e.expiresAt) {
			live = append(live, e)
		}
	}
	c.entries = live
	// Evict oldest if at capacity.
	if len(c.entries) >= c.maxSize {
		c.entries = c.entries[1:]
	}
	c.entries = append(c.entries, cacheEntry{
		chatID:    chatID,
		emb:       emb,
		response:  response,
		expiresAt: now.Add(c.ttl),
	})
}
