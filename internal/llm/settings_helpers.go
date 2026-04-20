package llm

import (
	"context"
	"strconv"
	"strings"
)

// Typed helpers over the generic SettingsStore. Values are stored as strings
// (see sqlite/postgres schema) so every getter converts on read with a safe
// fallback. Nil stores are tolerated — callers can pass settings directly
// without guarding against a missing DB.
//
// All helpers return the default on: nil store, missing key, empty string,
// or parse error. They never log or panic; treat storage as best-effort.

// GetIntSetting reads an integer setting, returning def when the value is
// absent or unparseable.
func GetIntSetting(ctx context.Context, s SettingsStore, key string, def int) int {
	if s == nil {
		return def
	}
	v, ok, err := s.GetSetting(ctx, key)
	if err != nil || !ok {
		return def
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return def
	}
	return n
}

// GetBoolSetting reads a boolean setting. Accepts "1"/"0", "true"/"false",
// "yes"/"no", and "on"/"off" (case-insensitive). Returns def on miss.
func GetBoolSetting(ctx context.Context, s SettingsStore, key string, def bool) bool {
	if s == nil {
		return def
	}
	v, ok, err := s.GetSetting(ctx, key)
	if err != nil || !ok {
		return def
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	}
	return def
}

// GetStringSetting reads a string setting, returning def when absent.
// Whitespace is preserved; trim at the call site if needed.
func GetStringSetting(ctx context.Context, s SettingsStore, key string, def string) string {
	if s == nil {
		return def
	}
	v, ok, err := s.GetSetting(ctx, key)
	if err != nil || !ok {
		return def
	}
	return v
}

// PutIntSetting stores an integer — thin wrapper for symmetry with the
// getters.
func PutIntSetting(ctx context.Context, s SettingsStore, key string, n int) error {
	if s == nil {
		return nil
	}
	return s.PutSetting(ctx, key, strconv.Itoa(n))
}

// PutBoolSetting stores a boolean as "true"/"false".
func PutBoolSetting(ctx context.Context, s SettingsStore, key string, v bool) error {
	if s == nil {
		return nil
	}
	str := "false"
	if v {
		str = "true"
	}
	return s.PutSetting(ctx, key, str)
}
