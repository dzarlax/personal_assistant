package adminapi

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

const authCookieName = "admin_auth"

// requireAuth wraps a handler, allowing access when:
//  1. TrustForwardAuth=true AND the configured header (default
//     X-authentik-username) is non-empty — i.e. Traefik/authentik has
//     authenticated the caller upstream.
//  2. A valid `?token=<ADMIN_API_TOKEN>` query is present — bootstrap flow;
//     the server sets an auth cookie and redirects to the same URL without
//     the token.
//  3. Cookie `admin_auth` matches the configured token.
//  4. `Authorization: Bearer <token>` matches the configured token (for
//     curl / monitoring).
//
// Otherwise responds 401. Constant-time token compare prevents timing attacks.
func (s *Server) requireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 1. Forward-auth (Authentik via Traefik).
		if s.cfg.TrustForwardAuth && r.Header.Get(s.cfg.ForwardAuthHeader) != "" {
			next.ServeHTTP(w, r)
			return
		}

		// 2. Bootstrap via ?token=...
		if q := r.URL.Query().Get("token"); q != "" && s.tokenMatches(q) {
			http.SetCookie(w, &http.Cookie{
				Name:     authCookieName,
				Value:    s.cfg.Token,
				Path:     "/",
				HttpOnly: true,
				Secure:   r.TLS != nil || strings.EqualFold(r.Header.Get("X-Forwarded-Proto"), "https"),
				SameSite: http.SameSiteStrictMode,
				MaxAge:   60 * 60 * 24 * 30, // 30 days
			})
			// Strip the token from the URL and redirect.
			q := r.URL.Query()
			q.Del("token")
			r.URL.RawQuery = q.Encode()
			http.Redirect(w, r, r.URL.String(), http.StatusSeeOther)
			return
		}

		// 3. Cookie.
		if c, err := r.Cookie(authCookieName); err == nil && s.tokenMatches(c.Value) {
			next.ServeHTTP(w, r)
			return
		}

		// 4. Bearer.
		if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
			if s.tokenMatches(strings.TrimPrefix(h, "Bearer ")) {
				next.ServeHTTP(w, r)
				return
			}
		}

		w.Header().Set("WWW-Authenticate", `Bearer realm="admin"`)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	})
}

func (s *Server) tokenMatches(got string) bool {
	if s.cfg.Token == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(s.cfg.Token)) == 1
}
