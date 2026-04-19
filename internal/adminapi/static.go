package adminapi

import (
	"embed"
	"io/fs"
	"net/http"
)

// Embedded at build time by Dockerfile (curl → static/). In dev, placeholder
// files live in the repo so `go build` works without Docker.
//
//go:embed static
var staticFS embed.FS

// staticFileServer returns a handler for /static/*.
func staticFileServer() http.Handler {
	sub, err := fs.Sub(staticFS, "static")
	if err != nil {
		panic(err)
	}
	return http.FileServer(http.FS(sub))
}
