# Admin UI static assets

These files are placeholders so `go build` works without Docker. The **real**
assets are downloaded from upstream at Docker build time — see the `curl`
step in the repo-root `Dockerfile` (`ARG DS_VERSION`, `ARG HTMX_VERSION`,
`ARG ASSETS_CACHEBUST`).

To force a refresh of the design system on your next build, pass a new
cache-bust value:

```bash
docker build --build-arg ASSETS_CACHEBUST=$(date +%s) .
```

CI passes this automatically so every built image ships with the current
design system `@main`.

Do **not** commit real builds of `dzarlax.css`, `dzarlax.js`, or
`htmx.min.js` — they bloat diffs and go stale.
