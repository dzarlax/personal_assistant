FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS builder

ARG TARGETARCH

# Admin UI static assets. Re-fetched on every build when ASSETS_CACHEBUST
# differs (CI passes a timestamp / commit SHA). For local iteration, the
# default value keeps the cached layer.
#
# Override for a one-off local refresh:
#   docker build --build-arg ASSETS_CACHEBUST=$(date +%s) .
ARG DS_VERSION=main
ARG HTMX_VERSION=2.0.3
ARG ASSETS_CACHEBUST=pinned

RUN apk add --no-cache curl

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Overwrite the committed placeholders with real assets. The ASSETS_CACHEBUST
# ARG is referenced so that changing it invalidates this layer.
RUN echo "assets cachebust: ${ASSETS_CACHEBUST}" && \
    curl -fsSL "https://cdn.jsdelivr.net/gh/dzarlax/design-system@${DS_VERSION}/dist/dzarlax.css" \
         -o internal/adminapi/static/dzarlax.css && \
    curl -fsSL "https://cdn.jsdelivr.net/gh/dzarlax/design-system@${DS_VERSION}/dist/dzarlax.js" \
         -o internal/adminapi/static/dzarlax.js && \
    curl -fsSL "https://unpkg.com/htmx.org@${HTMX_VERSION}/dist/htmx.min.js" \
         -o internal/adminapi/static/htmx.min.js

RUN CGO_ENABLED=0 GOOS=linux GOARCH=$TARGETARCH go build -ldflags="-w -s" -o bin/agent ./cmd/agent

FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

COPY --from=builder /app/bin/agent .

ENTRYPOINT ["./agent"]
