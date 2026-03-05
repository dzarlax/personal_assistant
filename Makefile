.PHONY: setup build run docker-build docker-up docker-down logs

setup:
	go mod tidy

build: setup
	go build -o bin/agent ./cmd/agent

run: setup
	set -a && . ./.env && set +a && go run ./cmd/agent

# Copy example files that don't exist yet
init-config:
	@[ -f .env ] || (cp .env.example .env && echo "Created .env — fill in your secrets")
	@[ -f config/mcp.json ] || (cp config/mcp.json.example config/mcp.json && echo "Created config/mcp.json from example")
	@mkdir -p data

docker-build: setup
	docker compose build

docker-up: init-config
	docker compose up -d

docker-down:
	docker compose down

logs:
	docker compose logs -f agent
