.PHONY: up down logs restart test ingest clean build status sync

# ---------------------------------------------------------------------------
# Dependency Management (uv)
# ---------------------------------------------------------------------------
sync:
	uv sync

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
up:
	docker compose up -d

build:
	docker compose build --no-cache

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

status:
	@echo "── Containers ──"
	@docker compose ps
	@echo ""
	@echo "── Qdrant health ──"
	@curl -sf http://localhost:6333/healthz && echo " OK" || echo " UNREACHABLE"
	@echo ""
	@echo "── MCP server health ──"
	@curl -sf http://localhost:8000/healthz && echo " OK" || echo " UNREACHABLE"
	@echo ""
	@echo "── MCP Prometheus metrics endpoint ──"
	@curl -sf http://localhost:8000/metrics | head -n 5 || echo " UNREACHABLE"

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
ingest:
	@test -n "$(REPO_PATH)" || (echo "Usage: make ingest REPO_PATH=/path/to/repo REPO_NAME=my-svc" && exit 1)
	@test -n "$(REPO_NAME)" || (echo "Usage: make ingest REPO_PATH=/path/to/repo REPO_NAME=my-svc" && exit 1)
	uv run python ingest.py $(REPO_PATH) --repo-name $(REPO_NAME) $(if $(DOMAIN),--domain-context $(DOMAIN),)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: up
	@echo "Waiting for Qdrant to be healthy …"
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		curl -sf http://localhost:6333/healthz > /dev/null 2>&1 && break; \
		echo "  retry $$i …"; sleep 2; \
	done
	uv run pytest -v --tb=short -x
	@echo "Tests passed ✅"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: down
	docker volume rm -f qudrant-mcp_qdrant_storage qudrant-mcp_huggingface_cache 2>/dev/null || true
	@echo "Cleaned ✅"
