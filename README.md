# ContextCore-MCP

A local, Dockerized RAG service that ingests multi-repo source code and business documentation into a self-hosted **Qdrant** vector database, and exposes retrieval tools via a **FastMCP** (SSE) server to support AI agents across the entire SDLC.

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────────┐
│  Your Repos  │────▶│  ingest.py  │────▶│   Qdrant (Docker)    │
│  & Docs      │     │  Tree-sitter│     │   propr_knowledge_   │
└─────────────┘     │  + Markdown │     │   base (384-d cos)   │
                     └─────────────┘     └──────────┬───────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │   server.py          │
                                          │   FastMCP SSE :8000  │
                                          ├──────────────────────┤
                                          │ get_epic_planning_   │
                                          │   context            │
                                          │ get_implementation_  │
                                          │   ast                │
                                          │ get_testing_context  │
                                          │ get_code_review_     │
                                          │   guidelines         │
                                          │ get_deployment_      │
                                          │   infrastructure     │
                                          └──────────┬───────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │   AI Agents (MCP)    │
                                          └──────────────────────┘
```

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11 |
| Package & Dep Management | `uv` (pyproject.toml + uv.lock) |
| Vector DB | Qdrant (Docker) |
| Embeddings | `all-MiniLM-L6-v2` (384-d) |
| AST Parser | tree-sitter + tree-sitter-languages |
| MCP Server | FastMCP (SSE transport) |
| Infra | Docker & Docker Compose |

## Quick Start

```bash
# 1. Clone
git clone git@github.com:Venkatpandey/ContextCore-MCP.git
cd ContextCore-MCP

# 2. Sync dependencies with uv (creates virtualenv & installs all locked deps)
uv sync

# 3. Start Qdrant + MCP server
make up

# 4. Ingest a repo
make ingest REPO_PATH=/path/to/your/repo REPO_NAME=my-service DOMAIN=payments
# Or directly:
uv run python ingest.py /path/to/your/repo --repo-name my-service --domain-context payments

# 5. Connect your AI agent to http://localhost:8000/sse
```

## Propr Integration Example

Ingest the Propr monorepo and its docs, then wire the MCP server into your agent workflow:

### Step 1 — Ingest Propr services

```bash
# Ingest the main backend
uv run python ingest.py ~/code/propr-backend \
    --repo-name propr-backend \
    --domain-context property-management

# Ingest the frontend
uv run python ingest.py ~/code/propr-frontend \
    --repo-name propr-frontend \
    --domain-context property-management

# Ingest shared business docs (PRDs, ADRs, standards)
uv run python ingest.py ~/code/propr-docs \
    --repo-name propr-docs \
    --domain-context property-management
```

### Step 2 — Use the MCP tools in your agent

**Planning agent** — breaking an epic into issues:

```
Tool:  get_epic_planning_context
Args:  topic="tenant onboarding flow", repo_name="propr-docs"

→ Returns PRDs, architecture decisions, and requirements
  related to tenant onboarding.
```

**Coding agent** — implementing a feature:

```
Tool:  get_implementation_ast
Args:  issue_description="Add lease renewal endpoint to the tenant service",
       repo_name="propr-backend"

→ Returns the existing TenantService class, lease models,
  and related API route handlers.
```

**Testing agent** — writing tests for new code:

```
Tool:  get_testing_context
Args:  modified_code="async def renew_lease(self, lease_id: str)...",
       repo_name="propr-backend"

→ Returns existing test patterns (test_tenant.py) and the
  original unmodified source code for the lease module.
```

**Review agent** — enforcing standards on a PR:

```
Tool:  get_code_review_guidelines
Args:  code_diff="+ async def renew_lease(self, ...",
       repo_name="propr-docs"

→ Returns team coding standards, naming conventions,
  and relevant ADRs.
```

**Deployment agent** — deploying a service:

```
Tool:  get_deployment_infrastructure
Args:  target_service="tenant-service",
       repo_name="propr-backend"

→ Returns the Dockerfile, CI/CD pipeline YAML,
  and Terraform configs for the tenant service.
```

### Step 3 — MCP client configuration

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "contextcore": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Or in Antigravity / Cursor / other MCP-compatible IDEs, point to `http://localhost:8000/sse`.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make up` | Start Qdrant + MCP server containers |
| `make down` | Stop containers |
| `make build` | Rebuild Docker images (no cache) |
| `make restart` | Stop then start |
| `make logs` | Tail container logs |
| `make status` | Check container + service health |
| `make ingest REPO_PATH=... REPO_NAME=...` | Ingest a repo |
| `make test` | Run integration tests (starts Qdrant if needed) |
| `make clean` | Stop containers and delete volumes |

## Payload Schema

Every vector in Qdrant uses this exact payload:

```json
{
  "repo_name": "propr-backend",
  "domain_context": "property-management",
  "document_type": "code",
  "file_path": "src/services/tenant.py",
  "entity_name": "TenantService",
  "content": "class TenantService:\n    ..."
}
```

`document_type` is strictly one of: `business_logic`, `code`, `test_code`, `infrastructure`.

## Testing

Tests run against a live Qdrant Docker container — no mocks for the data layer:

```bash
# Start Qdrant, then run tests
make test

# Or manually via uv
docker compose up -d qdrant
uv run pytest -v
```

**Test coverage:**

- File classification (test vs code vs infra)
- Tree-sitter AST chunking (Python, TypeScript)
- Markdown header splitting + contextual headers
- Full ingestion pipeline → Qdrant upsert → search
- Idempotent upserts (re-ingest produces same point count)
- Qdrant filter logic (must, should/OR)
- MCP tool search helpers + result formatting

## Observability & Telemetry (Pluggable Prometheus & Grafana)

ContextCore-MCP is designed to be completely **pluggable** with your existing monitoring infrastructure. It does not bundle a dedicated Grafana server; instead, it exposes standard Prometheus metrics and provides a ready-to-import Grafana dashboard.

### 1. Attaching an Existing Prometheus

Add the scrape jobs from [`monitoring/prometheus-scrape-example.yml`](monitoring/prometheus-scrape-example.yml) to your existing `prometheus.yml`:

```yaml
scrape_configs:
  # FastMCP server metrics (tool invocations, latency, embeddings)
  - job_name: "contextcore-mcp"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["localhost:8000"]
        labels:
          service: "contextcore-mcp"

  # Qdrant native vector database metrics (vector count, CPU, memory)
  - job_name: "qdrant"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["localhost:6333"]
        labels:
          service: "qdrant"
```

### 2. Importing the Grafana Dashboard

A pre-built, production dashboard is provided at [`monitoring/contextcore-mcp-grafana-dashboard.json`](monitoring/contextcore-mcp-grafana-dashboard.json).

To import it into any existing Grafana instance:
1. In Grafana, navigate to **Dashboards** → **New** → **Import**.
2. Click **Upload dashboard JSON file** and select `monitoring/contextcore-mcp-grafana-dashboard.json` (or paste the JSON content).
3. Select your existing **Prometheus** data source from the dropdown when prompted (`${DS_PROMETHEUS}`).
4. Click **Import**.

### Key Exported Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `mcp_tool_calls_total{tool_name, status}` | Counter | Total MCP tool calls (by tool & success/error) |
| `mcp_tool_duration_seconds{tool_name}` | Histogram | Execution duration of each tool call |
| `mcp_qdrant_query_duration_seconds{tool_name}` | Histogram | Latency of Qdrant vector retrieval |
| `mcp_qdrant_results_returned{tool_name}` | Histogram | Number of matches returned from Qdrant |
| `mcp_embedding_duration_seconds` | Histogram | Latency of SentenceTransformers embeddings |
| `mcp_ingest_chunks_total{repo_name, document_type}` | Counter | Ingested chunks by repository and document type |
| `mcp_ingest_batch_duration_seconds` | Histogram | Latency of batch upsert operations |
| `collections_vector_total` | Gauge | Native Qdrant: total vectors across collections |
| `cpu_cores_used` | Gauge | Native Qdrant: CPU utilization |

---

## Supported Languages

The Tree-sitter parser handles: **Python**, **TypeScript**, **TSX**, **JavaScript**, **JSX**, **Go**, **Rust**, **Java**, and **Scala** (`.scala`, `.sc`).

Extend by adding entries to `TS_LANGUAGE_MAP` and `ENTITY_NODE_TYPES` in [`ingest.py`](ingest.py).

Extend by adding entries to `TS_LANGUAGE_MAP` and `ENTITY_NODE_TYPES` in [`ingest.py`](ingest.py).

## License

MIT
