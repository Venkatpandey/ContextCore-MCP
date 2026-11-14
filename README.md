# ContextCore-MCP

<p align="center">
  <a href="https://github.com/Venkatpandey/ContextCore-MCP/actions/workflows/ci.yml"><img src="https://github.com/Venkatpandey/ContextCore-MCP/actions/workflows/ci.yml/badge.svg" alt="CI Status" /></a>
  <a href="https://github.com/Venkatpandey/ContextCore-MCP/releases"><img src="https://img.shields.io/github/v/release/Venkatpandey/ContextCore-MCP?logo=github&color=blue" alt="Latest Release" /></a>
  <a href="https://github.com/Venkatpandey/ContextCore-MCP/pkgs/container/contextcore-mcp"><img src="https://img.shields.io/badge/docker-ghcr.io-blue?logo=docker&logoColor=white" alt="Docker Container" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white" alt="Python 3.11+" /></a>
  <a href="https://modelcontextprotocol.io"><img src="https://img.shields.io/badge/MCP-Protocol%202024--11--05-green?logo=anthropic" alt="Model Context Protocol" /></a>
  <a href="https://qdrant.tech"><img src="https://img.shields.io/badge/Vector%20DB-Qdrant-red?logo=qdrant" alt="Vector DB: Qdrant" /></a>
</p>

> **Universal Code Intelligence & SDLC RAG Service for AI Agents** — Powered by FastMCP (SSE & Stdio), Qdrant Vector Search, and Tree-sitter AST parsing.

---

A local and cloud-ready **RAG service** that ingests multi-repository source code and business documentation into a self-hosted **Qdrant** vector database, exposing five high-precision retrieval tools via **FastMCP** to supercharge AI coding agents across the entire Software Development Lifecycle (SDLC).

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

## AI Agent Integrations

ContextCore-MCP connects with any MCP-compatible AI agent or IDE over **SSE** (`http://localhost:8000/sse`) or **Stdio** (`uv run python server.py --transport stdio`).

### 1. Claude (Claude Desktop & Claude Code)

Add to your `claude_desktop_config.json` (or project `.claude.json`):

```json
{
  "mcpServers": {
    "contextcore": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Or via Stdio:
```json
{
  "mcpServers": {
    "contextcore": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ContextCore-MCP", "python", "server.py", "--transport", "stdio"],
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

**Example Prompt:**
> *"Find the class and method AST implementations responsible for user authentication in the `auth-service` repo using ContextCore."*
> → Claude automatically calls `get_implementation_ast(issue_description="user authentication handlers", repo_name="auth-service")`.

### 2. OpenAI Codex / Cursor / Antigravity

In **Cursor** (`.cursor/mcp.json`) or **Antigravity** (`~/.gemini/config/mcp_config.json`):

```json
{
  "mcpServers": {
    "contextcore": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ContextCore-MCP", "python", "server.py", "--transport", "stdio"],
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

**Example Prompt:**
> *"Review this diff against our project guidelines and ADRs using ContextCore."*
> → Codex/Cursor calls `get_code_review_guidelines(code_diff="...", repo_name="core-service")`.

### 3. Autonomous Agents (e.g. ProPR)

ContextCore-MCP integrates cleanly with autonomous agents like [**ProPR** (`integry/propr`)](https://github.com/integry/propr) to power end-to-end issue-to-PR workflows:

```bash
# Ingest any target codebase into ContextCore
uv run python ingest.py /path/to/target-repo --repo-name propr --domain-context agent-orchestration
```

When an autonomous agent resolves an issue, it queries the 5 SDLC tools:
- `get_epic_planning_context`: Queries PRDs and ADRs for architectural context, with automatic code ground-truth fallback when documentation is sparse or absent.
- `get_implementation_ast`: Locates specific classes, functions, and interfaces.
- `get_testing_context`: Retrieves matching test conventions and fixtures.
- `get_code_review_guidelines`: Verifies proposed patches against repo rules.
- `get_deployment_infrastructure`: Retrieves Dockerfiles and CI definitions.

---

## Universal Docker & Cloud Deployment

ContextCore-MCP is packaged with standard **Docker** and **Docker Compose** configurations, allowing you to deploy it on any cloud platform or orchestrator of your choice:

- **Any VM / VPS** (AWS EC2, GCP Compute Engine, DigitalOcean, Hetzner):
  ```bash
  docker compose up -d
  ```
- **Managed Container Platforms** (AWS ECS / App Runner, GCP Cloud Run, Azure Container Apps, Railway, Fly.io):
  - FastMCP runs from the root [`Dockerfile`](Dockerfile) exposing port `8000`.
  - Qdrant runs from the official `qdrant/qdrant:latest` image with a persistent volume mounted at `/qdrant/storage`.

### AWS CDK Reference Example
For teams deploying to AWS, an infrastructure-as-code reference implementation using **AWS ECS Fargate**, **Amazon EFS** (for persistent vector storage), and an **Application Load Balancer** is provided:
- 📁 **CDK Code**: [`examples/aws-cdk/`](examples/aws-cdk/)
- 📖 **Deployment Guide**: [`docs/aws-deployment-guide.md`](docs/aws-deployment-guide.md)

---

## Automated CI/CD Ingestion (GitHub Actions)

To ensure your Qdrant knowledge base is always up-to-date with your codebase, you can automate ingestion via GitHub Actions so that **every time a commit or Pull Request is merged to `main`, the repository is automatically re-ingested**.

A ready-to-use workflow template is provided at [`examples/github-workflow-ingest.yml`](examples/github-workflow-ingest.yml).

### How It Works

```
 Developer PR Merged to `main`
              │
              ▼
 GitHub Actions Runner Launches (`ubuntu-latest`)
              │
              ├─▶ 1. Check out repository code
              ├─▶ 2. Setup Python 3.11 via `astral-sh/setup-uv`
              ├─▶ 3. Clone `ContextCore-MCP` ingestion engine
              └─▶ 4. Run `uv run python ingest.py ..`
                            │
                            ▼
           Upsert Embeddings to Qdrant Cluster
```

### Idempotency & Clean Updates

ContextCore-MCP generates **deterministic UUIDs (UUID5)** for each chunk:
```python
uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_name}:{file_path}:{chunk_index}")
```
- **Zero duplicates**: Running ingestion on every merge will safely update existing vectors in place.
- **Fast execution**: Chunks that haven't changed retain their IDs and are quickly upserted or verified in Qdrant.

### Setup Instructions for Any GitHub Repo

1. In your target repository, copy [`examples/github-workflow-ingest.yml`](examples/github-workflow-ingest.yml) to `.github/workflows/contextcore-ingest.yml`.
2. In your GitHub repository settings (**Settings** → **Secrets and variables** → **Actions**):
   - Add Secret `QDRANT_URL` (e.g. `https://your-cluster.cloud.qdrant.io:6333` or your public endpoint) OR `QDRANT_HOST` / `QDRANT_PORT`.
   - Add Secret `QDRANT_API_KEY` (if authentication is enabled).
   - (Optional) Add Variable `DOMAIN_CONTEXT` (e.g. `agent-orchestration` or `payments`).
3. Whenever a PR is merged into `main`, GitHub Actions will trigger and stream updated ASTs and documents directly into Qdrant.

---

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
| `mcp_embedding_cache_hits_total` | Counter | Total query embedding LRU cache hits |
| `mcp_embedding_cache_misses_total` | Counter | Total query embedding LRU cache misses |
| `mcp_ingest_chunks_total{repo_name, document_type}` | Counter | Ingested chunks by repository and document type |
| `mcp_ingest_batch_duration_seconds` | Histogram | Latency of batch upsert operations |
| `collections_vector_total` | Gauge | Native Qdrant: total vectors across collections |
| `cpu_cores_used` | Gauge | Native Qdrant: CPU utilization |

---

## Supported Languages

The Tree-sitter parser handles: **Python**, **TypeScript**, **TSX**, **JavaScript**, **JSX**, **Go**, **Rust**, **Java**, **Scala** (`.scala`, `.sc`), and **Kotlin** (`.kt`, `.kts`).

Extend by adding entries to `TS_LANGUAGE_MAP` and `ENTITY_NODE_TYPES` in [`ingest.py`](ingest.py).

## License

MIT
