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

ContextCore-MCP integrates directly with [**ProPR** (`integry/propr`)](https://github.com/integry/propr) — the open-source platform that turns GitHub issues into tested, production-ready Pull Requests using AI coding agents running in isolated Docker worktrees.

ContextCore-MCP serves as ProPR's semantic memory and AST retrieval backbone:

```
┌────────────────────────────────┐
│   GitHub Issue / PR Created    │
└───────────────┬────────────────┘
                ▼
┌────────────────────────────────┐
│       ProPR Orchestrator       │
│  (https://github.com/integry/  │
│              propr)            │
└───────┬────────────────┬───────┘
        │                │
        │ FastMCP (SSE)  │
        ▼                ▼
┌────────────────────────────────┐
│        ContextCore-MCP         │
│  • get_epic_planning_context   │
│  • get_implementation_ast      │
│  • get_testing_context         │
│  • get_code_review_guidelines  │
│  • get_deployment_infrastr.    │
└───────────────┬────────────────┘
                ▼
┌────────────────────────────────┐
│   Qdrant Knowledge Base        │
│   (384-d Cosine Vectors)       │
└────────────────────────────────┘
```

### Step 1 — Ingest the ProPR Codebase & Documentation

Clone and ingest `integry/propr` (or any target repository) into your local or cloud Qdrant instance:

```bash
# Clone integry/propr
git clone https://github.com/integry/propr.git /tmp/propr

# Ingest with domain-context set to agent-orchestration
uv run python ingest.py /tmp/propr \
    --repo-name propr \
    --domain-context agent-orchestration
```

### Step 2 — SDLC Agent Invocations in ProPR Workflows

When ProPR processes an issue, its agents query ContextCore-MCP across each step of the pipeline:

**1. Planning & Task Breakdown Agent** (`get_epic_planning_context`):
```
Tool:  get_epic_planning_context
Args:  topic="docker sandbox isolation and git worktree management", repo_name="propr"

→ Returns architecture decision records (ADRs), execution environment PRDs,
  and container security policies.
```

**2. Implementation / Coding Agent** (`get_implementation_ast`):
```
Tool:  get_implementation_ast
Args:  issue_description="Add support for custom subagent execution timeout", repo_name="propr"

→ Returns the exact AST nodes for Runner execution classes, Task timeout handlers,
  and worker dispatch loops.
```

**3. Test Generation Agent** (`get_testing_context`):
```
Tool:  get_testing_context
Args:  modified_code="async def execute_task_with_timeout(self, task: Task, timeout: int)...", repo_name="propr"

→ Returns existing pytest/asyncio test patterns, fixture definitions, and the
  original unmodified execution engine source.
```

**4. Pull Request Review Agent** (`get_code_review_guidelines`):
```
Tool:  get_code_review_guidelines
Args:  code_diff="+ async def execute_task_with_timeout(...", repo_name="propr"

→ Returns PR contribution standards, linting guidelines, and error-handling requirements.
```

**5. Sandboxed Runtime & CI Agent** (`get_deployment_infrastructure`):
```
Tool:  get_deployment_infrastructure
Args:  target_service="agent-runner", repo_name="propr"

→ Returns Dockerfile specifications, docker-compose configurations, and CI/CD workflows.
```

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

## Testing with Local Repositories (e.g., TuneFlow)

You can immediately test ContextCore-MCP with any local codebase on your machine. For example, testing with an Android/Kotlin music player like `/Users/venpan/Documents/myProjects/TuneFlow`:

### Step 1 — Ingest the Local Codebase

```bash
uv run python ingest.py /Users/venpan/Documents/myProjects/TuneFlow \
    --repo-name TuneFlow \
    --domain-context music-player
```

This will walk the repository, parse Kotlin (`.kt`), Gradle infrastructure, and Markdown documentation, and upsert vectors into Qdrant.

### Step 2 — Query via FastMCP Tools

You can query the running server directly or connect your agent to `http://localhost:8000/sse`:

```python
# Query implementation AST for audio playback
get_implementation_ast(
    issue_description="audio playback background service and media player controls",
    repo_name="TuneFlow"
)
# → Returns PlaybackViewModel, AudioPlayerService, and ExoPlayer integration ASTs!

# Query design and architecture docs
get_epic_planning_context(
    topic="music player architecture and design system",
    repo_name="TuneFlow"
)
# → Returns docs/design-system.md and UI theme specifications!
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

The Tree-sitter parser handles: **Python**, **TypeScript**, **TSX**, **JavaScript**, **JSX**, **Go**, **Rust**, **Java**, **Scala** (`.scala`, `.sc`), and **Kotlin** (`.kt`, `.kts`).

Extend by adding entries to `TS_LANGUAGE_MAP` and `ENTITY_NODE_TYPES` in [`ingest.py`](ingest.py).

## License

MIT
