"""
ContextCore-MCP — FastMCP Server
==================================
Exposes five SDLC-aware retrieval tools over SSE transport with built-in
Prometheus telemetry (/metrics) and healthcheck (/healthz) endpoints.

Each tool:
  1. Encodes the incoming query with `all-MiniLM-L6-v2` (384-d).
  2. Searches the Qdrant `propr_knowledge_base` collection with strict
     metadata filters.
  3. Returns the top-k matching chunks as structured results.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SearchParams,
)
from sentence_transformers import SentenceTransformer
from starlette.applications import Starlette

from telemetry import (
    EMBEDDING_DURATION_SECONDS,
    QDRANT_QUERY_DURATION_SECONDS,
    QDRANT_RESULTS_RETURNED,
    healthz_endpoint,
    prometheus_metrics_endpoint,
    track_tool,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "propr_knowledge_base")
TOP_K = 10

# ---------------------------------------------------------------------------
# Globals — initialised at module load (once per process)
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logger.info("Loading embedding model (all-MiniLM-L6-v2) …")
_model = SentenceTransformer("all-MiniLM-L6-v2")

if QDRANT_URL:
    logger.info("Connecting to Qdrant at %s …", QDRANT_URL)
    _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    logger.info("Connecting to Qdrant at %s:%s …", QDRANT_HOST, QDRANT_PORT)
    _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

# ---------------------------------------------------------------------------
# FastMCP application with Telemetry
# ---------------------------------------------------------------------------


class InstrumentedFastMCP(FastMCP):
    """FastMCP server with Prometheus metrics and health check routes."""

    def sse_app(self) -> Starlette:
        app = super().sse_app()
        app.add_route("/metrics", prometheus_metrics_endpoint, methods=["GET"])
        app.add_route("/healthz", healthz_endpoint, methods=["GET"])
        return app


SERVER_HOST = os.getenv("MCP_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("MCP_PORT", "8000"))

mcp = InstrumentedFastMCP(
    "ContextCore-MCP",
    instructions=(
        "ContextCore-MCP provides SDLC-aware retrieval tools that search a "
        "Qdrant knowledge base of ingested source code and business documents. "
        "Use the appropriate tool based on your current workflow phase."
    ),
    host=SERVER_HOST,
    port=SERVER_PORT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed(text: str) -> list[float]:
    """Encode *text* into a 384-d vector with telemetry."""
    with EMBEDDING_DURATION_SECONDS.time():
        return _model.encode(text, show_progress_bar=False).tolist()


def _search(
    query_vector: list[float],
    must_conditions: list[FieldCondition],
    should_conditions: list[FieldCondition] | None = None,
    top_k: int = TOP_K,
    tool_name: str = "search",
) -> list[dict[str, Any]]:
    """Run a Qdrant search with the supplied filter conditions and latency tracking."""
    filter_kwargs: dict[str, Any] = {"must": must_conditions}
    if should_conditions:
        filter_kwargs["should"] = should_conditions

    with QDRANT_QUERY_DURATION_SECONDS.labels(tool_name=tool_name).time():
        response = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(**filter_kwargs),
            limit=top_k,
            search_params=SearchParams(exact=False, hnsw_ef=128),
            with_payload=True,
        )

    results = [
        {
            "score": hit.score,
            "repo_name": hit.payload.get("repo_name"),
            "document_type": hit.payload.get("document_type"),
            "file_path": hit.payload.get("file_path"),
            "entity_name": hit.payload.get("entity_name"),
            "content": hit.payload.get("content"),
        }
        for hit in response.points
    ]
    QDRANT_RESULTS_RETURNED.labels(tool_name=tool_name).observe(len(results))
    return results


def _format_results(results: list[dict[str, Any]]) -> str:
    """Format search results into a readable string for the LLM consumer."""
    if not results:
        return "No relevant results found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"--- Result {i} (score: {r['score']:.4f}) ---\n"
            f"File: {r['file_path']}\n"
            f"Entity: {r['entity_name']}\n"
            f"Type: {r['document_type']}\n"
            f"Repo: {r['repo_name']}\n\n"
            f"{r['content']}\n"
        )
    return "\n".join(parts)


# ── Tool 1: Epic Planning ─────────────────────────────────────────────────


@mcp.tool()
@track_tool("get_epic_planning_context")
def get_epic_planning_context(topic: str, repo_name: str) -> str:
    """Retrieve PRDs, architecture docs, and ADRs to help the planning agent
    break down an epic into issues.

    Args:
        topic: The epic topic or feature area to search for.
        repo_name: The repository to scope the search to.
    """
    logger.info("get_epic_planning_context(topic=%r, repo=%r)", topic, repo_name)
    vec = _embed(topic)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(
                key="document_type", match=MatchValue(value="business_logic")
            ),
        ],
        tool_name="get_epic_planning_context",
    )
    return _format_results(results)


# ── Tool 2: Implementation AST ────────────────────────────────────────────


@mcp.tool()
@track_tool("get_implementation_ast")
def get_implementation_ast(issue_description: str, repo_name: str) -> str:
    """Retrieve exact functions, classes, and interfaces needed for a coding
    agent to implement feature logic.

    Args:
        issue_description: A natural-language description of the issue or
            feature to implement.
        repo_name: The repository to scope the search to.
    """
    logger.info(
        "get_implementation_ast(issue=%r, repo=%r)", issue_description, repo_name
    )
    vec = _embed(issue_description)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(key="document_type", match=MatchValue(value="code")),
        ],
        tool_name="get_implementation_ast",
    )
    return _format_results(results)


# ── Tool 3: Testing Context ───────────────────────────────────────────────


@mcp.tool()
@track_tool("get_testing_context")
def get_testing_context(modified_code: str, repo_name: str) -> str:
    """Retrieve existing test patterns and the original unmodified source code
    so the testing agent can write coverage for the exact delta.

    Args:
        modified_code: The code that was modified or added, used as a
            semantic search query.
        repo_name: The repository to scope the search to.
    """
    logger.info("get_testing_context(repo=%r)", repo_name)
    vec = _embed(modified_code)

    with QDRANT_QUERY_DURATION_SECONDS.labels(
        tool_name="get_testing_context"
    ).time():
        response = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_name", match=MatchValue(value=repo_name)
                    ),
                ],
                should=[
                    FieldCondition(
                        key="document_type", match=MatchValue(value="test_code")
                    ),
                    FieldCondition(
                        key="document_type", match=MatchValue(value="code")
                    ),
                ],
            ),
            limit=TOP_K,
            search_params=SearchParams(exact=False, hnsw_ef=128),
            with_payload=True,
        )

    formatted = [
        {
            "score": hit.score,
            "repo_name": hit.payload.get("repo_name"),
            "document_type": hit.payload.get("document_type"),
            "file_path": hit.payload.get("file_path"),
            "entity_name": hit.payload.get("entity_name"),
            "content": hit.payload.get("content"),
        }
        for hit in response.points
    ]
    QDRANT_RESULTS_RETURNED.labels(tool_name="get_testing_context").observe(
        len(formatted)
    )
    return _format_results(formatted)


# ── Tool 4: Code Review Guidelines ────────────────────────────────────────


@mcp.tool()
@track_tool("get_code_review_guidelines")
def get_code_review_guidelines(code_diff: str, repo_name: str) -> str:
    """Retrieve team coding standards and past architectural decisions to
    enforce during automated PR reviews.

    Args:
        code_diff: The unified diff or code snippet to review.
        repo_name: The repository to scope the search to.
    """
    logger.info("get_code_review_guidelines(repo=%r)", repo_name)
    vec = _embed(code_diff)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(
                key="document_type", match=MatchValue(value="business_logic")
            ),
        ],
        tool_name="get_code_review_guidelines",
    )
    return _format_results(results)


# ── Tool 5: Deployment Infrastructure ─────────────────────────────────────


@mcp.tool()
@track_tool("get_deployment_infrastructure")
def get_deployment_infrastructure(target_service: str, repo_name: str) -> str:
    """Retrieve Dockerfiles, CI/CD yamls, and Terraform snippets for the
    deployment agent.

    Args:
        target_service: The service or component to deploy.
        repo_name: The repository to scope the search to.
    """
    logger.info(
        "get_deployment_infrastructure(service=%r, repo=%r)",
        target_service,
        repo_name,
    )
    vec = _embed(target_service)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(
                key="document_type", match=MatchValue(value="infrastructure")
            ),
        ],
        tool_name="get_deployment_infrastructure",
    )
    return _format_results(results)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(
        "Starting ContextCore-MCP server on %s:%d (SSE + Metrics) …",
        SERVER_HOST,
        SERVER_PORT,
    )
    mcp.run(transport="sse")
