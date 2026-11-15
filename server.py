"""
ContextCore-MCP — FastMCP Server
==================================
Exposes SDLC-aware retrieval tools over SSE transport with built-in
Prometheus telemetry (/metrics) and healthcheck (/healthz) endpoints.

Each tool:
  1. Encodes the incoming query with `all-MiniLM-L6-v2` (384-d).
  2. Searches the Qdrant `propr_knowledge_base` collection with strict
     metadata filters, including optional revision provenance filters.
  3. Returns the top-k matching chunks as structured results with
     commit SHA, git ref, environment, and call-graph lineage.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    SearchParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from starlette.applications import Starlette

from telemetry import (
    EMBEDDING_CACHE_HITS_TOTAL,
    EMBEDDING_CACHE_MISSES_TOTAL,
    EMBEDDING_DURATION_SECONDS,
    QDRANT_QUERY_DURATION_SECONDS,
    QDRANT_RESULTS_RETURNED,
    healthz_endpoint,
    prometheus_metrics_endpoint,
    track_tool,
)

# ---------------------------------------------------------------------------
# Logging (directed to stderr to preserve stdout for JSON-RPC in stdio mode)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
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
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("true", "1", "yes")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logger.info("Loading embedding model (all-MiniLM-L6-v2) …")
_model = SentenceTransformer("all-MiniLM-L6-v2")

if QDRANT_URL:
    logger.info("Connecting to Qdrant at %s (prefer_grpc=%s) …", QDRANT_URL, QDRANT_PREFER_GRPC)
    _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=QDRANT_PREFER_GRPC)
else:
    logger.info(
        "Connecting to Qdrant at %s:%s (grpc_port=%s, prefer_grpc=%s) …",
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_GRPC_PORT,
        QDRANT_PREFER_GRPC,
    )
    _qdrant = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
        api_key=QDRANT_API_KEY,
    )


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Ensure the target Qdrant collection exists before queries run."""
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s'.", collection_name)
    except Exception as exc:
        logger.warning("Could not auto-create collection '%s': %s", collection_name, exc)


ensure_collection(_qdrant, COLLECTION_NAME)

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


@functools.lru_cache(maxsize=int(os.getenv("EMBEDDING_CACHE_SIZE", "2048")))
def _raw_embed(text: str) -> tuple[float, ...]:
    """Internal cached embedding function returning an immutable tuple."""
    with EMBEDDING_DURATION_SECONDS.time():
        return tuple(_model.encode(text, show_progress_bar=False).tolist())


def _embed(text: str) -> list[float]:
    """Encode *text* into a 384-d vector with LRU caching and hit/miss telemetry."""
    hits_before = _raw_embed.cache_info().hits
    vec = list(_raw_embed(text))
    if _raw_embed.cache_info().hits > hits_before:
        EMBEDDING_CACHE_HITS_TOTAL.inc()
    else:
        EMBEDDING_CACHE_MISSES_TOTAL.inc()
    return vec


def _search(
    query_vector: list[float],
    must_conditions: list[FieldCondition],
    should_conditions: list[FieldCondition] | None = None,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
    tool_name: str = "search",
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> list[dict[str, Any]]:
    """Run a Qdrant search with the supplied filter conditions and latency tracking.

    When *target_revision* is set, an OR filter is appended so that results
    match either ``git_ref`` or ``commit_sha``.  When *target_environment* is
    set, results are restricted to that deployment tier.
    """
    all_must = list(must_conditions)

    # Revision scoping: match git_ref OR commit_sha
    if target_revision:
        # We need a nested filter to express OR between two fields.
        # Qdrant ``should`` at the top level is OR'd with ``must``, so
        # we wrap the revision check as a nested must filter with should.
        all_must.append(
            FieldCondition(key="git_ref", match=MatchValue(value=target_revision))
        )
        # Also try matching by commit_sha — we add it as a should so
        # *either* field can match.  However Qdrant's semantics require
        # us to use should_conditions at the top-level for this OR.
        revision_should = [
            FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
            FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
        ]
        # Remove the must we just added and use should instead
        all_must.pop()
        should_conditions = (should_conditions or []) + revision_should

    if target_environment:
        all_must.append(
            FieldCondition(key="environment", match=MatchValue(value=target_environment))
        )

    filter_kwargs: dict[str, Any] = {"must": all_must}
    if should_conditions:
        filter_kwargs["should"] = should_conditions

    query_kwargs: dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "query": query_vector,
        "query_filter": Filter(**filter_kwargs),
        "limit": top_k,
        "search_params": SearchParams(exact=False, hnsw_ef=128),
        "with_payload": True,
    }
    if score_threshold is not None:
        query_kwargs["score_threshold"] = score_threshold

    try:
        with QDRANT_QUERY_DURATION_SECONDS.labels(tool_name=tool_name).time():
            response = _qdrant.query_points(**query_kwargs)
    except Exception as exc:
        logger.warning(
            "Qdrant query failed (collection '%s' may not exist or is empty): %s",
            COLLECTION_NAME,
            exc,
        )
        QDRANT_RESULTS_RETURNED.labels(tool_name=tool_name).observe(0)
        return []

    results = [
        {
            "score": hit.score,
            "repo_name": hit.payload.get("repo_name"),
            "document_type": hit.payload.get("document_type"),
            "file_path": hit.payload.get("file_path"),
            "entity_name": hit.payload.get("entity_name"),
            "content": hit.payload.get("content"),
            "commit_sha": hit.payload.get("commit_sha", ""),
            "git_ref": hit.payload.get("git_ref", ""),
            "environment": hit.payload.get("environment", ""),
            "calls": hit.payload.get("calls", []),
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
        header = (
            f"--- Result {i} (score: {r['score']:.4f}) ---\n"
            f"File: {r['file_path']}\n"
            f"Entity: {r['entity_name']}\n"
            f"Type: {r['document_type']}\n"
            f"Repo: {r['repo_name']}\n"
        )
        # Provenance metadata (only rendered when present)
        commit_sha = r.get("commit_sha", "")
        git_ref = r.get("git_ref", "")
        environment = r.get("environment", "")
        if commit_sha or git_ref or environment:
            provenance_parts: list[str] = []
            if commit_sha:
                provenance_parts.append(f"Commit: {commit_sha}")
            if git_ref:
                provenance_parts.append(f"Ref: {git_ref}")
            if environment:
                provenance_parts.append(f"Env: {environment}")
            header += " | ".join(provenance_parts) + "\n"

        # Call-graph lineage
        calls = r.get("calls", [])
        if calls:
            header += f"Calls: {', '.join(calls)}\n"

        header += f"\n{r['content']}\n"
        parts.append(header)
    return "\n".join(parts)


# ── Tool 1: Epic Planning ─────────────────────────────────────────────────


@mcp.tool()
@track_tool("get_epic_planning_context")
def get_epic_planning_context(
    topic: str,
    repo_name: str,
    include_code_fallback: bool = True,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Retrieve PRDs, architecture docs, and ADRs to help the planning agent
    break down an epic into issues.

    If documentation is sparse (fewer than 3 results) or absent, and
    *include_code_fallback* is True, it automatically supplements or falls back
    to core implementation code (classes and functions) to provide ground truth reality.

    Args:
        topic: The epic topic or feature area to search for.
        repo_name: The repository to scope the search to.
        include_code_fallback: Automatically retrieve relevant code AST entities
            when documentation is insufficient. Defaults to True.
        target_revision: Optional git ref or commit SHA to scope results to a
            specific codebase snapshot.
        target_environment: Optional deployment tier to scope results
            (e.g. 'production', 'staging').
    """
    logger.info(
        "get_epic_planning_context(topic=%r, repo=%r, code_fallback=%s, rev=%s, env=%s)",
        topic,
        repo_name,
        include_code_fallback,
        target_revision,
        target_environment,
    )
    vec = _embed(topic)
    doc_results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(
                key="document_type", match=MatchValue(value="business_logic")
            ),
        ],
        score_threshold=0.25,
        tool_name="get_epic_planning_context",
        target_revision=target_revision,
        target_environment=target_environment,
    )

    code_results: list[dict[str, Any]] = []
    if len(doc_results) < 3 and include_code_fallback:
        code_results = _search(
            query_vector=vec,
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
                FieldCondition(key="document_type", match=MatchValue(value="code")),
            ],
            top_k=5,
            score_threshold=0.25,
            tool_name="get_epic_planning_context_code_fallback",
            target_revision=target_revision,
            target_environment=target_environment,
        )

    if not doc_results and not code_results:
        return "No relevant results found."

    parts: list[str] = []
    if doc_results:
        if code_results:
            parts.append("=== 📄 Architectural Documentation (Design Intent) ===")
        parts.append(_format_results(doc_results))

    if code_results:
        if doc_results:
            parts.append("=== 💻 Codebase Reality (Ground Truth ASTs — Sparse Docs Detected) ===")
        else:
            parts.append("=== 💻 Codebase Reality (Ground Truth ASTs — No Documentation Found) ===")
        parts.append(_format_results(code_results))

    return "\n\n".join(parts)


# ── Tool 2: Implementation AST ────────────────────────────────────────────


@mcp.tool()
@track_tool("get_implementation_ast")
def get_implementation_ast(
    issue_description: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Retrieve exact functions, classes, and interfaces needed for a coding
    agent to implement feature logic.

    Args:
        issue_description: A natural-language description of the issue or
            feature to implement.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results to a
            specific codebase snapshot.
        target_environment: Optional deployment tier to scope results
            (e.g. 'production', 'staging').
    """
    logger.info(
        "get_implementation_ast(issue=%r, repo=%r, rev=%s, env=%s)",
        issue_description, repo_name, target_revision, target_environment,
    )
    vec = _embed(issue_description)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(key="document_type", match=MatchValue(value="code")),
        ],
        tool_name="get_implementation_ast",
        target_revision=target_revision,
        target_environment=target_environment,
    )
    return _format_results(results)


# ── Tool 3: Testing Context ───────────────────────────────────────────────


@mcp.tool()
@track_tool("get_testing_context")
def get_testing_context(
    modified_code: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Retrieve existing test patterns and the original unmodified source code
    so the testing agent can write coverage for the exact delta.

    Args:
        modified_code: The code that was modified or added, used as a
            semantic search query.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results to a
            specific codebase snapshot.
        target_environment: Optional deployment tier to scope results
            (e.g. 'production', 'staging').
    """
    logger.info("get_testing_context(repo=%r, rev=%s, env=%s)", repo_name, target_revision, target_environment)
    vec = _embed(modified_code)

    # Build conditions
    must_conds = [
        FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
    ]
    should_conds: list[FieldCondition] = [
        FieldCondition(key="document_type", match=MatchValue(value="test_code")),
        FieldCondition(key="document_type", match=MatchValue(value="code")),
    ]

    if target_environment:
        must_conds.append(
            FieldCondition(key="environment", match=MatchValue(value=target_environment))
        )

    # For revision scoping we use the shared _search helper when there's a
    # target_revision, otherwise fall back to the original inline query for
    # backward compatibility with the should filter semantics.
    if target_revision:
        # Add revision should conditions
        should_conds.extend([
            FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
            FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
        ])

    with QDRANT_QUERY_DURATION_SECONDS.labels(
        tool_name="get_testing_context"
    ).time():
        response = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=must_conds,
                should=should_conds,
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
            "commit_sha": hit.payload.get("commit_sha", ""),
            "git_ref": hit.payload.get("git_ref", ""),
            "environment": hit.payload.get("environment", ""),
            "calls": hit.payload.get("calls", []),
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
def get_code_review_guidelines(
    code_diff: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Retrieve team coding standards and past architectural decisions to
    enforce during automated PR reviews.

    Args:
        code_diff: The unified diff or code snippet to review.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results.
        target_environment: Optional deployment tier to scope results.
    """
    logger.info("get_code_review_guidelines(repo=%r, rev=%s)", repo_name, target_revision)
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
        target_revision=target_revision,
        target_environment=target_environment,
    )
    return _format_results(results)


# ── Tool 5: Deployment Infrastructure ─────────────────────────────────────


@mcp.tool()
@track_tool("get_deployment_infrastructure")
def get_deployment_infrastructure(
    target_service: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Retrieve Dockerfiles, CI/CD yamls, and Terraform snippets for the
    deployment agent.

    Args:
        target_service: The service or component to deploy.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results.
        target_environment: Optional deployment tier to scope results.
    """
    logger.info(
        "get_deployment_infrastructure(service=%r, repo=%r, rev=%s, env=%s)",
        target_service,
        repo_name,
        target_revision,
        target_environment,
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
        target_revision=target_revision,
        target_environment=target_environment,
    )
    return _format_results(results)


# ── Tool 6: Call-Path Lineage ─────────────────────────────────────────────


@mcp.tool()
@track_tool("get_call_path")
def get_call_path(
    entity_name: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
) -> str:
    """Trace callers and callees of a function or method by its entity name.

    Uses the ``calls`` field extracted during AST ingestion to build an
    inverted call-graph view without requiring full-file token dumps.

    Returns two sections:
      1. **Callees** — functions/methods that *entity_name* calls.
      2. **Callers** — functions/methods whose ``calls`` list contains
         *entity_name*.

    Args:
        entity_name: The function or method name to trace (e.g.
            ``AuthService.login``).
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results.
        target_environment: Optional deployment tier to scope results.
    """
    logger.info(
        "get_call_path(entity=%r, repo=%r, rev=%s, env=%s)",
        entity_name, repo_name, target_revision, target_environment,
    )

    # ── 1. Find the entity itself to extract its callees ──────────────
    must_conds = [
        FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
        FieldCondition(key="entity_name", match=MatchValue(value=entity_name)),
    ]
    if target_environment:
        must_conds.append(
            FieldCondition(key="environment", match=MatchValue(value=target_environment))
        )

    filter_kwargs: dict[str, Any] = {"must": must_conds}

    if target_revision:
        filter_kwargs["should"] = [
            FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
            FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
        ]

    try:
        with QDRANT_QUERY_DURATION_SECONDS.labels(tool_name="get_call_path").time():
            entity_response = _qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(**filter_kwargs),
                limit=10,
                with_payload=True,
            )
        entity_points = entity_response[0]
    except Exception as exc:
        logger.warning("get_call_path: entity lookup failed: %s", exc)
        entity_points = []

    parts: list[str] = []

    if entity_points:
        # Collect callees from the entity's own calls field
        all_callees: list[str] = []
        seen_callees: set[str] = set()
        for pt in entity_points:
            for callee in pt.payload.get("calls", []):
                if callee not in seen_callees:
                    seen_callees.add(callee)
                    all_callees.append(callee)

        parts.append(f"=== 📞 Callees of '{entity_name}' ===")
        if all_callees:
            for c in all_callees:
                parts.append(f"  → {c}")
        else:
            parts.append("  (no outgoing calls detected)")

        parts.append("")

        # Include provenance from the first match
        first = entity_points[0].payload
        commit = first.get("commit_sha", "")
        ref = first.get("git_ref", "")
        env = first.get("environment", "")
        if commit or ref or env:
            prov = []
            if commit:
                prov.append(f"Commit: {commit}")
            if ref:
                prov.append(f"Ref: {ref}")
            if env:
                prov.append(f"Env: {env}")
            parts.append(f"Provenance: {' | '.join(prov)}")
            parts.append("")
    else:
        parts.append(f"Entity '{entity_name}' not found in repo '{repo_name}'.")
        parts.append("")

    # ── 2. Find callers — entities whose calls list contains entity_name ──
    caller_must = [
        FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
        FieldCondition(key="calls", match=MatchValue(value=entity_name)),
    ]
    if target_environment:
        caller_must.append(
            FieldCondition(key="environment", match=MatchValue(value=target_environment))
        )

    caller_filter: dict[str, Any] = {"must": caller_must}
    if target_revision:
        caller_filter["should"] = [
            FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
            FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
        ]

    try:
        caller_response = _qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(**caller_filter),
            limit=20,
            with_payload=True,
        )
        caller_points = caller_response[0]
    except Exception as exc:
        logger.warning("get_call_path: caller lookup failed: %s", exc)
        caller_points = []

    parts.append(f"=== 📥 Callers of '{entity_name}' ===")
    if caller_points:
        for pt in caller_points:
            caller_name = pt.payload.get("entity_name", "<unknown>")
            caller_file = pt.payload.get("file_path", "")
            parts.append(f"  ← {caller_name}  ({caller_file})")
    else:
        parts.append("  (no callers found)")

    QDRANT_RESULTS_RETURNED.labels(tool_name="get_call_path").observe(
        len(entity_points) + len(caller_points)
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextCore-MCP Server")
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default=os.getenv("MCP_TRANSPORT", "sse"),
        help="Transport type ('sse' or 'stdio'). Default: sse",
    )
    args, _ = parser.parse_known_args()

    if args.transport == "stdio":
        logger.info("Starting ContextCore-MCP server on stdio …")
        mcp.run(transport="stdio")
    else:
        logger.info(
            "Starting ContextCore-MCP server on %s:%d (SSE + Metrics) …",
            SERVER_HOST,
            SERVER_PORT,
        )
        mcp.run(transport="sse")
