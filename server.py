"""
ContextCore-MCP — FastMCP Server
==================================
Exposes SDLC-aware retrieval tools over SSE transport with built-in
Prometheus telemetry (/metrics) and healthcheck (/healthz) endpoints.

Each tool:
  1. Encodes the incoming query with `all-MiniLM-L6-v2` (384-d).
  2. Searches the Qdrant `contextcore_knowledge_base` collection with strict
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
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contextcore_knowledge_base")
TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "2000"))
PREVIEW_CALLS = int(os.getenv("PREVIEW_CALLS", "8"))
MAX_CHUNK_CHARS = 16000

# ---------------------------------------------------------------------------
# Globals — initialised at module load (once per process)
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("true", "1", "yes")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logger.info("Loading text embedding model (all-MiniLM-L6-v2) …")
_model = SentenceTransformer("all-MiniLM-L6-v2")

DUAL_EMBEDDING_ENABLED = os.getenv("DUAL_EMBEDDING", "false").lower() in ("true", "1", "yes")

_code_tokenizer: Any = None
_code_model: Any = None
if DUAL_EMBEDDING_ENABLED:
    try:
        logger.info("Loading code embedding model (microsoft/unixcoder-base) …")
        from transformers import AutoModel, AutoTokenizer
        _code_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        _code_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        _code_model.eval()
        logger.info("Loaded code embedding model (microsoft/unixcoder-base).")
    except Exception as exc:
        logger.warning("Could not load code model (microsoft/unixcoder-base): %s", exc)

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


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    dual_embedding: bool = DUAL_EMBEDDING_ENABLED,
) -> None:
    """Ensure the target Qdrant collection exists before queries run."""
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            if dual_embedding:
                vectors_config: Any = {
                    "text": VectorParams(size=384, distance=Distance.COSINE),
                    "code": VectorParams(size=768, distance=Distance.COSINE),
                }
            else:
                vectors_config = VectorParams(size=384, distance=Distance.COSINE)

            client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            logger.info("Created collection '%s' (dual_embedding=%s).", collection_name, dual_embedding)
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


@functools.lru_cache(maxsize=int(os.getenv("EMBEDDING_CACHE_SIZE", "2048")))
def _raw_embed_code(code: str) -> tuple[float, ...] | None:
    """Encode code query using UniXcoder into a 768-d normalized vector."""
    if _code_model is None or _code_tokenizer is None:
        return None
    import torch

    with EMBEDDING_DURATION_SECONDS.time():
        inputs = _code_tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = _code_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return tuple(normalized[0].tolist())


def _embed_code(code: str) -> list[float] | None:
    """Encode code query with UniXcoder LRU cache, returning 768-d list or None."""
    res = _raw_embed_code(code)
    return list(res) if res is not None else None


@functools.lru_cache(maxsize=64)
def _collection_uses_named_vectors(collection_name: str = COLLECTION_NAME) -> bool:
    """Check whether Qdrant collection is configured with named vectors."""
    try:
        col_info = _qdrant.get_collection(collection_name)
        return isinstance(col_info.config.params.vectors, dict)
    except Exception:
        return False


def _search(
    query_vector: list[float],
    must_conditions: list[FieldCondition],
    should_conditions: list[FieldCondition] | None = None,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
    tool_name: str = "search",
    target_revision: str | None = None,
    target_environment: str | None = None,
    vector_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run a Qdrant search with the supplied filter conditions and latency tracking.

    When *target_revision* is set, a nested OR filter requires results to
    match either ``git_ref`` or ``commit_sha``. When *target_environment* is
    set, results are restricted to that deployment tier.
    When the collection uses named vectors, *vector_name* specifies which vector
    index to search ('text' or 'code').
    """
    all_must = list(must_conditions)

    if target_revision:
        all_must.append(
            Filter(should=[
                FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
                FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
            ])
        )

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

    # If the collection uses named vectors, provide the target vector name
    if _collection_uses_named_vectors(COLLECTION_NAME):
        target_using = vector_name or "text"
        query_kwargs["using"] = target_using

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
            "point_id": str(hit.id),
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


def _format_results(results: list[dict[str, Any]], preview_chars: int = PREVIEW_CHARS) -> str:
    """Format search results. Complete entities under preview_chars are returned in full.
    Oversized entities are bounded with reference to get_chunk."""
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
        if r.get("point_id"):
            header += f"Point ID: {r['point_id']}\n"
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
            extra = len(calls) - PREVIEW_CALLS
            shown = ", ".join(calls[:PREVIEW_CALLS])
            if extra > 0:
                shown += f" (+{extra} more)"
            header += f"Calls: {shown}\n"

        content = r.get("content") or ""
        if len(content) > preview_chars:
            preview = content[:preview_chars]
            preview += f"\n[Preview truncated at {preview_chars}/{len(content)} chars; use get_chunk with Point ID '{r.get('point_id')}' for more.]"
        else:
            preview = content
        header += f"\n{preview}\n"
        parts.append(header)
    return "\n".join(parts)


@mcp.tool()
@track_tool("get_chunk")
def get_chunk(
    point_id: str,
    repo_name: str,
    offset: int = 0,
    max_chars: int = 8000,
    target_revision: str | None = None,
) -> str:
    """Read a search result in bounded pages by Point ID and repository.

    Use this after a context search identifies a relevant source. Pass the
    returned next offset to read more of a large chunk.
    """
    if offset < 0 or not 1 <= max_chars <= MAX_CHUNK_CHARS:
        return f"Invalid range: offset must be nonnegative and max_chars 1–{MAX_CHUNK_CHARS}."
    try:
        points = _qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("get_chunk retrieval failed: %s", exc)
        return "Chunk not found."
    if not points:
        return "Chunk not found."
    payload = points[0].payload or {}
    if payload.get("repo_name") != repo_name:
        return "Chunk not found."
    if target_revision and target_revision not in (
        payload.get("git_ref"), payload.get("commit_sha")
    ):
        return "Chunk not found at the requested revision."
    content = payload.get("content") or ""
    if offset > len(content):
        return "Offset exceeds chunk length."
    end = min(len(content), offset + max_chars)
    header = (
        f"File: {payload.get('file_path')}\n"
        f"Entity: {payload.get('entity_name')}\n"
        f"Commit: {payload.get('commit_sha', '')}\n"
        f"Ref: {payload.get('git_ref', '')}\n"
        f"Offset: {offset}; Next offset: {end}; Total chars: {len(content)}\n\n"
    )
    return header + content[offset:end]


# ── Tool 1: Epic Planning ─────────────────────────────────────────────────


@mcp.tool()
@track_tool("get_epic_planning_context")
def get_epic_planning_context(
    topic: str,
    repo_name: str,
    include_code_fallback: bool = True,
    target_revision: str | None = None,
    target_environment: str | None = None,
    top_k: int = TOP_K,
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
        top_k: Number of top matching results to retrieve. Defaults to 3.
    """
    logger.info(
        "get_epic_planning_context(topic=%r, repo=%r, code_fallback=%s, rev=%s, env=%s, top_k=%d)",
        topic,
        repo_name,
        include_code_fallback,
        target_revision,
        target_environment,
        top_k,
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
        top_k=top_k,
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
            top_k=top_k,
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
    top_k: int = TOP_K,
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
        top_k: Number of top matching results to retrieve. Defaults to 3.
    """
    logger.info(
        "get_implementation_ast(issue=%r, repo=%r, rev=%s, env=%s, top_k=%d)",
        issue_description, repo_name, target_revision, target_environment, top_k,
    )
    # Check if dual embedding / named vector 'code' is available
    use_code_vector = _collection_uses_named_vectors(COLLECTION_NAME)
    code_vec = _embed_code(issue_description) if use_code_vector else None

    if use_code_vector and code_vec is not None:
        vec = code_vec
        target_vname = "code"
    else:
        vec = _embed(issue_description)
        target_vname = "text"

    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(key="document_type", match=MatchValue(value="code")),
        ],
        top_k=top_k,
        tool_name="get_implementation_ast",
        target_revision=target_revision,
        target_environment=target_environment,
        vector_name=target_vname,
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
    top_k: int = TOP_K,
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
        top_k: Number of top matching results to retrieve. Defaults to 3.
    """
    logger.info("get_testing_context(repo=%r, rev=%s, env=%s, top_k=%d)", repo_name, target_revision, target_environment, top_k)
    # Check if dual embedding / named vector 'code' is available
    use_code_vector = _collection_uses_named_vectors(COLLECTION_NAME)
    code_vec = _embed_code(modified_code) if use_code_vector else None

    if use_code_vector and code_vec is not None:
        vec = code_vec
        using_vname: str | None = "code"
    else:
        vec = _embed(modified_code)
        using_vname = "text" if use_code_vector else None

    results = _search(
        query_vector=vec,
        must_conditions=[FieldCondition(key="repo_name", match=MatchValue(value=repo_name))],
        should_conditions=[
            FieldCondition(key="document_type", match=MatchValue(value="test_code")),
            FieldCondition(key="document_type", match=MatchValue(value="code")),
        ],
        top_k=top_k,
        tool_name="get_testing_context",
        target_revision=target_revision,
        target_environment=target_environment,
        vector_name=using_vname,
    )
    return _format_results(results)


# ── Tool 4: Code Review Guidelines ────────────────────────────────────────


@mcp.tool()
@track_tool("get_code_review_guidelines")
def get_code_review_guidelines(
    code_diff: str,
    repo_name: str,
    target_revision: str | None = None,
    target_environment: str | None = None,
    top_k: int = TOP_K,
) -> str:
    """Retrieve team coding standards and past architectural decisions to
    enforce during automated PR reviews.

    Args:
        code_diff: The unified diff or code snippet to review.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results.
        target_environment: Optional deployment tier to scope results.
        top_k: Number of top matching results to retrieve. Defaults to 3.
    """
    logger.info("get_code_review_guidelines(repo=%r, rev=%s, top_k=%d)", repo_name, target_revision, top_k)
    vec = _embed(code_diff)
    results = _search(
        query_vector=vec,
        must_conditions=[
            FieldCondition(key="repo_name", match=MatchValue(value=repo_name)),
            FieldCondition(
                key="document_type", match=MatchValue(value="business_logic")
            ),
        ],
        top_k=top_k,
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
    top_k: int = TOP_K,
) -> str:
    """Retrieve Dockerfiles, CI/CD yamls, and Terraform snippets for the
    deployment agent.

    Args:
        target_service: The service or component to deploy.
        repo_name: The repository to scope the search to.
        target_revision: Optional git ref or commit SHA to scope results.
        target_environment: Optional deployment tier to scope results.
        top_k: Number of top matching results to retrieve. Defaults to 3.
    """
    logger.info(
        "get_deployment_infrastructure(service=%r, repo=%r, rev=%s, env=%s, top_k=%d)",
        target_service,
        repo_name,
        target_revision,
        target_environment,
        top_k,
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
        top_k=top_k,
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
        # If not found by exact name and entity_name is unqualified (no dot),
        # look for qualified entities ending with f".{entity_name}"
        if not entity_points and "." not in entity_name:
            repo_must: list[Any] = [
                FieldCondition(key="repo_name", match=MatchValue(value=repo_name))
            ]
            if target_environment:
                repo_must.append(
                    FieldCondition(key="environment", match=MatchValue(value=target_environment))
                )
            repo_filter_kwargs: dict[str, Any] = {"must": repo_must}
            if target_revision:
                repo_filter_kwargs["should"] = [
                    FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
                    FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
                ]
            candidate_resp = _qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(**repo_filter_kwargs),
                limit=100,
                with_payload=True,
            )
            entity_points = [
                pt
                for pt in candidate_resp[0]
                if (pt.payload or {}).get("entity_name", "").endswith(f".{entity_name}")
            ]
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
    caller_must = [FieldCondition(key="repo_name", match=MatchValue(value=repo_name))]
    if target_environment:
        caller_must.append(
            FieldCondition(key="environment", match=MatchValue(value=target_environment))
        )
    if target_revision:
        caller_must.append(
            Filter(
                should=[
                    FieldCondition(key="git_ref", match=MatchValue(value=target_revision)),
                    FieldCondition(key="commit_sha", match=MatchValue(value=target_revision)),
                ]
            )
        )

    caller_calls_conditions = [
        FieldCondition(key="calls", match=MatchValue(value=entity_name)),
    ]
    if "." in entity_name:
        leaf = entity_name.split(".")[-1]
        caller_calls_conditions.append(
            FieldCondition(key="calls", match=MatchValue(value=leaf))
        )

    caller_must.append(Filter(should=caller_calls_conditions))
    caller_filter: dict[str, Any] = {"must": caller_must}

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
