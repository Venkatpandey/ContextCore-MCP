"""
ContextCore-MCP — Telemetry & Prometheus Metrics
==================================================
Defines Prometheus instruments for tracking:
  - MCP tool invocations and latency
  - Text embedding computation times
  - Qdrant vector retrieval latency and hit counts
  - Ingestion statistics
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

# Tool invocation counters and latency
MCP_TOOL_CALLS_TOTAL = Counter(
    "mcp_tool_calls_total",
    "Total count of FastMCP tool invocations",
    ["tool_name", "status"],
)

MCP_TOOL_DURATION_SECONDS = Histogram(
    "mcp_tool_duration_seconds",
    "Execution duration of FastMCP tool calls in seconds",
    ["tool_name"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Embedding latency and caching
EMBEDDING_DURATION_SECONDS = Histogram(
    "mcp_embedding_duration_seconds",
    "Time taken to encode text with sentence-transformers",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

EMBEDDING_CACHE_HITS_TOTAL = Counter(
    "mcp_embedding_cache_hits_total",
    "Total count of query embedding LRU cache hits",
)

EMBEDDING_CACHE_MISSES_TOTAL = Counter(
    "mcp_embedding_cache_misses_total",
    "Total count of query embedding LRU cache misses",
)

# Qdrant search latency and result count
QDRANT_QUERY_DURATION_SECONDS = Histogram(
    "mcp_qdrant_query_duration_seconds",
    "Duration of vector query execution against Qdrant in seconds",
    ["tool_name"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

QDRANT_RESULTS_RETURNED = Histogram(
    "mcp_qdrant_results_returned",
    "Number of results returned by Qdrant per tool call",
    ["tool_name"],
    buckets=(0, 1, 2, 5, 10, 20),
)

# Ingestion pipeline metrics
INGEST_CHUNKS_TOTAL = Counter(
    "mcp_ingest_chunks_total",
    "Total number of chunks ingested into Qdrant",
    ["repo_name", "document_type"],
)

INGEST_BATCH_DURATION_SECONDS = Histogram(
    "mcp_ingest_batch_duration_seconds",
    "Duration of batch upsert operations into Qdrant",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


# ---------------------------------------------------------------------------
# Helpers & Decorators
# ---------------------------------------------------------------------------

def track_tool(tool_name: str) -> Callable:
    """Decorator to measure tool execution duration and call status."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.perf_counter() - start_time
                MCP_TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status=status).inc()
                MCP_TOOL_DURATION_SECONDS.labels(tool_name=tool_name).observe(duration)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Starlette HTTP endpoints
# ---------------------------------------------------------------------------

async def prometheus_metrics_endpoint(_request: Request) -> Response:
    """Exposes standard Prometheus metrics in text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


async def healthz_endpoint(_request: Request) -> Response:
    """Simple healthcheck endpoint."""
    return Response(content="OK", media_type="text/plain")
