"""
Tests for Prometheus telemetry endpoints and metric instrumentation.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from server import mcp
from telemetry import (
    MCP_TOOL_CALLS_TOTAL,
    MCP_TOOL_DURATION_SECONDS,
    track_tool,
)


@pytest.fixture
def client() -> TestClient:
    """Starlette test client wrapping FastMCP's SSE app."""
    app = mcp.sse_app()
    return TestClient(app)


def test_healthz_endpoint(client: TestClient):
    """The /healthz endpoint should return 200 OK."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.text == "OK"


def test_metrics_endpoint(client: TestClient):
    """The /metrics endpoint should expose Prometheus text metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "mcp_tool_calls_total" in response.text
    assert "mcp_tool_duration_seconds" in response.text


def test_track_tool_decorator():
    """Verify that track_tool records calls and latency."""
    @track_tool("test_metric_tool")
    def sample_tool(succeed: bool = True):
        if not succeed:
            raise RuntimeError("Failure")
        return "ok"

    # Success call
    assert sample_tool(succeed=True) == "ok"

    # Failing call
    with pytest.raises(RuntimeError):
        sample_tool(succeed=False)

    # Verify metric increments
    success_count = MCP_TOOL_CALLS_TOTAL.labels(
        tool_name="test_metric_tool", status="success"
    )._value.get()
    error_count = MCP_TOOL_CALLS_TOTAL.labels(
        tool_name="test_metric_tool", status="error"
    )._value.get()

    assert success_count >= 1
    assert error_count >= 1


def test_metrics_endpoint_includes_cache_metrics(client: TestClient):
    """Verify that LRU cache hits and misses are tracked and exported on /metrics."""
    from server import _embed, _raw_embed

    _raw_embed.cache_clear()
    unique_query = "unique integration telemetry query test"
    
    # 1. First invocation: Cache Miss
    _embed(unique_query)
    
    # 2. Second invocation: Cache Hit
    _embed(unique_query)
    
    # 3. Verify /metrics output
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mcp_embedding_cache_hits_total" in response.text
    assert "mcp_embedding_cache_misses_total" in response.text
