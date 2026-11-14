"""
Integration tests for the FastMCP server tools.

These tests spin up the real embedding model and hit a live Qdrant instance.
Run with: ``make test``
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION_NAME, ensure_collection, upsert_chunks, walk_repo


# We import the server module's internal helpers rather than hitting HTTP,
# keeping tests fast while still validating the full search logic.
from server import (
    _embed,
    _search,
    _format_results,
)
from qdrant_client.models import FieldCondition, MatchValue


class TestServerTools:
    """Validate the MCP tool search logic against live Qdrant."""

    REPO_NAME = "propr-server-test"

    @pytest.fixture(autouse=True, scope="class")
    def _ingest_fixture(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Ingest the sample repo once for all tests in this class."""
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)
        yield
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    # ── Embedding sanity ──────────────────────────────────────────────

    def test_embed_returns_384_dim(self):
        vec = _embed("hello world")
        assert len(vec) == 384

    # ── Tool: get_epic_planning_context ────────────────────────────────

    def test_planning_context_finds_prd(self):
        vec = _embed("payment processing requirements")
        results = _search(
            query_vector=vec,
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                FieldCondition(key="document_type", match=MatchValue(value="business_logic")),
            ],
        )
        assert len(results) > 0
        assert any("payment" in r["content"].lower() for r in results)

    # ── Tool: get_implementation_ast ───────────────────────────────────

    def test_implementation_ast_finds_code(self):
        vec = _embed("authenticate user login")
        results = _search(
            query_vector=vec,
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                FieldCondition(key="document_type", match=MatchValue(value="code")),
            ],
        )
        assert len(results) > 0
        assert all(r["document_type"] == "code" for r in results)

    # ── Tool: get_deployment_infrastructure ────────────────────────────

    def test_deployment_infra_finds_dockerfile(self):
        vec = _embed("docker container deployment")
        results = _search(
            query_vector=vec,
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                FieldCondition(key="document_type", match=MatchValue(value="infrastructure")),
            ],
        )
        assert len(results) > 0
        assert any("Dockerfile" in r["file_path"] or "ci.yml" in r["file_path"] for r in results)

    # ── _format_results ───────────────────────────────────────────────

    def test_format_results_empty(self):
        assert _format_results([]) == "No relevant results found."

    def test_format_results_structure(self):
        fake = [
            {
                "score": 0.95,
                "repo_name": "r",
                "document_type": "code",
                "file_path": "a.py",
                "entity_name": "foo",
                "content": "def foo(): pass",
            }
        ]
        out = _format_results(fake)
        assert "Result 1" in out
        assert "0.9500" in out
        assert "def foo()" in out
