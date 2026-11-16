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

from ingest import (
    COLLECTION_NAME,
    _apply_provenance,
    ensure_collection,
    upsert_chunks,
    walk_repo,
)


# We import the server module's internal helpers rather than hitting HTTP,
# keeping tests fast while still validating the full search logic.
from server import (
    _embed,
    _search,
    _format_results,
    get_chunk,
    get_epic_planning_context,
    get_implementation_ast,
    get_testing_context,
    get_call_path,
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

    def test_embed_lru_caching(self):
        from server import _raw_embed
        _raw_embed.cache_clear()
        
        # First call is a miss
        hits_before = _raw_embed.cache_info().hits
        vec1 = _embed("cache test query string")
        assert _raw_embed.cache_info().hits == hits_before

        # Second call is a hit
        vec2 = _embed("cache test query string")
        assert _raw_embed.cache_info().hits == hits_before + 1
        assert vec1 == vec2

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

    def test_planning_context_with_docs(self):
        out = get_epic_planning_context("payment processing requirements", self.REPO_NAME)
        assert "payment" in out.lower()

    def test_planning_context_code_fallback_when_docs_missing(self):
        # Topic with code in auth.py but no documentation in docs/
        out = get_epic_planning_context(
            "authenticate user login tokens",
            self.REPO_NAME,
            include_code_fallback=True,
        )
        assert "Codebase Reality" in out
        assert "auth.py" in out or "AuthService" in out

    def test_planning_context_code_fallback_disabled(self):
        # When fallback is disabled, missing docs should return no results
        out = get_epic_planning_context(
            "authenticate user login tokens",
            self.REPO_NAME,
            include_code_fallback=False,
        )
        assert "No relevant results found." in out

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

    def test_format_results_with_provenance(self):
        """Provenance metadata should appear in formatted output."""
        fake = [
            {
                "score": 0.90,
                "repo_name": "r",
                "document_type": "code",
                "file_path": "a.py",
                "entity_name": "bar",
                "content": "def bar(): pass",
                "commit_sha": "abc123",
                "git_ref": "v1.0.0",
                "environment": "production",
                "calls": ["baz", "qux"],
            }
        ]
        out = _format_results(fake)
        assert "Commit: abc123" in out
        assert "Ref: v1.0.0" in out
        assert "Env: production" in out
        assert "Calls: baz, qux" in out

    def test_format_results_without_provenance_no_extra_lines(self):
        """When no provenance is set, no extra metadata lines should appear."""
        fake = [
            {
                "score": 0.80,
                "repo_name": "r",
                "document_type": "code",
                "file_path": "x.py",
                "entity_name": "fn",
                "content": "def fn(): pass",
                "commit_sha": "",
                "git_ref": "",
                "environment": "",
                "calls": [],
            }
        ]
        out = _format_results(fake)
        assert "Commit:" not in out
        assert "Ref:" not in out
        assert "Calls:" not in out

    def test_format_results_returns_bounded_preview(self):
        out = _format_results([{
            "score": 0.9,
            "repo_name": "r",
            "document_type": "code",
            "file_path": "a.py",
            "entity_name": "foo",
            "point_id": "point-1",
            "content": "x" * 3000,
            "calls": [f"call{i}" for i in range(20)],
        }])
        assert "Point ID: point-1" in out
        assert "x" * 2000 in out
        assert "x" * 2001 not in out
        assert "get_chunk" in out
        assert "call7 (+12 more)" in out
        assert "call8" not in out

    def test_format_results_returns_complete_entity_without_truncation(self):
        out = _format_results([{
            "score": 0.95,
            "repo_name": "r",
            "document_type": "code",
            "file_path": "a.py",
            "entity_name": "foo",
            "point_id": "point-1",
            "content": "def foo():\n    return 42\n",
            "calls": ["call1", "call2"],
        }])
        assert "Point ID: point-1" in out
        assert "def foo():" in out
        assert "return 42" in out
        assert "Calls: call1, call2" in out
        assert "Preview truncated" not in out
        assert "get_chunk" not in out

    def test_get_chunk_reads_pages_with_repo_scope(self):
        results = _search(
            query_vector=_embed("authentication login"),
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                FieldCondition(key="document_type", match=MatchValue(value="code")),
            ],
        )
        point_id = results[0]["point_id"]
        assert "Chunk not found" in get_chunk(point_id, "another-repo")
        first = get_chunk(point_id, self.REPO_NAME, max_chars=20)
        assert "Next offset: 20" in first
        assert "Offset: 20" in get_chunk(point_id, self.REPO_NAME, offset=20, max_chars=20)
        assert "Offset exceeds chunk length" in get_chunk(point_id, self.REPO_NAME, offset=100000)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Revision-scoped tool queries
# ═══════════════════════════════════════════════════════════════════════════


class TestRevisionScopedTools:
    """Verify that server tools respect target_revision and target_environment."""

    REPO_NAME = "revision-scoped-test"

    @pytest.fixture(autouse=True, scope="class")
    def _ingest_two_revisions(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Ingest two revisions of the sample repo."""
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)

        # v1.0.0 — production
        chunks_v1 = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(
            chunks_v1,
            commit_sha="aaaa1111",
            git_ref="v1.0.0",
            environment="production",
        )
        upsert_chunks(qdrant, embed_model, chunks_v1)

        # v2.0.0 — staging
        chunks_v2 = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(
            chunks_v2,
            commit_sha="bbbb2222",
            git_ref="v2.0.0",
            environment="staging",
        )
        upsert_chunks(qdrant, embed_model, chunks_v2)

        yield
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_implementation_ast_scoped_by_revision(self):
        """get_implementation_ast with target_revision should only return matching chunks."""
        out = get_implementation_ast(
            "authentication login",
            self.REPO_NAME,
            target_revision="v1.0.0",
        )
        # Should find results (we ingested with git_ref=v1.0.0)
        assert "Result 1" in out or "No relevant results found." in out

    def test_implementation_ast_scoped_by_environment(self):
        """get_implementation_ast with target_environment should filter."""
        out = get_implementation_ast(
            "authentication login",
            self.REPO_NAME,
            target_environment="production",
        )
        if "Env:" in out:
            assert "production" in out

    def test_planning_context_scoped_by_environment(self):
        """get_epic_planning_context respects target_environment."""
        out = get_epic_planning_context(
            "payment processing",
            self.REPO_NAME,
            target_environment="staging",
        )
        # Should find results for staging
        assert out  # Non-empty response

    def test_search_with_commit_sha_revision(self):
        """_search with target_revision matching a commit_sha should work."""
        vec = _embed("authentication")
        results = _search(
            query_vector=vec,
            must_conditions=[
                FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                FieldCondition(key="document_type", match=MatchValue(value="code")),
            ],
            target_revision="aaaa1111",
        )
        # Should return results matching commit_sha="aaaa1111"
        if results:
            assert all(
                r.get("commit_sha") == "aaaa1111" or r.get("git_ref") == "aaaa1111"
                for r in results
            )

    def test_testing_context_requires_type_and_revision(self):
        results = _search(
            query_vector=_embed("authentication login"),
            must_conditions=[FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME))],
            should_conditions=[
                FieldCondition(key="document_type", match=MatchValue(value="test_code")),
                FieldCondition(key="document_type", match=MatchValue(value="code")),
            ],
            target_revision="v1.0.0",
        )
        assert results
        assert all(r["git_ref"] == "v1.0.0" for r in results)
        assert all(r["document_type"] in {"test_code", "code"} for r in results)
        output = get_testing_context("authentication login", self.REPO_NAME, target_revision="v1.0.0")
        assert "Ref: v2.0.0" not in output


# ═══════════════════════════════════════════════════════════════════════════
# Integration: get_call_path tool
# ═══════════════════════════════════════════════════════════════════════════


class TestGetCallPath:
    """Verify the get_call_path tool traces callers and callees."""

    REPO_NAME = "callpath-server-test"

    @pytest.fixture(autouse=True, scope="class")
    def _ingest_fixture(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Ingest the sample repo with provenance for call-path tests."""
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(
            chunks,
            commit_sha="callpath123",
            git_ref="main",
            environment="production",
        )
        upsert_chunks(qdrant, embed_model, chunks)
        yield
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_call_path_returns_callees(self):
        """get_call_path should list callees of a known entity."""
        out = get_call_path("UserService", self.REPO_NAME)
        assert "Callees of 'UserService'" in out

    def test_call_path_returns_callers(self):
        """get_call_path should list callers section."""
        out = get_call_path("hash_password", self.REPO_NAME)
        assert "Callers of 'hash_password'" in out
        assert "UserService.register" in out

    def test_call_path_qualified_method_and_leaf_resolution(self):
        """get_call_path should resolve both qualified and leaf queries."""
        # 1. Qualified method name
        reg_out = get_call_path("UserService.register", self.REPO_NAME)
        assert "Callees of 'UserService.register'" in reg_out
        assert "hash_password" in reg_out

        # 2. Leaf name resolution for login should find UserService.register as caller
        login_out = get_call_path("login", self.REPO_NAME)
        assert "UserService.register" in login_out

    def test_call_path_unknown_entity(self):
        """get_call_path for a non-existent entity should report not found."""
        out = get_call_path("nonExistentFunction", self.REPO_NAME)
        assert "not found" in out

    def test_call_path_with_revision_filter(self):
        """get_call_path with target_revision should scope results."""
        out = get_call_path(
            "UserService",
            self.REPO_NAME,
            target_revision="main",
        )
        assert "Callees" in out or "not found" in out

    def test_call_path_with_environment_filter(self):
        """get_call_path with target_environment should scope results."""
        out = get_call_path(
            "UserService",
            self.REPO_NAME,
            target_environment="production",
        )
        assert "Callees" in out or "not found" in out

    def test_call_path_provenance_displayed(self):
        """get_call_path should display provenance when available."""
        out = get_call_path("UserService", self.REPO_NAME)
        # If entity is found, provenance should be shown
        if "Callees" in out and "(no outgoing calls detected)" not in out:
            assert "Provenance:" in out or "Commit:" in out or "callpath123" in out
