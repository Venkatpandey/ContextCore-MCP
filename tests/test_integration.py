"""
Docker integration tests for the ingestion pipeline.

Run with: ``make test``  (starts Qdrant via docker compose first).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ingest import (
    COLLECTION_NAME,
    Chunk,
    _chunk_code_with_ast,
    _chunk_markdown,
    _classify_code_file,
    _is_test_file,
    ensure_collection,
    upsert_chunks,
    walk_repo,
)


# ═══════════════════════════════════════════════════════════════════════════
# Unit-level: classification helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestFileClassification:
    """Verify that files are tagged with the correct document_type."""

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("tests/test_auth.py", True),
            ("src/__tests__/Login.test.tsx", True),
            ("spec/models/user_spec.rb", True),
            ("src/auth.py", False),
            ("docs/readme.md", False),
        ],
    )
    def test_is_test_file(self, path: str, expected: bool):
        assert _is_test_file(Path(path)) is expected

    @pytest.mark.parametrize(
        "path, expected",
        [
            ("src/auth.py", "code"),
            ("tests/test_auth.py", "test_code"),
            (".github/workflows/ci.yml", "infrastructure"),
        ],
    )
    def test_classify_code_file(self, path: str, expected: str):
        assert _classify_code_file(Path(path)) == expected


# ═══════════════════════════════════════════════════════════════════════════
# Unit-level: AST chunking
# ═══════════════════════════════════════════════════════════════════════════


class TestASTChunking:
    """Verify Tree-sitter correctly splits code into per-entity chunks."""

    def test_python_functions_and_classes(self):
        source = '''class Foo:
    def bar(self):
        pass

def standalone():
    return 1
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("src/foo.py"), "test-repo", "general"
        )
        names = {c.entity_name for c in chunks}
        assert "Foo" in names
        assert "standalone" in names
        assert all(c.document_type == "code" for c in chunks)

    def test_typescript_interface_and_class(self):
        source = '''export interface Config {
  port: number;
}

export class Server {
  start() {}
}
'''
        chunks = _chunk_code_with_ast(
            source, "typescript", Path("src/server.ts"), "test-repo", "general"
        )
        # Should have at least 2 chunks (interface + class, via export_statement)
        assert len(chunks) >= 2

    def test_test_file_tagged_correctly(self):
        source = '''def test_something():
    assert True
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("tests/test_x.py"), "test-repo", "general"
        )
        assert all(c.document_type == "test_code" for c in chunks)

    def test_scala_classes_objects_traits(self):
        source = '''package com.propr.services

trait TenantRepository {
  def findById(id: String): Option[Tenant]
}

case class Tenant(id: String, name: String)

object TenantService {
  def processTenant(t: Tenant): Boolean = true
}

def calculateTax(amount: Double): Double = amount * 0.2
'''
        chunks = _chunk_code_with_ast(
            source, "scala", Path("src/TenantService.scala"), "test-repo", "general"
        )
        names = {c.entity_name for c in chunks}
        assert "TenantRepository" in names
        assert "Tenant" in names
        assert "TenantService" in names
        assert "calculateTax" in names
        assert all(c.document_type == "code" for c in chunks)

    def test_empty_source_produces_no_chunks(self):
        chunks = _chunk_code_with_ast(
            "", "python", Path("src/empty.py"), "test-repo", "general"
        )
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════
# Unit-level: Markdown chunking
# ═══════════════════════════════════════════════════════════════════════════


class TestMarkdownChunking:
    """Verify Markdown is split on H1/H2 with contextual headers."""

    def test_splits_on_headers(self):
        md = """# My Doc

Intro paragraph.

## Section A

Content A.

## Section B

Content B.
"""
        chunks = _chunk_markdown(md, Path("docs/my-doc.md"), "repo", "general")
        assert len(chunks) == 3  # intro + A + B
        assert all(c.document_type == "business_logic" for c in chunks)

    def test_contextual_header_prepended(self):
        md = """# Title

## Deep Dive

Some detail.
"""
        chunks = _chunk_markdown(md, Path("docs/x.md"), "repo", "general")
        deep = [c for c in chunks if c.entity_name == "Deep Dive"][0]
        assert deep.content.startswith("[Document: Title] [Deep Dive]")

    def test_single_paragraph_no_headers(self):
        md = "Just a plain paragraph with no headers."
        chunks = _chunk_markdown(md, Path("docs/plain.md"), "repo", "general")
        assert len(chunks) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Integration: full pipeline against Docker Qdrant
# ═══════════════════════════════════════════════════════════════════════════


class TestIngestionIntegration:
    """End-to-end ingestion into a live Qdrant Docker container."""

    REPO_NAME = "propr-integration-test"

    @pytest.fixture(autouse=True)
    def _clean_collection(self, qdrant: QdrantClient):
        """Delete and recreate the collection before each test class run."""
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)
        yield
        # Cleanup after
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_walk_repo_extracts_all_types(self, sample_repo: Path):
        """walk_repo should produce code, test_code, business_logic, and infra chunks."""
        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        types = {c.document_type for c in chunks}
        assert "code" in types, f"Missing 'code' — got {types}"
        assert "test_code" in types, f"Missing 'test_code' — got {types}"
        assert "business_logic" in types, f"Missing 'business_logic' — got {types}"
        assert "infrastructure" in types, f"Missing 'infrastructure' — got {types}"

    def test_upsert_and_search(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Ingest the sample repo and verify vectors are searchable in Qdrant."""
        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        count = upsert_chunks(qdrant, embed_model, chunks)
        assert count == len(chunks)
        assert count > 0

        # Verify collection has points
        info = qdrant.get_collection(COLLECTION_NAME)
        assert info.points_count == count

    def test_search_filters_by_document_type(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Search with document_type filter should return only matching chunks."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)

        vec = embed_model.encode("authentication login").tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_type", match=MatchValue(value="code")
                    ),
                    FieldCondition(
                        key="repo_name",
                        match=MatchValue(value=self.REPO_NAME),
                    ),
                ]
            ),
            limit=5,
            with_payload=True,
        ).points
        assert len(results) > 0
        assert all(r.payload["document_type"] == "code" for r in results)

    def test_search_business_logic_returns_prd(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Searching for 'payment processing' in business_logic should find the PRD."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)

        vec = embed_model.encode("payment processing stripe").tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value="business_logic"),
                    ),
                ]
            ),
            limit=3,
            with_payload=True,
        ).points
        assert len(results) > 0
        assert any("payment" in r.payload["content"].lower() for r in results)

    def test_idempotent_upsert(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Upserting the same chunks twice should not duplicate vectors."""
        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)
        first_count = qdrant.get_collection(COLLECTION_NAME).points_count

        # Upsert again
        upsert_chunks(qdrant, embed_model, chunks)
        second_count = qdrant.get_collection(COLLECTION_NAME).points_count

        assert first_count == second_count

    def test_search_testing_context_or_filter(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """The OR filter (test_code | code) should return both types."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)

        vec = embed_model.encode("login authentication test").tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_name",
                        match=MatchValue(value=self.REPO_NAME),
                    ),
                ],
                should=[
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value="test_code"),
                    ),
                    FieldCondition(
                        key="document_type",
                        match=MatchValue(value="code"),
                    ),
                ],
            ),
            limit=10,
            with_payload=True,
        ).points
        types_returned = {r.payload["document_type"] for r in results}
        # Should have at least one of the OR'd types
        assert types_returned & {"test_code", "code"}
        # Should NOT have business_logic or infrastructure
        assert "business_logic" not in types_returned
        assert "infrastructure" not in types_returned


class TestGrpcIntegration:
    """Integration test suite explicitly verifying Qdrant gRPC transport operations."""

    GRPC_COLLECTION = "grpc_integration_test_collection"

    @pytest.fixture
    def grpc_client(self) -> QdrantClient:
        """Create a live Qdrant client explicitly using gRPC transport."""
        client = QdrantClient(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
        try:
            client.delete_collection(self.GRPC_COLLECTION)
        except Exception:
            pass
        yield client
        try:
            client.delete_collection(self.GRPC_COLLECTION)
        except Exception:
            pass

    def test_grpc_collection_upsert_and_query(
        self,
        grpc_client: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Verify that ensure_collection, upsert_chunks, and query_points work over gRPC."""
        from ingest import ensure_collection, upsert_chunks, walk_repo

        # 1. Ensure collection over gRPC
        ensure_collection(grpc_client, collection_name=self.GRPC_COLLECTION)
        collections = [c.name for c in grpc_client.get_collections().collections]
        assert self.GRPC_COLLECTION in collections

        # 2. Extract and upsert chunks over gRPC
        chunks = list(walk_repo(sample_repo, "grpc-repo", "payments"))
        assert len(chunks) > 0
        count = upsert_chunks(
            grpc_client,
            embed_model,
            chunks,
            collection_name=self.GRPC_COLLECTION,
        )
        assert count == len(chunks)

        # 3. Query points over gRPC
        query_vec = embed_model.encode("user authentication").tolist()
        results = grpc_client.query_points(
            collection_name=self.GRPC_COLLECTION,
            query=query_vec,
            limit=5,
            with_payload=True,
        ).points
        assert len(results) > 0
        assert results[0].payload["repo_name"] == "grpc-repo"
