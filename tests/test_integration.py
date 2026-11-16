"""
Docker integration tests for the ingestion pipeline.

Run with: ``make test``  (starts Qdrant via docker compose first).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ingest import (
    COLLECTION_NAME,
    Chunk,
    _apply_provenance,
    _chunk_code_with_ast,
    _chunk_markdown,
    _classify_code_file,
    _extract_calls,
    _is_test_file,
    _resolve_commit_sha,
    ensure_collection,
    upsert_chunks,
    walk_repo,
)


def test_revision_provenance_requires_a_pinned_clean_snapshot(tmp_path: Path):
    sha = "a" * 40
    with pytest.raises(ValueError, match="requires a Git checkout"):
        _resolve_commit_sha(tmp_path, "", "main")
    assert _resolve_commit_sha(tmp_path, sha, "main") == sha
    with pytest.raises(ValueError, match="full lowercase"):
        _resolve_commit_sha(tmp_path, "abc", "main")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "source.py"
    source.write_text("print('v1')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test",
                    "-c", "user.email=test@example.com", "commit", "-qm", "initial"], check=True)
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    assert _resolve_commit_sha(tmp_path, "", "main") == head
    with pytest.raises(ValueError, match="differs"):
        _resolve_commit_sha(tmp_path, sha, "main")
    source.write_text("print('v2')\n")
    with pytest.raises(ValueError, match="tracked changes"):
        _resolve_commit_sha(tmp_path, head, "main")


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
# Unit-level: Call-graph extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestCallGraphExtraction:
    """Verify that _extract_calls captures function/method invocations."""

    def test_python_simple_calls(self):
        source = '''def process():
    result = compute()
    save(result)
    return result
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("src/proc.py"), "test-repo", "general"
        )
        assert len(chunks) == 1
        calls = chunks[0].calls
        assert "compute" in calls
        assert "save" in calls

    def test_python_method_calls(self):
        source = '''class Handler:
    def handle(self, request):
        data = self.validate(request)
        self.process(data)
        return self.respond(data)
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("src/handler.py"), "test-repo", "general"
        )
        assert len(chunks) == 1
        calls = chunks[0].calls
        assert any("validate" in c for c in calls)
        assert any("process" in c for c in calls)
        assert any("respond" in c for c in calls)

    def test_python_no_calls_returns_empty(self):
        source = '''def constant():
    return 42
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("src/const.py"), "test-repo", "general"
        )
        assert len(chunks) == 1
        assert chunks[0].calls == []

    def test_calls_deduplicated(self):
        source = '''def repetitive():
    foo()
    foo()
    foo()
'''
        chunks = _chunk_code_with_ast(
            source, "python", Path("src/rep.py"), "test-repo", "general"
        )
        assert len(chunks) == 1
        assert chunks[0].calls.count("foo") == 1

    def test_typescript_calls(self):
        source = '''export function orchestrate(input: string): string {
  const validated = validate(input);
  const result = transform(validated);
  return format(result);
}
'''
        chunks = _chunk_code_with_ast(
            source, "typescript", Path("src/orch.ts"), "test-repo", "general"
        )
        assert len(chunks) >= 1
        # Find the chunk for orchestrate (may be inside an export_statement)
        orch_chunk = next(
            (c for c in chunks if "orchestrate" in c.content), None
        )
        assert orch_chunk is not None
        assert "validate" in orch_chunk.calls
        assert "transform" in orch_chunk.calls
        assert "format" in orch_chunk.calls

    def test_calls_included_in_payload(self):
        """Verify calls appear in to_payload() output."""
        chunk = Chunk(
            repo_name="r",
            domain_context="general",
            document_type="code",
            file_path="a.py",
            entity_name="fn",
            content="def fn(): pass",
            calls=["bar", "baz"],
        )
        payload = chunk.to_payload()
        assert payload["calls"] == ["bar", "baz"]


# ═══════════════════════════════════════════════════════════════════════════
# Unit-level: Revision provenance
# ═══════════════════════════════════════════════════════════════════════════


class TestRevisionProvenance:
    """Verify provenance metadata on Chunk dataclass and payload."""

    def test_chunk_default_provenance_empty(self):
        chunk = Chunk(
            repo_name="r",
            domain_context="general",
            document_type="code",
            file_path="a.py",
            entity_name="fn",
            content="def fn(): pass",
        )
        assert chunk.commit_sha == ""
        assert chunk.git_ref == ""
        assert chunk.environment == ""
        assert chunk.build_id == ""
        assert chunk.artifact_digest == ""

    def test_provenance_in_payload(self):
        chunk = Chunk(
            repo_name="r",
            domain_context="general",
            document_type="code",
            file_path="a.py",
            entity_name="fn",
            content="def fn(): pass",
            commit_sha="abc123",
            git_ref="v1.0.0",
            environment="production",
            build_id="build-42",
            artifact_digest="sha256:deadbeef",
        )
        payload = chunk.to_payload()
        assert payload["commit_sha"] == "abc123"
        assert payload["git_ref"] == "v1.0.0"
        assert payload["environment"] == "production"
        assert payload["build_id"] == "build-42"
        assert payload["artifact_digest"] == "sha256:deadbeef"

    def test_apply_provenance_stamps_all_chunks(self):
        chunks = [
            Chunk(
                repo_name="r",
                domain_context="general",
                document_type="code",
                file_path="a.py",
                entity_name="fn1",
                content="def fn1(): pass",
            ),
            Chunk(
                repo_name="r",
                domain_context="general",
                document_type="code",
                file_path="b.py",
                entity_name="fn2",
                content="def fn2(): pass",
            ),
        ]
        _apply_provenance(
            chunks,
            commit_sha="abc123",
            git_ref="main",
            environment="staging",
        )
        for c in chunks:
            assert c.commit_sha == "abc123"
            assert c.git_ref == "main"
            assert c.environment == "staging"

    def test_deterministic_id_includes_provenance(self):
        """Same content at different revisions should produce different IDs."""
        base_kwargs = dict(
            repo_name="r",
            domain_context="general",
            document_type="code",
            file_path="a.py",
            entity_name="fn",
            content="def fn(): pass",
        )
        c1 = Chunk(**base_kwargs, commit_sha="aaa", git_ref="v1.0", environment="prod")
        c2 = Chunk(**base_kwargs, commit_sha="bbb", git_ref="v2.0", environment="prod")
        c3 = Chunk(**base_kwargs, commit_sha="aaa", git_ref="v1.0", environment="prod")
        assert c1.deterministic_id() != c2.deterministic_id()
        assert c1.deterministic_id() == c3.deterministic_id()


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

    def test_walk_repo_parses_mjs(self, tmp_path: Path):
        source = tmp_path / "repair-context.mjs"
        source.write_text("export function selectRepairContext() { return true; }\n")
        chunks = list(walk_repo(tmp_path, self.REPO_NAME, "agents"))
        assert any(c.file_path == source.name and "selectRepairContext" in c.content for c in chunks)

    def test_walk_repo_indexes_shell_script(self, tmp_path: Path):
        source = tmp_path / "check-branch-name.sh"
        source.write_text("#!/bin/sh\n# Validate a branch name\ngrep -qE '^(feat|fix)/'\n")
        chunks = list(walk_repo(tmp_path, self.REPO_NAME, "agents"))
        assert any(c.file_path == source.name and c.document_type == "code" for c in chunks)

    def test_walk_repo_uses_tracked_files_in_git(self, tmp_path: Path):
        import subprocess

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        (tmp_path / "tracked.py").write_text("def tracked(): pass\n")
        (tmp_path / "untracked.py").write_text("def untracked(): pass\n")
        subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
        chunks = list(walk_repo(tmp_path, self.REPO_NAME, "agents"))
        assert {c.file_path for c in chunks} == {"tracked.py"}

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


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Provenance metadata round-trip through Qdrant
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenanceIntegration:
    """Verify provenance metadata survives ingestion and is filterable."""

    REPO_NAME = "provenance-test"

    @pytest.fixture(autouse=True)
    def _clean_collection(self, qdrant: QdrantClient):
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)
        yield
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_provenance_stored_and_retrievable(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Chunks ingested with provenance should carry it in Qdrant payloads."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(
            chunks,
            commit_sha="a1b2c3d4e5f67890",
            git_ref="v1.2.0",
            environment="production",
            build_id="ci-123",
            artifact_digest="sha256:cafe",
        )
        upsert_chunks(qdrant, embed_model, chunks)

        # Scroll all points and check provenance
        all_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
        )
        assert len(all_points) > 0
        for pt in all_points:
            assert pt.payload["commit_sha"] == "a1b2c3d4e5f67890"
            assert pt.payload["git_ref"] == "v1.2.0"
            assert pt.payload["environment"] == "production"
            assert pt.payload["build_id"] == "ci-123"
            assert pt.payload["artifact_digest"] == "sha256:cafe"

    def test_filter_by_environment(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Qdrant should be able to filter chunks by environment."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))

        # Ingest half as production, half as staging
        prod_chunks = chunks[: len(chunks) // 2]
        stg_chunks = chunks[len(chunks) // 2 :]
        _apply_provenance(prod_chunks, commit_sha="aaa", environment="production")
        _apply_provenance(stg_chunks, commit_sha="bbb", environment="staging")

        upsert_chunks(qdrant, embed_model, prod_chunks + stg_chunks)

        vec = embed_model.encode("authentication login").tolist()
        prod_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_name", match=MatchValue(value=self.REPO_NAME)
                    ),
                    FieldCondition(
                        key="environment", match=MatchValue(value="production")
                    ),
                ]
            ),
            limit=50,
            with_payload=True,
        ).points

        assert len(prod_results) > 0
        assert all(
            r.payload["environment"] == "production" for r in prod_results
        )

    def test_filter_by_git_ref(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Qdrant should filter chunks by git_ref."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(chunks, commit_sha="abc", git_ref="v2.0.0")
        upsert_chunks(qdrant, embed_model, chunks)

        vec = embed_model.encode("authentication").tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="repo_name", match=MatchValue(value=self.REPO_NAME)
                    ),
                    FieldCondition(
                        key="git_ref", match=MatchValue(value="v2.0.0")
                    ),
                ]
            ),
            limit=10,
            with_payload=True,
        ).points

        assert len(results) > 0
        assert all(r.payload["git_ref"] == "v2.0.0" for r in results)

    def test_multi_revision_isolation(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Two revisions of the same repo should not bleed into each other's queries."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks_v1 = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(chunks_v1, commit_sha="v1hash", git_ref="v1.0.0", environment="production")
        upsert_chunks(qdrant, embed_model, chunks_v1)

        chunks_v2 = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        _apply_provenance(chunks_v2, commit_sha="v2hash", git_ref="v2.0.0", environment="staging")
        upsert_chunks(qdrant, embed_model, chunks_v2)

        vec = embed_model.encode("authentication").tolist()

        # Query v1 only
        v1_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=Filter(
                must=[
                    FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                    FieldCondition(key="git_ref", match=MatchValue(value="v1.0.0")),
                ]
            ),
            limit=20,
            with_payload=True,
        ).points

        assert len(v1_results) > 0
        assert all(r.payload["git_ref"] == "v1.0.0" for r in v1_results)
        assert all(r.payload["environment"] == "production" for r in v1_results)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Call-graph data round-trip through Qdrant
# ═══════════════════════════════════════════════════════════════════════════


class TestCallGraphIntegration:
    """Verify call-graph lineage survives ingestion and is queryable."""

    REPO_NAME = "callgraph-test"

    @pytest.fixture(autouse=True)
    def _clean_collection(self, qdrant: QdrantClient):
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        ensure_collection(qdrant)
        yield
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def test_calls_stored_in_qdrant_payload(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Chunks with calls should carry them through to Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)

        # Find the UserService.register chunk (it calls hash_password, self.auth.login, _send_welcome_email)
        all_points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                    FieldCondition(key="entity_name", match=MatchValue(value="UserService")),
                ]
            ),
            limit=10,
            with_payload=True,
        )

        # UserService is a class — it should have calls from its methods
        if all_points:
            calls = all_points[0].payload.get("calls", [])
            assert isinstance(calls, list)
            # Should contain at least hash_password or self.auth.login
            assert len(calls) > 0

    def test_find_callers_via_calls_filter(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """We can find callers of a function by filtering on the calls field."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        chunks = list(walk_repo(sample_repo, self.REPO_NAME, "payments"))
        upsert_chunks(qdrant, embed_model, chunks)

        # Find entities that call hash_password
        callers, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="repo_name", match=MatchValue(value=self.REPO_NAME)),
                    FieldCondition(key="calls", match=MatchValue(value="hash_password")),
                ]
            ),
            limit=10,
            with_payload=True,
        )

        assert len(callers) > 0
        # The caller should be UserService (which calls hash_password in register)
        caller_names = [p.payload["entity_name"] for p in callers]
        assert any("UserService" in name or "register" in name for name in caller_names)


class TestGrpcIntegration:
    """Integration test suite explicitly verifying Qdrant gRPC transport operations."""

    GRPC_COLLECTION = "grpc_integration_test_collection"

    @pytest.fixture
    def grpc_client(self) -> QdrantClient:
        """Create a live Qdrant client explicitly using gRPC transport."""
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            prefer_grpc=True,
        )
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


# ═══════════════════════════════════════════════════════════════════════════
# Integration: Dual-Embedding (Named Vectors: 'text' + 'code')
# ═══════════════════════════════════════════════════════════════════════════


class TestDualEmbeddingIntegration:
    """Verify named vector dual embedding (MiniLM-L6 + UniXcoder) in Qdrant."""

    DUAL_COLLECTION = "test_dual_embedding_collection"

    @pytest.fixture(autouse=True)
    def _clean_collection(self, qdrant: QdrantClient):
        try:
            qdrant.delete_collection(self.DUAL_COLLECTION)
        except Exception:
            pass
        yield
        try:
            qdrant.delete_collection(self.DUAL_COLLECTION)
        except Exception:
            pass

    def test_dual_embedding_ensure_and_upsert(
        self,
        qdrant: QdrantClient,
        embed_model: SentenceTransformer,
        sample_repo: Path,
    ):
        """Verify that ensure_collection creates named vectors ('text' + 'code') and upserts both."""
        from transformers import AutoModel, AutoTokenizer
        from ingest import ensure_collection, upsert_chunks, walk_repo

        # 1. Ensure collection with dual_embedding=True
        ensure_collection(qdrant, collection_name=self.DUAL_COLLECTION, dual_embedding=True)
        col_info = qdrant.get_collection(self.DUAL_COLLECTION)
        assert isinstance(col_info.config.params.vectors, dict)
        assert "text" in col_info.config.params.vectors
        assert "code" in col_info.config.params.vectors
        assert col_info.config.params.vectors["text"].size == 384
        assert col_info.config.params.vectors["code"].size == 768

        # 2. Load code model and tokenizer
        code_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        code_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        code_model.eval()

        # 3. Upsert chunks
        chunks = list(walk_repo(sample_repo, "dual-repo", "payments"))
        count = upsert_chunks(
            qdrant,
            embed_model,
            chunks,
            collection_name=self.DUAL_COLLECTION,
            code_model=code_model,
            code_tokenizer=code_tokenizer,
        )
        assert count == len(chunks)

        # 4. Query using 'text' vector (384-d)
        text_query = embed_model.encode("user login authentication").tolist()
        text_results = qdrant.query_points(
            collection_name=self.DUAL_COLLECTION,
            query=text_query,
            using="text",
            limit=5,
            with_payload=True,
        ).points
        assert len(text_results) > 0

        # 5. Query using 'code' vector (768-d from UniXcoder)
        import torch
        inputs = code_tokenizer("def login(username, password):", return_tensors="pt")
        with torch.no_grad():
            outputs = code_model(**inputs)
            code_emb = torch.nn.functional.normalize(outputs.last_hidden_state.mean(dim=1), p=2, dim=1)
        code_vec = code_emb[0].tolist()

        code_results = qdrant.query_points(
            collection_name=self.DUAL_COLLECTION,
            query=code_vec,
            using="code",
            limit=5,
            with_payload=True,
        ).points
        assert len(code_results) > 0
        assert any("auth" in r.payload["file_path"].lower() for r in code_results)
