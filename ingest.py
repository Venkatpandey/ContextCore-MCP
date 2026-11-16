"""
ContextCore-MCP — Ingestion Pipeline
=====================================
Walks a target directory and ingests source code and business documentation
into a Qdrant vector database.

- Source code (.py, .ts, .js, .tsx, .jsx, .go, .rs, .java) is parsed with
  Tree-sitter and chunked by AST entity (function, class, interface).
- Call-graph extraction: for each function/method chunk, top-level call
  expressions and method invocations are extracted as a ``calls`` list.
- Markdown documents (.md) are split by H1/H2 headers with contextual
  chunk headers prepended.
- Each chunk is embedded with `all-MiniLM-L6-v2` (384-d) and upserted
  into the `contextcore_knowledge_base` collection in Qdrant.
- Revision provenance (commit_sha, git_ref, environment, build_id,
  artifact_digest) is attached to every chunk payload for deployment-aware
  retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from telemetry import (
    INGEST_BATCH_DURATION_SECONDS,
    INGEST_CHUNKS_TOTAL,
)

# ---------------------------------------------------------------------------
# Tree-sitter imports
# ---------------------------------------------------------------------------
from tree_sitter_languages import get_language, get_parser  # type: ignore[import-untyped]

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
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "contextcore_knowledge_base")
EMBEDDING_DIM = 384
CODE_EMBEDDING_DIM = 768
BATCH_SIZE = 100

# Dual embedding flag (all-MiniLM-L6-v2 + microsoft/unixcoder-base).
# Defaults to False for lightweight execution, can be enabled via DUAL_EMBEDDING=true or CLI --dual-embedding
DUAL_EMBEDDING_ENABLED = os.getenv("DUAL_EMBEDDING", "false").lower() in ("true", "1", "yes")

# Map file extensions → tree-sitter language names
TS_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".sh": "bash",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".scala": "scala",
    ".sc": "scala",
    ".kt": "kotlin",
    ".kts": "kotlin",
}

# Tree-sitter node types considered "top-level entities" per language.
# Expand as needed.
ENTITY_NODE_TYPES: dict[str, set[str]] = {
    "bash": {"function_definition"},
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
        "lexical_declaration",  # const Foo = () => …
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    },
    "java": {
        "class_declaration",
        "interface_declaration",
        "method_declaration",
        "enum_declaration",
    },
    "scala": {
        "class_definition",
        "trait_definition",
        "object_definition",
        "function_definition",
        "type_definition",
        "enum_definition",
    },
    "kotlin": {
        "class_declaration",
        "object_declaration",
        "function_declaration",
        "property_declaration",
    },
}

# Container types that house methods/functions and can be broken down into
# granular method chunks + high-level interface outline chunks.
CONTAINER_NODE_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "javascript": {"class_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "rust": {"impl_item", "trait_item"},
    "scala": {"class_definition", "trait_definition", "object_definition"},
    "kotlin": {"class_declaration", "object_declaration"},
}

# Method / member function node types inside containers.
METHOD_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "typescript": {"method_definition"},
    "tsx": {"method_definition"},
    "javascript": {"method_definition"},
    "java": {"method_declaration", "constructor_declaration"},
    "rust": {"function_item"},
    "scala": {"function_definition"},
    "kotlin": {"function_declaration"},
    "go": {"method_declaration"},
}

# Infrastructure file patterns
INFRA_PATTERNS: set[str] = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitlab-ci.yml",
    "Jenkinsfile",
    "Makefile",
    "Taskfile.yml",
}
INFRA_EXTENSIONS: set[str] = {".tf", ".hcl", ".yaml", ".yml", ".toml"}
INFRA_DIR_MARKERS: set[str] = {
    ".github",
    ".circleci",
    "terraform",
    "infra",
    "deploy",
    "k8s",
    "helm",
    "charts",
}

# Test file heuristics
TEST_PATH_MARKERS: set[str] = {
    "test",
    "tests",
    "spec",
    "specs",
    "__tests__",
    "test_",
    "_test",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single embeddable chunk destined for Qdrant."""

    repo_name: str
    domain_context: str
    document_type: str  # business_logic | code | test_code | infrastructure
    file_path: str
    entity_name: str
    content: str
    # Revision provenance
    commit_sha: str = ""
    git_ref: str = ""
    environment: str = ""
    build_id: str = ""
    artifact_digest: str = ""
    # Call-graph lineage (populated for code chunks)
    calls: list[str] = field(default_factory=list)

    def to_payload(self) -> dict:
        payload = {
            "repo_name": self.repo_name,
            "domain_context": self.domain_context,
            "document_type": self.document_type,
            "file_path": self.file_path,
            "entity_name": self.entity_name,
            "content": self.content,
        }
        # Provenance metadata — always present (empty string if unset)
        payload["commit_sha"] = self.commit_sha
        payload["git_ref"] = self.git_ref
        payload["environment"] = self.environment
        payload["build_id"] = self.build_id
        payload["artifact_digest"] = self.artifact_digest
        # Call-graph lineage
        payload["calls"] = self.calls
        return payload

    def deterministic_id(self) -> str:
        """Create a stable UUID from content + metadata so re-runs are idempotent.

        The fingerprint includes revision provenance so that the same code
        ingested at different revisions occupies distinct Qdrant points when
        the content has changed, but identical content across revisions
        collapses to the same point."""
        fingerprint = (
            f"{self.repo_name}:{self.file_path}:{self.entity_name}"
            f":{self.commit_sha}:{self.git_ref}:{self.environment}"
            f":{self.content}"
        )
        return str(uuid.UUID(hashlib.md5(fingerprint.encode()).hexdigest()))


# ---------------------------------------------------------------------------
# Helpers — file classification
# ---------------------------------------------------------------------------

def _is_infra_file(path: Path) -> bool:
    """Return True if the file looks like infrastructure / CI-CD config."""
    if path.name in INFRA_PATTERNS:
        return True
    if path.suffix in INFRA_EXTENSIONS:
        # Only count .yaml/.yml/.toml if they live in an infra-ish directory
        for part in path.parts:
            if part.lower() in INFRA_DIR_MARKERS:
                return True
        # Also match CI config at root level like .github/workflows/*.yml
        return False
    return False


def _is_test_file(path: Path) -> bool:
    """Return True if the file looks like a test file."""
    name_lower = path.stem.lower()
    parts_lower = {p.lower() for p in path.parts}
    if parts_lower & TEST_PATH_MARKERS:
        return True
    if name_lower.startswith("test_") or name_lower.endswith("_test"):
        return True
    if name_lower.endswith(".spec") or name_lower.endswith(".test"):
        return True
    # Handle compound extensions: foo.test.ts, foo.spec.tsx
    suffixes = "".join(path.suffixes).lower()
    if ".test." in suffixes or ".spec." in suffixes:
        return True
    return False


def _classify_code_file(rel_path: Path) -> str:
    """Return document_type for a source code file."""
    if _is_infra_file(rel_path):
        return "infrastructure"
    if _is_test_file(rel_path):
        return "test_code"
    return "code"


# ---------------------------------------------------------------------------
# AST-based code chunking & Outline generation
# ---------------------------------------------------------------------------

def _extract_go_receiver_type(recv_node, source_bytes: bytes) -> str | None:
    """Extract receiver type name from a Go method receiver node."""
    for ch in recv_node.children:
        if ch.type == "parameter_declaration":
            for sub in ch.children:
                if sub.type == "pointer_type":
                    for t in sub.children:
                        if t.type == "type_identifier":
                            return source_bytes[t.start_byte: t.end_byte].decode("utf-8", errors="replace")
                elif sub.type == "type_identifier":
                    return source_bytes[sub.start_byte: sub.end_byte].decode("utf-8", errors="replace")
    return None


def _extract_entity_name(node, source_bytes: bytes, lang: str) -> str:
    """Best-effort extraction of the entity name from a tree-sitter node."""
    # Special handling for Go method_declaration: extract Receiver.MethodName
    if lang == "go" and node.type == "method_declaration":
        name_node = node.child_by_field_name("name")
        m_name = (
            source_bytes[name_node.start_byte: name_node.end_byte].decode("utf-8", errors="replace")
            if name_node
            else None
        )
        recv_node = node.child_by_field_name("receiver")
        recv_type = _extract_go_receiver_type(recv_node, source_bytes) if recv_node else None
        if m_name and recv_type:
            return f"{recv_type}.{m_name}"
        if m_name:
            return m_name

    # Rust impl_item has 'type' child rather than 'name'
    if lang == "rust" and node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        if type_node:
            return source_bytes[type_node.start_byte: type_node.end_byte].decode("utf-8", errors="replace")

    # For decorated definitions (Python), recurse into the inner definition
    if node.type == "decorated_definition":
        for child in node.children:
            if (
                child.type in ENTITY_NODE_TYPES.get(lang, set())
                or child.type in CONTAINER_NODE_TYPES.get(lang, set())
                or child.type in METHOD_NODE_TYPES.get(lang, set())
            ):
                return _extract_entity_name(child, source_bytes, lang)

    # For export_statement, look at the declaration child
    if node.type == "export_statement":
        for child in node.children:
            if (
                "declaration" in child.type
                or child.type in CONTAINER_NODE_TYPES.get(lang, set())
                or child.type in ENTITY_NODE_TYPES.get(lang, set())
            ):
                return _extract_entity_name(child, source_bytes, lang)

    # Use tree-sitter field 'name' if present (most accurate across grammars)
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return source_bytes[name_node.start_byte: name_node.end_byte].decode("utf-8", errors="replace")

    # Fallback: inspect direct children for identifier types
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier") or (
            lang == "bash" and child.type == "word"
        ):
            return source_bytes[child.start_byte: child.end_byte].decode(
                "utf-8", errors="replace"
            )

    return "<anonymous>"


def _build_python_class_outline(node, source_bytes: bytes) -> str:
    """Build a concise interface outline for a large Python class."""
    block_node = None
    for child in node.children:
        if child.type == "block":
            block_node = child
            break
    if block_node is None:
        return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")

    header = source_bytes[node.start_byte: block_node.start_byte].decode("utf-8", errors="replace").strip()
    lines: list[str] = [header]

    for stmt in block_node.children:
        if stmt.type == "expression_statement":
            text = source_bytes[stmt.start_byte: stmt.end_byte].decode("utf-8", errors="replace").strip()
            if text.startswith(('"""', "'''", '"', "'")):
                lines.append(f"    {text}")
        elif stmt.type in ("function_definition", "decorated_definition"):
            target = stmt
            if target.type == "decorated_definition":
                for ch in target.children:
                    if ch.type == "function_definition":
                        target = ch
                        break
            sig = ""
            for ch in target.children:
                if ch.type == "block":
                    sig = source_bytes[target.start_byte: ch.start_byte].decode("utf-8", errors="replace").strip()
                    break
            if sig:
                lines.append(f"    {sig} ...")
            else:
                lines.append(f"    {source_bytes[stmt.start_byte: stmt.end_byte].decode('utf-8', errors='replace').strip()}")
    return "\n".join(lines)


def _build_ts_class_outline(node, source_bytes: bytes) -> str:
    """Build a concise interface outline for a TypeScript / JavaScript class."""
    body_node = None
    for child in node.children:
        if child.type == "class_body":
            body_node = child
            break
    if body_node is None:
        return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")

    header = source_bytes[node.start_byte: body_node.start_byte].decode("utf-8", errors="replace").strip()
    lines: list[str] = [f"{header} {{"]

    for member in body_node.children:
        if member.type == "method_definition":
            sig = ""
            for ch in member.children:
                if ch.type == "statement_block":
                    sig = source_bytes[member.start_byte: ch.start_byte].decode("utf-8", errors="replace").strip()
                    break
            if sig:
                lines.append(f"    {sig};")
            else:
                lines.append(f"    {source_bytes[member.start_byte: member.end_byte].decode('utf-8', errors='replace').strip()}")
        elif member.type in ("public_field_definition", "field_definition", "property_definition"):
            text = source_bytes[member.start_byte: member.end_byte].decode("utf-8", errors="replace").strip()
            lines.append(f"    {text}")
    lines.append("}")
    return "\n".join(lines)


def _build_java_class_outline(node, source_bytes: bytes) -> str:
    """Build a concise interface outline for a Java class."""
    body_node = None
    for child in node.children:
        if child.type == "class_body":
            body_node = child
            break
    if body_node is None:
        return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")

    header = source_bytes[node.start_byte: body_node.start_byte].decode("utf-8", errors="replace").strip()
    lines: list[str] = [f"{header} {{"]

    for member in body_node.children:
        if member.type in ("method_declaration", "constructor_declaration"):
            sig = ""
            for ch in member.children:
                if ch.type in ("block", "constructor_body"):
                    sig = source_bytes[member.start_byte: ch.start_byte].decode("utf-8", errors="replace").strip()
                    break
            if sig:
                lines.append(f"    {sig};")
            else:
                lines.append(f"    {source_bytes[member.start_byte: member.end_byte].decode('utf-8', errors='replace').strip()}")
        elif member.type == "field_declaration":
            text = source_bytes[member.start_byte: member.end_byte].decode("utf-8", errors="replace").strip()
            lines.append(f"    {text}")
    lines.append("}")
    return "\n".join(lines)


def _build_rust_impl_outline(node, source_bytes: bytes) -> str:
    """Build a concise interface outline for a Rust impl or trait block."""
    body_node = None
    for child in node.children:
        if child.type == "declaration_list":
            body_node = child
            break
    if body_node is None:
        return source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")

    header = source_bytes[node.start_byte: body_node.start_byte].decode("utf-8", errors="replace").strip()
    lines: list[str] = [f"{header} {{"]

    for member in body_node.children:
        if member.type == "function_item":
            sig = ""
            for ch in member.children:
                if ch.type == "block":
                    sig = source_bytes[member.start_byte: ch.start_byte].decode("utf-8", errors="replace").strip()
                    break
            if sig:
                lines.append(f"    {sig};")
            else:
                lines.append(f"    {source_bytes[member.start_byte: member.end_byte].decode('utf-8', errors='replace').strip()}")
    lines.append("}")
    return "\n".join(lines)


def _build_container_outline(node, source_bytes: bytes, lang: str) -> str:
    """Generate a clean outline summary for large classes/containers."""
    if lang == "python":
        return _build_python_class_outline(node, source_bytes)
    if lang in ("typescript", "tsx", "javascript"):
        return _build_ts_class_outline(node, source_bytes)
    if lang == "java":
        return _build_java_class_outline(node, source_bytes)
    if lang == "rust":
        return _build_rust_impl_outline(node, source_bytes)

    full = source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
    if len(full) > 1500:
        return full[:1500] + "\n// ... [Outline truncated for embedding efficiency] ..."
    return full


# ---------------------------------------------------------------------------
# Call-graph extraction from AST nodes
# ---------------------------------------------------------------------------

# Node types that represent call expressions across supported languages.
_CALL_NODE_TYPES: set[str] = {
    "call",                    # Python
    "call_expression",         # JS / TS / Go / Rust / Java / Kotlin
    "method_invocation",       # Java
    "navigation_expression",   # Kotlin (chained calls)
}


def _extract_calls(node, source_bytes: bytes) -> list[str]:
    """Walk *node*'s subtree and return deduplicated callee names.

    Captures both qualified paths (e.g. ``self.auth.login``, ``db.find``)
    and leaf names (e.g. ``login``, ``find``) so that call-path searches by
    either short or full name resolve accurately.
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add_call(name: str | None) -> None:
        if not name:
            return
        if name not in seen:
            seen.add(name)
            result.append(name)
        if "." in name:
            leaf = name.split(".")[-1]
            if leaf and leaf not in seen:
                seen.add(leaf)
                result.append(leaf)

    def _callee_name(call_node) -> str | None:
        """Return the best-effort callee name from a call node."""
        fn = call_node.children[0] if call_node.children else None
        if fn is None:
            return None

        # Simple identifier: foo()
        if fn.type in ("identifier", "property_identifier"):
            return source_bytes[fn.start_byte: fn.end_byte].decode(
                "utf-8", errors="replace"
            )
        # Attribute / member access: self.bar(), obj.method()
        if fn.type in (
            "attribute",            # Python
            "member_expression",    # JS / TS
            "selector_expression",  # Go
            "field_expression",     # Rust
        ):
            return source_bytes[fn.start_byte: fn.end_byte].decode(
                "utf-8", errors="replace"
            )
        # Java method_invocation — the node itself contains the name
        if call_node.type == "method_invocation":
            for child in call_node.children:
                if child.type == "identifier":
                    return source_bytes[child.start_byte: child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
        return None

    def _walk(n) -> None:
        if n.type in _CALL_NODE_TYPES:
            name = _callee_name(n)
            _add_call(name)
        for child in n.children:
            _walk(child)

    _walk(node)
    return result


def _chunk_code_with_ast(
    source_code: str,
    lang_name: str,
    rel_path: Path,
    repo_name: str,
    domain_context: str,
) -> list[Chunk]:
    """Parse *source_code* with tree-sitter into granular method, class, and entity chunks."""
    language = get_language(lang_name)
    parser = get_parser(lang_name)

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    entity_types = ENTITY_NODE_TYPES.get(lang_name, set())
    container_types = CONTAINER_NODE_TYPES.get(lang_name, set())
    method_types = METHOD_NODE_TYPES.get(lang_name, set())
    doc_type = _classify_code_file(rel_path)
    chunks: list[Chunk] = []

    def _process_container(c_node, container_name: str) -> None:
        # 1. Walk body and extract granular method chunks
        body_node = None
        for child in c_node.children:
            if child.type in ("block", "class_body", "declaration_list", "template_body"):
                body_node = child
                break
        members = body_node.children if body_node else c_node.children
        for member in members:
            if member.type in method_types:
                m_name = _extract_entity_name(member, source_bytes, lang_name)
                q_name = f"{container_name}.{m_name}" if m_name != "<anonymous>" else container_name
                m_text = source_bytes[member.start_byte: member.end_byte].decode("utf-8", errors="replace")
                m_calls = _extract_calls(member, source_bytes)
                chunks.append(
                    Chunk(
                        repo_name=repo_name,
                        domain_context=domain_context,
                        document_type=doc_type,
                        file_path=str(rel_path),
                        entity_name=q_name,
                        content=m_text,
                        calls=m_calls,
                    )
                )

        # 2. Container chunk itself (concise interface outline if large, full text if small)
        full_text = source_bytes[c_node.start_byte: c_node.end_byte].decode("utf-8", errors="replace")
        if len(full_text) > 1500:
            c_content = _build_container_outline(c_node, source_bytes, lang_name)
        else:
            c_content = full_text
        c_calls = _extract_calls(c_node, source_bytes)
        chunks.append(
            Chunk(
                repo_name=repo_name,
                domain_context=domain_context,
                document_type=doc_type,
                file_path=str(rel_path),
                entity_name=container_name,
                content=c_content,
                calls=c_calls,
            )
        )

    def _process_top_level(node) -> None:
        name = _extract_entity_name(node, source_bytes, lang_name)
        text = source_bytes[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
        calls = _extract_calls(node, source_bytes)
        chunks.append(
            Chunk(
                repo_name=repo_name,
                domain_context=domain_context,
                document_type=doc_type,
                file_path=str(rel_path),
                entity_name=name,
                content=text,
                calls=calls,
            )
        )

    def _walk(node) -> None:
        # Export statement (TypeScript / JavaScript)
        if node.type == "export_statement":
            for child in node.children:
                if child.type in container_types:
                    c_name = _extract_entity_name(child, source_bytes, lang_name)
                    _process_container(child, c_name)
                    return
                elif child.type in entity_types:
                    _process_top_level(child)
                    return
            return

        # Decorated definition (Python decorated class or function)
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in container_types:
                    c_name = _extract_entity_name(child, source_bytes, lang_name)
                    _process_container(node, c_name)
                    return
                elif child.type in entity_types:
                    _process_top_level(node)
                    return

        # Containers (Classes, interfaces, impls)
        if node.type in container_types:
            c_name = _extract_entity_name(node, source_bytes, lang_name)
            _process_container(node, c_name)
            return

        # Standalone top-level entities
        if node.type in entity_types:
            _process_top_level(node)
            return

        for child in node.children:
            _walk(child)

    _walk(tree.root_node)

    # If the parser found no top-level entities (e.g. a script with only
    # top-level statements), fall back to the whole file as a single chunk.
    if not chunks and source_code.strip():
        chunks.append(
            Chunk(
                repo_name=repo_name,
                domain_context=domain_context,
                document_type=doc_type,
                file_path=str(rel_path),
                entity_name=rel_path.name,
                content=source_code,
            )
        )

    return chunks

    return chunks


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------
_HEADER_RE = re.compile(r"^(#{1,2})\s+(.*)", re.MULTILINE)


def _chunk_markdown(
    text: str,
    rel_path: Path,
    repo_name: str,
    domain_context: str,
) -> list[Chunk]:
    """Split markdown by H1/H2 boundaries and prepend the document title."""
    lines = text.split("\n")
    # Extract the document title (first H1, or filename)
    doc_title: str = rel_path.stem
    for line in lines:
        m = re.match(r"^#\s+(.*)", line)
        if m:
            doc_title = m.group(1).strip()
            break

    # Determine document type
    doc_type = "business_logic"
    if _is_infra_file(rel_path):
        doc_type = "infrastructure"

    # Split on H1/H2 boundaries
    sections: list[tuple[str, str]] = []  # (header, body)
    current_header = doc_title
    current_lines: list[str] = []

    for line in lines:
        m = re.match(r"^#{1,2}\s+(.*)", line)
        if m:
            # Flush previous section
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))
            current_header = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_header, body))

    # If no sections were produced, treat the whole file as one chunk
    if not sections and text.strip():
        sections = [(doc_title, text.strip())]

    chunks: list[Chunk] = []
    for header, body in sections:
        # Contextual Chunk Header — prepend document title so every chunk
        # carries context about which document it came from.
        contextual_content = f"[Document: {doc_title}] [{header}]\n\n{body}"
        chunks.append(
            Chunk(
                repo_name=repo_name,
                domain_context=domain_context,
                document_type=doc_type,
                file_path=str(rel_path),
                entity_name=header,
                content=contextual_content,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Infrastructure file chunking (YAML / Dockerfile / Terraform / etc.)
# ---------------------------------------------------------------------------

def _chunk_infra_file(
    text: str,
    rel_path: Path,
    repo_name: str,
    domain_context: str,
) -> list[Chunk]:
    """Treat the entire infra file as a single chunk."""
    return [
        Chunk(
            repo_name=repo_name,
            domain_context=domain_context,
            document_type="infrastructure",
            file_path=str(rel_path),
            entity_name=rel_path.name,
            content=text,
        )
    ]


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------
SKIP_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".tox",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "target",  # Rust / Java
    ".gradle",
    ".gradle-home",
    ".idea",
    ".kotlin",
}


def walk_repo(
    root: Path,
    repo_name: str,
    domain_context: str,
) -> Generator[Chunk, None, None]:
    """Yield Chunk objects for every parseable file under *root*."""
    for fpath in _repo_files(root):
        rel_path = fpath.relative_to(root)
        suffix = fpath.suffix.lower()
        if suffix not in TS_LANGUAGE_MAP and suffix != ".md" and not _is_infra_file(rel_path):
            continue

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError) as exc:
            logger.warning("Skipping %s: %s", fpath, exc)
            continue

        if not text.strip():
            continue

        if suffix in TS_LANGUAGE_MAP:
            lang = TS_LANGUAGE_MAP[suffix]
            yield from _chunk_code_with_ast(
                text, lang, rel_path, repo_name, domain_context
            )
        elif suffix == ".md":
            yield from _chunk_markdown(text, rel_path, repo_name, domain_context)
        elif _is_infra_file(rel_path):
            yield from _chunk_infra_file(text, rel_path, repo_name, domain_context)


def _repo_files(root: Path) -> Generator[Path, None, None]:
    """Use tracked files in Git repositories; walk ordinary directories."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--cached", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        tracked = None

    if (root / ".git").exists() and (tracked is None or tracked.returncode != 0):
        raise RuntimeError(f"Cannot list tracked files in Git repository: {root}")

    if tracked is not None and tracked.returncode == 0:
        for name in tracked.stdout.split(b"\0"):
            if name:
                path = root / os.fsdecode(name)
                if path.is_file() and not path.is_symlink():
                    yield path
        return

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_file() and not path.is_symlink():
                yield path


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    dual_embedding: bool = DUAL_EMBEDDING_ENABLED,
) -> None:
    """Create the collection if it doesn't already exist.

    When *dual_embedding* is True, the collection is configured with named vectors:
      - 'text': 384-d vector (all-MiniLM-L6-v2) for NL and markdown queries.
      - 'code': 768-d vector (microsoft/unixcoder-base) for AST code syntax matching.
    When False, a single unnamed 384-d vector is used.
    """
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info("Collection '%s' already exists — skipping creation.", collection_name)
        return

    if dual_embedding:
        vectors_config: Any = {
            "text": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            "code": VectorParams(size=CODE_EMBEDDING_DIM, distance=Distance.COSINE),
        }
    else:
        vectors_config = VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    logger.info("Created collection '%s' (dual_embedding=%s).", collection_name, dual_embedding)


def _encode_unixcoder_batch(
    code_model: Any,
    code_tokenizer: Any,
    texts: list[str],
    device: str = "cpu",
) -> list[list[float]]:
    """Encode a batch of code snippets using UniXcoder into normalized 768-d vectors."""
    import torch

    inputs = code_tokenizer(
        texts,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = code_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized.cpu().tolist()


def upsert_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[Chunk],
    collection_name: str = COLLECTION_NAME,
    code_model: Any = None,
    code_tokenizer: Any = None,
) -> int:
    """Embed and upsert a list of chunks into Qdrant in batches. Returns count.

    Supports both single-vector collections and dual-embedding named-vector
    collections ('text' and 'code').
    """
    if not chunks:
        return 0

    # Inspect collection to see if it uses named vectors
    col_info = client.get_collection(collection_name)
    is_named_vectors = isinstance(col_info.config.params.vectors, dict)

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c.content for c in batch]
        text_embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # If named vectors are enabled and code model is provided, compute code embeddings
        code_embeddings: list[list[float]] | None = None
        if is_named_vectors and code_model is not None and code_tokenizer is not None:
            code_embeddings = _encode_unixcoder_batch(code_model, code_tokenizer, texts)

        points = []
        for idx, (chunk, text_emb) in enumerate(zip(batch, text_embeddings)):
            if is_named_vectors:
                point_vector: dict[str, list[float]] = {"text": text_emb}
                if code_embeddings is not None:
                    point_vector["code"] = code_embeddings[idx]
                points.append(
                    PointStruct(
                        id=chunk.deterministic_id(),
                        vector=point_vector,
                        payload=chunk.to_payload(),
                    )
                )
            else:
                points.append(
                    PointStruct(
                        id=chunk.deterministic_id(),
                        vector=text_emb,
                        payload=chunk.to_payload(),
                    )
                )

        with INGEST_BATCH_DURATION_SECONDS.time():
            client.upsert(collection_name=collection_name, points=points)

        for chunk in batch:
            INGEST_CHUNKS_TOTAL.labels(
                repo_name=chunk.repo_name,
                document_type=chunk.document_type,
            ).inc()

        total += len(points)
        logger.info("Upserted batch %d–%d (%d points)", i, i + len(batch), len(points))

    return total


# ---------------------------------------------------------------------------
# Git provenance helpers
# ---------------------------------------------------------------------------

def _auto_detect_commit_sha(target_dir: Path) -> str:
    """Return the current HEAD commit SHA of *target_dir* if it is a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _resolve_commit_sha(target_dir: Path, supplied: str, git_ref: str) -> str:
    """Require a verifiable revision whenever ingestion advertises one."""
    detected = _auto_detect_commit_sha(target_dir)
    if supplied and not re.fullmatch(r"[a-f0-9]{40}", supplied):
        raise ValueError("--commit-sha must be a full lowercase Git SHA")
    if detected and supplied and supplied != detected:
        raise ValueError("--commit-sha differs from the target checkout HEAD")
    commit_sha = supplied or detected
    if git_ref and not commit_sha:
        raise ValueError("--git-ref requires a Git checkout or --commit-sha")
    if detected:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(target_dir), capture_output=True, text=True, timeout=10,
        )
        if status.returncode != 0 or status.stdout:
            raise ValueError("Git checkout has tracked changes; index a clean revision")
    return commit_sha


def _apply_provenance(
    chunks: list[Chunk],
    *,
    commit_sha: str = "",
    git_ref: str = "",
    environment: str = "",
    build_id: str = "",
    artifact_digest: str = "",
) -> None:
    """Stamp every chunk in *chunks* with the supplied provenance metadata."""
    for chunk in chunks:
        chunk.commit_sha = commit_sha
        chunk.git_ref = git_ref
        chunk.environment = environment
        chunk.build_id = build_id
        chunk.artifact_digest = artifact_digest


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest source code and docs into Qdrant for ContextCore-MCP.",
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="Root directory of the repo to ingest.",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Logical name for this repository (used in Qdrant payload).",
    )
    parser.add_argument(
        "--domain-context",
        type=str,
        default="general",
        help="Business domain label (e.g. 'payments', 'identity').",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: '{COLLECTION_NAME}').",
    )
    # Revision provenance flags
    parser.add_argument(
        "--commit-sha",
        type=str,
        default="",
        help="Git commit hash. Auto-detected from target_dir HEAD if omitted.",
    )
    parser.add_argument(
        "--git-ref",
        type=str,
        default="",
        help="Git branch or release tag (e.g. 'main', 'v1.2.0').",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="",
        help="Deployment tier (e.g. 'production', 'staging', 'canary', 'local').",
    )
    parser.add_argument(
        "--build-id",
        type=str,
        default="",
        help="CI build ID for traceability.",
    )
    parser.add_argument(
        "--artifact-digest",
        type=str,
        default="",
        help="Container image digest or artifact hash.",
    )
    parser.add_argument(
        "--dual-embedding",
        action="store_true",
        default=DUAL_EMBEDDING_ENABLED,
        help="Generate dual embeddings using text (MiniLM-L6) and code (UniXcoder) models. Default: True.",
    )
    parser.add_argument(
        "--no-dual-embedding",
        action="store_false",
        dest="dual_embedding",
        help="Disable dual embedding and use single text vector only.",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=os.getenv("QDRANT_URL"),
        help="Qdrant server full URL (e.g. https://xyz.cloud.qdrant.io:6333). Overrides host/port.",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant server host.",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant server HTTP port.",
    )
    parser.add_argument(
        "--qdrant-grpc-port",
        type=int,
        default=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        help="Qdrant server gRPC port.",
    )
    parser.add_argument(
        "--prefer-grpc",
        action="store_true",
        default=os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("true", "1", "yes"),
        help="Use gRPC transport for faster point upserts.",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant API key for authentication.",
    )

    args = parser.parse_args()
    target = Path(args.target_dir).resolve()

    if not target.is_dir():
        logger.error("Target directory does not exist: %s", target)
        sys.exit(1)

    # A named ref without a pinned commit would admit stale results across runs.
    commit_sha = _resolve_commit_sha(target, args.commit_sha, args.git_ref)
    if commit_sha:
        logger.info("Revision provenance: commit_sha=%s", commit_sha[:12])
    if args.git_ref:
        logger.info("Revision provenance: git_ref=%s", args.git_ref)
    if args.environment:
        logger.info("Revision provenance: environment=%s", args.environment)

    logger.info("Loading text embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    code_model: Any = None
    code_tokenizer: Any = None
    if args.dual_embedding:
        logger.info("Loading code embedding model (microsoft/unixcoder-base) …")
        from transformers import AutoModel, AutoTokenizer
        code_tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        code_model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        code_model.eval()

    if args.qdrant_url:
        logger.info("Connecting to Qdrant at %s (prefer_grpc=%s) …", args.qdrant_url, args.prefer_grpc)
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, prefer_grpc=args.prefer_grpc)
    else:
        logger.info(
            "Connecting to Qdrant at %s:%d (grpc_port=%d, prefer_grpc=%s) …",
            args.qdrant_host,
            args.qdrant_port,
            args.qdrant_grpc_port,
            args.prefer_grpc,
        )
        client = QdrantClient(
            host=args.qdrant_host,
            port=args.qdrant_port,
            grpc_port=args.qdrant_grpc_port,
            prefer_grpc=args.prefer_grpc,
            api_key=args.qdrant_api_key,
        )

    ensure_collection(client, collection_name=args.collection, dual_embedding=args.dual_embedding)

    logger.info("Walking %s (repo=%s, domain=%s) …", target, args.repo_name, args.domain_context)
    all_chunks: list[Chunk] = list(walk_repo(target, args.repo_name, args.domain_context))
    logger.info("Extracted %d chunks.", len(all_chunks))

    if not all_chunks:
        logger.warning("No chunks found — nothing to ingest.")
        sys.exit(0)

    # Stamp provenance on every chunk
    _apply_provenance(
        all_chunks,
        commit_sha=commit_sha,
        git_ref=args.git_ref,
        environment=args.environment,
        build_id=args.build_id,
        artifact_digest=args.artifact_digest,
    )

    count = upsert_chunks(
        client,
        model,
        all_chunks,
        collection_name=args.collection,
        code_model=code_model,
        code_tokenizer=code_tokenizer,
    )
    logger.info("✅  Ingestion complete — %d vectors upserted into '%s'.", count, args.collection)


if __name__ == "__main__":
    main()
