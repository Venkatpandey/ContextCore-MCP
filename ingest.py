"""
ContextCore-MCP — Ingestion Pipeline
=====================================
Walks a target directory and ingests source code and business documentation
into a Qdrant vector database.

- Source code (.py, .ts, .js, .tsx, .jsx, .go, .rs, .java) is parsed with
  Tree-sitter and chunked by AST entity (function, class, interface).
- Markdown documents (.md) are split by H1/H2 headers with contextual
  chunk headers prepended.
- Each chunk is embedded with `all-MiniLM-L6-v2` (384-d) and upserted
  into the `propr_knowledge_base` collection in Qdrant.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

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
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "propr_knowledge_base")
EMBEDDING_DIM = 384
BATCH_SIZE = 100

# Map file extensions → tree-sitter language names
TS_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
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

    def to_payload(self) -> dict:
        return {
            "repo_name": self.repo_name,
            "domain_context": self.domain_context,
            "document_type": self.document_type,
            "file_path": self.file_path,
            "entity_name": self.entity_name,
            "content": self.content,
        }

    def deterministic_id(self) -> str:
        """Create a stable UUID from content + metadata so re-runs are idempotent."""
        fingerprint = f"{self.repo_name}:{self.file_path}:{self.entity_name}:{self.content}"
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
# AST-based code chunking
# ---------------------------------------------------------------------------

def _extract_entity_name(node, source_bytes: bytes, lang: str) -> str:
    """Best-effort extraction of the entity name from a tree-sitter node."""
    # Most languages have a `name` child of type `identifier`
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return source_bytes[child.start_byte: child.end_byte].decode(
                "utf-8", errors="replace"
            )
    # For decorated definitions (Python), recurse into the inner definition
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ENTITY_NODE_TYPES.get(lang, set()):
                return _extract_entity_name(child, source_bytes, lang)
    # For export_statement, look at the declaration child
    if node.type == "export_statement":
        for child in node.children:
            if "declaration" in child.type:
                return _extract_entity_name(child, source_bytes, lang)
    return "<anonymous>"


def _chunk_code_with_ast(
    source_code: str,
    lang_name: str,
    rel_path: Path,
    repo_name: str,
    domain_context: str,
) -> list[Chunk]:
    """Parse *source_code* with tree-sitter and return one Chunk per top-level entity."""
    language = get_language(lang_name)
    parser = get_parser(lang_name)

    source_bytes = source_code.encode("utf-8")
    tree = parser.parse(source_bytes)

    entity_types = ENTITY_NODE_TYPES.get(lang_name, set())
    doc_type = _classify_code_file(rel_path)
    chunks: list[Chunk] = []

    def _walk(node):  # type: ignore[no-untyped-def]
        if node.type in entity_types:
            text = source_bytes[node.start_byte: node.end_byte].decode(
                "utf-8", errors="replace"
            )
            name = _extract_entity_name(node, source_bytes, lang_name)
            chunks.append(
                Chunk(
                    repo_name=repo_name,
                    domain_context=domain_context,
                    document_type=doc_type,
                    file_path=str(rel_path),
                    entity_name=name,
                    content=text,
                )
            )
            # Don't recurse into nested definitions — we already captured
            # the outer entity. Nested helpers are included in the parent chunk.
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
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel_path = fpath.relative_to(root)
            suffix = fpath.suffix.lower()

            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError) as exc:
                logger.warning("Skipping %s: %s", fpath, exc)
                continue

            if not text.strip():
                continue

            # 1️⃣  Source code → AST chunking
            if suffix in TS_LANGUAGE_MAP:
                lang = TS_LANGUAGE_MAP[suffix]
                yield from _chunk_code_with_ast(
                    text, lang, rel_path, repo_name, domain_context
                )

            # 2️⃣  Markdown → header-based chunking
            elif suffix == ".md":
                yield from _chunk_markdown(
                    text, rel_path, repo_name, domain_context
                )

            # 3️⃣  Infrastructure files → whole-file chunking
            elif _is_infra_file(rel_path):
                yield from _chunk_infra_file(
                    text, rel_path, repo_name, domain_context
                )


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """Create the collection if it doesn't already exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        logger.info("Collection '%s' already exists — skipping creation.", collection_name)
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s'.", collection_name)


def upsert_chunks(
    client: QdrantClient,
    model: SentenceTransformer,
    chunks: list[Chunk],
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Embed and upsert a list of chunks into Qdrant in batches. Returns count."""
    if not chunks:
        return 0

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c.content for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id=chunk.deterministic_id(),
                vector=emb,
                payload=chunk.to_payload(),
            )
            for chunk, emb in zip(batch, embeddings)
        ]
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
        help="Qdrant server port.",
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

    logger.info("Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.qdrant_url:
        logger.info("Connecting to Qdrant at %s …", args.qdrant_url)
        client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    else:
        logger.info("Connecting to Qdrant at %s:%d …", args.qdrant_host, args.qdrant_port)
        client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, api_key=args.qdrant_api_key)

    ensure_collection(client, collection_name=args.collection)

    logger.info("Walking %s (repo=%s, domain=%s) …", target, args.repo_name, args.domain_context)
    all_chunks: list[Chunk] = list(walk_repo(target, args.repo_name, args.domain_context))
    logger.info("Extracted %d chunks.", len(all_chunks))

    if not all_chunks:
        logger.warning("No chunks found — nothing to ingest.")
        sys.exit(0)

    count = upsert_chunks(client, model, all_chunks, collection_name=args.collection)
    logger.info("✅  Ingestion complete — %d vectors upserted into '%s'.", count, args.collection)


if __name__ == "__main__":
    main()
