"""
Shared pytest fixtures for Docker integration tests.

Requires a running Qdrant instance on localhost:6333 (started via ``make up``).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
TEST_COLLECTION = "propr_knowledge_base"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def qdrant() -> QdrantClient:
    """Return a QdrantClient connected to the local Docker instance."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    # Block until Qdrant is actually reachable (max 20 s)
    for _ in range(10):
        try:
            client.get_collections()
            return client
        except Exception:
            time.sleep(2)
    pytest.fail("Qdrant not reachable on localhost:6333 — run `make up` first")


@pytest.fixture(scope="session")
def embed_model() -> SentenceTransformer:
    """Load the embedding model once per test session."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory) -> Path:
    """Create a tiny on-disk repo with Python, TS, Markdown, and infra files."""
    root = tmp_path_factory.mktemp("sample_repo")

    # ── Python source ──
    (root / "src").mkdir()
    (root / "src" / "auth.py").write_text(
        '''"""Authentication module."""

class AuthService:
    """Handles user authentication."""

    def login(self, username: str, password: str) -> str:
        """Authenticate a user and return a JWT token."""
        if not username or not password:
            raise ValueError("Credentials required")
        return f"token-for-{username}"

    def logout(self, token: str) -> bool:
        """Invalidate the given token."""
        return True


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    import hashlib
    return hashlib.sha256(plain.encode()).hexdigest()
'''
    )

    # ── TypeScript source ──
    (root / "src" / "payments.ts").write_text(
        '''export interface PaymentRequest {
  amount: number;
  currency: string;
  merchantId: string;
}

export class PaymentGateway {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async processPayment(req: PaymentRequest): Promise<string> {
    // Process the payment via Stripe
    return `txn_${req.merchantId}_${Date.now()}`;
  }
}

export function validateAmount(amount: number): boolean {
  return amount > 0 && amount < 1_000_000;
}
'''
    )

    # ── Test file ──
    (root / "tests").mkdir()
    (root / "tests" / "test_auth.py").write_text(
        '''"""Tests for the auth module."""
import pytest


def test_login_returns_token():
    from src.auth import AuthService
    svc = AuthService()
    assert svc.login("admin", "secret").startswith("token-")


def test_login_rejects_empty():
    from src.auth import AuthService
    svc = AuthService()
    with pytest.raises(ValueError):
        svc.login("", "")
'''
    )

    # ── Markdown / PRD ──
    (root / "docs").mkdir()
    (root / "docs" / "prd-payments.md").write_text(
        """# Payments PRD

## Overview
The payments module handles all monetary transactions for Propr tenants.

## Requirements
- Support Stripe and Adyen payment gateways
- PCI-DSS compliant tokenisation
- Idempotent payment processing

## Architecture Decision
We chose an event-sourced ledger so every state change is auditable.
"""
    )

    # ── Infrastructure ──
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text(
        """name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make test
"""
    )

    (root / "Dockerfile").write_text(
        """FROM python:3.11-slim
WORKDIR /app
COPY . .
CMD ["python", "main.py"]
"""
    )

    return root
