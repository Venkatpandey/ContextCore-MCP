FROM python:3.11-slim

WORKDIR /app

# Copy uv binary from the official uv image
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

# Install system dependencies for tree-sitter C extensions and curl for health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install project dependencies into the virtualenv using uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy project source code
COPY . .

# Complete project installation
RUN uv sync --frozen --no-dev

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "server.py"]
