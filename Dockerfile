FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --all-groups --frozen

COPY . .

RUN uv run python dabstep_agent/download_data.py && \
    uv run python dabstep_agent/generate_file_structures.py

EXPOSE 8888 8000

ENV JUPYTER_TOKEN=""

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
