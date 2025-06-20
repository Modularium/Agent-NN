#!/usr/bin/env bash
set -e

docker build -t agent-nn:latest -f Dockerfile .
ruff check .
mypy mcp || true
pytest || true
