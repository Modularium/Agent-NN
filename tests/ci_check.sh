#!/usr/bin/env bash
set -e
ruff check .
mypy mcp || true
pytest "$@"
