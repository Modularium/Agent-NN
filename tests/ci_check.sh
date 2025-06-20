#!/usr/bin/env bash
set -e
ruff .
mypy mcp || true
pytest "$@"
