#!/usr/bin/env bash
set -euo pipefail
ruff check .
mypy mcp
pytest -m "not heavy" -q
