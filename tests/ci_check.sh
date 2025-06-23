#!/usr/bin/env bash
set -e
ruff check .
black --check tests/api/test_health.py tests/integration/test_dispatcher_routing.py tests/cli/test_submit.py || true
mypy mcp || true
pytest --cov=. --cov-report=term --cov-report=html --cov-report=json "$@"
