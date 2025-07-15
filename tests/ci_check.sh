#!/usr/bin/env bash
set -e
ruff --version >/dev/null
black --version >/dev/null
pytest --version >/dev/null
ruff check .
black --check tests/api/test_health.py tests/integration/test_dispatcher_routing.py tests/cli/test_submit.py || true
mypy mcp || true

if [[ "$1" == "--full" ]]; then
    shift
    pytest --cov=. --cov-report=term --cov-report=html --cov-report=json "$@"
else
    pytest -m "not heavy" --cov=. --cov-report=term --cov-report=html --cov-report=json "$@"
fi
