#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements-openhands.txt"
if ! command -v pip >/dev/null; then
  echo "pip not found" >&2
  exit 1
fi
pip install -r "$REQ_FILE"
