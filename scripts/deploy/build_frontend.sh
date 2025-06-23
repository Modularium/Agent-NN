#!/usr/bin/env sh
# Build the React frontend using Vite
# Output will be placed under frontend/dist

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$REPO_ROOT/frontend/agent-ui"

if [ ! -f package.json ]; then
    echo "Error: frontend package.json not found" >&2
    exit 1
fi

npm install
npm run build

mkdir -p "$REPO_ROOT/frontend/dist"
cp -r dist/* "$REPO_ROOT/frontend/dist/"

