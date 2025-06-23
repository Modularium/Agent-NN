#!/usr/bin/env sh
# Build the React frontend using Vite
# Output will be placed under frontend/dist

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
FRONTEND_DIR="$REPO_ROOT/frontend/agent-ui"
DIST_DIR="$REPO_ROOT/frontend/dist"
cd "$FRONTEND_DIR"

if [ ! -f package.json ]; then
    echo "Error: frontend package.json not found" >&2
    exit 1
fi

npm install
npm run build

mkdir -p "$DIST_DIR"
if [ -d "dist" ]; then
    cp -r dist/* "$DIST_DIR/"
elif [ -d "../dist" ]; then
    cp -r ../dist/* "$DIST_DIR/"
else
    echo "Error: build output not found" >&2
    exit 1
fi

echo "✅ Build abgeschlossen: $DIST_DIR"

