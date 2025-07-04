#!/usr/bin/env sh
# Build the React frontend using Vite
# Output will be placed under frontend/dist
# Copies are skipped if Vite already builds into the target directory.
# Pass --clean to remove the dist folder before building.

set -eu

usage() {
    echo "Usage: $(basename "$0") [--clean]" >&2
}

CLEAN=false
if [ "${1:-}" = "--help" ]; then
    usage
    exit 0
elif [ "${1:-}" = "--clean" ]; then
    CLEAN=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
FRONTEND_DIR="$REPO_ROOT/frontend/agent-ui"
DIST_DIR="$REPO_ROOT/frontend/dist"
echo "Building frontend in $FRONTEND_DIR"
cd "$FRONTEND_DIR"

if [ ! -f package.json ]; then
    echo "Error: frontend package.json not found" >&2
    exit 1
fi

if $CLEAN && [ -d "$DIST_DIR" ]; then
    echo "Cleaning $DIST_DIR"
    rm -rf "$DIST_DIR"
fi

npm install
npm run build

if [ -d "dist" ]; then
    OUT_DIR="dist"
elif [ -d "../dist" ]; then
    OUT_DIR="../dist"
else
    echo "Error: build output not found" >&2
    exit 1
fi

mkdir -p "$DIST_DIR"
SRC_DIR=$(realpath "$OUT_DIR")
DEST_DIR=$(realpath "$DIST_DIR")
if [ "$SRC_DIR" = "$DEST_DIR" ]; then
    echo "[skip] source and destination are identical"
    exit 0
else
    cp -r "$SRC_DIR"/* "$DEST_DIR/"
fi

echo "✅ Build abgeschlossen: $DIST_DIR"

