#!/usr/bin/env sh
# Reset development environment: remove volumes, sessions and mocks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$REPO_ROOT"

VOLS="postgres_data chroma_data"
for v in $VOLS; do
    docker volume rm -f "${PWD##*/}_${v}" 2>/dev/null || true
done

rm -rf data/sessions/* data/vectorstore/* frontend/dist

echo "Development environment cleaned"

