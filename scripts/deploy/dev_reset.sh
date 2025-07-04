#!/usr/bin/env sh
# Reset development environment: remove volumes, sessions and mocks

set -eu

usage() {
    echo "Usage: $(basename "$0")" >&2
}

if [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$REPO_ROOT"
echo "Cleaning development environment"

VOLS="postgres_data chroma_data"
for v in $VOLS; do
    docker volume rm -f "${PWD##*/}_${v}" 2>/dev/null || true
done

rm -rf data/sessions/* data/vectorstore/* frontend/dist

echo "Development environment cleaned"

