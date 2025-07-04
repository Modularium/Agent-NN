#!/usr/bin/env sh
# Start all Docker services with basic health and environment checks

set -eu

usage() {
    echo "Usage: $(basename "$0") [--build] [--dry-run]" >&2
}

BUILD=false
DRY=false
for arg in "$@"; do
    case $arg in
        --help)
            usage
            exit 0
            ;;
        --build)
            BUILD=true
            ;;
        --dry-run)
            DRY=true
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")/.."
cd "$REPO_ROOT"

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Copy .env.example and adjust values." >&2
    exit 1
fi

# Check required ports
check_port() {
    PORT=$1
    if lsof -i ":$PORT" >/dev/null 2>&1; then
        echo "Port $PORT already in use" >&2
        exit 1
    fi
}

check_port 8000
check_port 3000

if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_CMD="docker-compose"
else
    DOCKER_CMD="docker compose"
fi

CMD="$DOCKER_CMD up -d"
if $BUILD; then
    CMD="$DOCKER_CMD up --build -d"
fi
echo "Running: $CMD"
if ! $DRY; then
    if ! eval "$CMD"; then
        echo "\033[1;31m[✗]\033[0m Docker start failed" >&2
        exit 1
    fi
fi

echo "\033[1;32m[✓]\033[0m Services started"

