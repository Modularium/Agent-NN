#!/usr/bin/env sh
# Start all Docker services with basic health and environment checks

set -e

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

docker compose up -d

echo "Services started"

