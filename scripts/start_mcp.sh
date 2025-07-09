#!/bin/bash
# Start MCP services for local development
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
cd "$SCRIPT_DIR/.."
docker_compose -f mcp/docker-compose.yml up --build
