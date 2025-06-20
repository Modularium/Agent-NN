#!/bin/bash
# Start MCP services for local development
set -e
cd "$(dirname "$0")/.."
docker compose -f mcp/docker-compose.yml up --build
