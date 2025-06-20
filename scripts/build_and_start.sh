#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
docker compose -f mcp/docker-compose.yml build
docker compose -f mcp/docker-compose.yml up
