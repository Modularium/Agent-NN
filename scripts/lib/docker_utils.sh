#!/bin/bash

has_docker() { command -v docker &>/dev/null; }

has_compose_file() {
    [[ -f "docker-compose.yml" || -f "docker/docker-compose.yml" ]]
}

start_compose() {
    local file="${1:-docker-compose.yml}"
    docker compose -f "$file" up -d || return 1
}
