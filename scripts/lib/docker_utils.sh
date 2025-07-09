#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../helpers/common.sh"

has_docker() { command -v docker &>/dev/null; }

has_docker_compose() {
    docker compose version &>/dev/null || command -v docker-compose &>/dev/null
}

docker_compose() {
    if docker compose version &>/dev/null; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

has_compose_file() {
    [[ -f "docker-compose.yml" || -f "docker/docker-compose.yml" ]]
}

start_compose() {
    local file="${1:-docker-compose.yml}"
    docker_compose -f "$file" up -d || return 1
}

export -f has_docker has_docker_compose has_compose_file start_compose
