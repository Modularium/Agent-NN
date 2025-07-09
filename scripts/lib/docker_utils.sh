#!/bin/bash

__docker_utils_init() {
    local dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    DOCKER_UTILS_DIR="$dir"
    source "$dir/log_utils.sh"
    source "$dir/../helpers/common.sh"
    source "$dir/status_utils.sh"
}

__docker_utils_init

has_docker() {
    if command -v docker &>/dev/null; then
        log_ok "Docker gefunden"
        return 0
    else
        log_err "Docker fehlt"
        return 1
    fi
}

has_docker_compose() {
    if docker compose version &>/dev/null; then
        log_ok "Docker Compose Plugin gefunden"
        return 0
    elif command -v docker-compose &>/dev/null; then
        log_ok "docker-compose gefunden"
        return 0
    fi
    log_err "Docker Compose fehlt"
    return 1
}

docker_compose() {
    if docker compose version &>/dev/null; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

load_compose_file() {
    local file="${1:-docker-compose.yml}"
    if [[ -f "$file" ]]; then
        echo "$file"
        return 0
    elif [[ -f "docker/$file" ]]; then
        echo "docker/$file"
        return 0
    fi
    log_err "Compose-Datei $file fehlt"
    return 1
}

start_docker_services() {
    local compose
    compose=$(load_compose_file "${1:-docker-compose.yml}") || return 1
    log_info "Starte Docker-Services..."
    if docker_compose -f "$compose" up -d; then
        log_ok "Docker-Services gestartet"
        update_status "docker" "ok" "$DOCKER_UTILS_DIR/../../.agentnn/status.json"
    else
        log_err "Docker-Services konnten nicht gestartet werden"
        return 1
    fi
}

export -f has_docker has_docker_compose docker_compose load_compose_file start_docker_services
