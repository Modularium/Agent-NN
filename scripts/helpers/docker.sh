#!/bin/bash
# -*- coding: utf-8 -*-

set -euo pipefail

# Only set HELPERS_DIR if it's not already set
if [[ -z "${HELPERS_DIR:-}" ]]; then
    HELPERS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Repository-Wurzel ableiten, falls nicht gesetzt
if [[ -z "${REPO_ROOT:-}" ]]; then
    REPO_ROOT="$(cd "$HELPERS_DIR/../.." && pwd)"
fi

# Source common.sh from the helpers directory
source "$HELPERS_DIR/common.sh"

# Globale Docker-Variablen
DOCKER_COMPOSE_COMMAND=""
DOCKER_COMPOSE_VERSION=""

find_compose_file() {
    local name="${1:-docker-compose.yml}"

    local search_paths=(
        "$PWD/$name"
        "$REPO_ROOT/$name"
        "$REPO_ROOT/scripts/$name"
        "$REPO_ROOT/deploy/$name"
        "$REPO_ROOT/docker/$name"
    )

    for f in "${search_paths[@]}"; do
        if [[ -f "$f" ]]; then
            echo "$f"
            return 0
        fi
    done

    if [[ -f "$REPO_ROOT/${name}.example" ]]; then
        log_err "Docker Compose Datei nicht gefunden: $name"
        log_err "Erstelle sie aus ${name}.example"
    else
        log_err "Docker Compose Datei nicht gefunden: $name"
    fi
    return 1
}

detect_docker_compose() {
    log_info "Erkenne Docker Compose..."
    
    # Prüfe Docker Compose Plugin (neuere Variante)
    if docker compose version &>/dev/null; then
        DOCKER_COMPOSE_COMMAND="docker compose"
        DOCKER_COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
        alias dc="docker compose"
        log_ok "Docker Compose Plugin erkannt (v$DOCKER_COMPOSE_VERSION)"
        return 0
    fi
    
    # Prüfe Docker Compose Classic (ältere Variante)
    if command -v docker-compose &>/dev/null; then
        DOCKER_COMPOSE_COMMAND="docker-compose"
        DOCKER_COMPOSE_VERSION=$(docker-compose version --short 2>/dev/null || echo "unknown")
        alias dc="docker-compose"
        log_ok "Docker Compose Classic erkannt (v$DOCKER_COMPOSE_VERSION)"
        return 0
    fi
    
    log_err "Docker Compose nicht gefunden!"
    log_err "Bitte installiere Docker Compose:"
    log_err "  - Plugin: https://docs.docker.com/compose/install/"
    log_err "  - Classic: pip install docker-compose"
    return 1
}

check_docker() {
    log_info "Prüfe Docker-Installation..."
    
    # Docker Command prüfen
    if ! check_command docker "Docker"; then
        log_err "Docker ist nicht installiert. Siehe: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    # Docker Daemon prüfen
    if ! docker info &>/dev/null; then
        log_err "Docker-Daemon ist nicht erreichbar"
        log_err "Stelle sicher, dass Docker läuft und dein Benutzer in der docker-Gruppe ist"
        log_err "  sudo systemctl start docker"
        log_err "  sudo usermod -aG docker \$USER"
        return 1
    fi

    if [[ ! -S /var/run/docker.sock ]]; then
        log_warn "docker.sock nicht verfügbar oder keine Berechtigung"
    fi
    
    log_ok "Docker ist verfügbar ($(docker --version))"
    
    # Docker Compose erkennen
    detect_docker_compose
    
    return 0
}

docker_compose_up() {
    local compose_file="${1:-docker-compose.yml}"
    local build_flag="${2:-}"
    local detach_flag="${3:--d}"
    
    if [[ -z "$DOCKER_COMPOSE_COMMAND" ]]; then
        if ! detect_docker_compose; then
            return 1
        fi
    fi
    
    compose_file=$(find_compose_file "$compose_file") || return 1
    
    log_info "Starte Docker-Services mit $DOCKER_COMPOSE_COMMAND..."
    log_debug "Compose-Datei: $compose_file"
    log_debug "Build-Flag: ${build_flag:-none}"
    log_debug "Detach-Flag: $detach_flag"
    
    local cmd="$DOCKER_COMPOSE_COMMAND"
    if [[ "$compose_file" != "docker-compose.yml" ]]; then
        cmd="$cmd -f $compose_file"
    fi
    
    cmd="$cmd up"
    if [[ -n "$build_flag" ]]; then
        cmd="$cmd $build_flag"
    fi
    if [[ -n "$detach_flag" ]]; then
        cmd="$cmd $detach_flag"
    fi
    
    log_debug "Führe aus: $cmd"
    
    if eval "$cmd"; then
        log_ok "Docker-Services gestartet"
        docker ps --format "table {{.Names}}\t{{.Status}}" || true
        return 0
    else
        log_err "Fehler beim Starten der Docker-Services"
        return 1
    fi
}

docker_compose_down() {
    local compose_file="${1:-docker-compose.yml}"
    
    if [[ -z "$DOCKER_COMPOSE_COMMAND" ]]; then
        if ! detect_docker_compose; then
            return 1
        fi
    fi
    
    compose_file=$(find_compose_file "$compose_file") || return 1

    log_info "Stoppe Docker-Services..."
    
    local cmd="$DOCKER_COMPOSE_COMMAND"
    if [[ "$compose_file" != "docker-compose.yml" ]]; then
        cmd="$cmd -f $compose_file"
    fi
    cmd="$cmd down"
    
    if eval "$cmd"; then
        log_ok "Docker-Services gestoppt"
        return 0
    else
        log_warn "Fehler beim Stoppen der Docker-Services"
        return 1
    fi
}
