#!/bin/bash
# -*- coding: utf-8 -*-
# Verbessertes MCP Services Management Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/lib/log_utils.sh"
source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/helpers/docker.sh"

# MCP-spezifische Konfiguration
MCP_COMPOSE_FILE="$REPO_ROOT/mcp/docker-compose.yml"
MCP_ENV_FILE="$REPO_ROOT/mcp/.env"
MCP_CONFIG_DIR="$REPO_ROOT/mcp/config"

# Service-Definitionen
declare -A MCP_SERVICES=(
    [dispatcher]="Task Dispatcher:8001:/health"
    [registry]="Agent Registry:8002:/health"
    [session_manager]="Session Manager:8003:/health"
    [vector_store]="Vector Store:8004:/health"
    [llm_gateway]="LLM Gateway:8005:/health"
    [worker_dev]="Dev Worker:8006:/health"
    [worker_loh]="LoH Worker:8007:/health"
    [worker_openhands]="OpenHands Worker:8008:/health"
)

usage() {
    cat << EOF
Usage: $(basename "$0") [COMMAND] [OPTIONS]

MCP Services Management für Agent-NN

COMMANDS:
    start           Starte alle MCP Services (default)
    stop            Stoppe alle MCP Services
    restart         Neustart aller MCP Services
    status          Zeige Service-Status
    logs            Zeige Service-Logs
    health          Führe Health-Check durch
    setup           Erste MCP-Konfiguration
    clean           Bereinige MCP-Daten und Container

OPTIONS:
    --service NAME  Nur spezifischen Service verwalten
    --build         Services neu bauen vor Start
    --follow        Logs kontinuierlich anzeigen
    --env ENV       Umgebung (dev|prod|test)
    -h, --help      Diese Hilfe anzeigen

BEISPIELE:
    $(basename "$0") start --build           # Alle Services neu bauen und starten
    $(basename "$0") logs --service registry --follow  # Registry-Logs anzeigen
    $(basename "$0") health                  # Health-Check aller Services
    $(basename "$0") setup                   # Erstmalige MCP-Konfiguration

EOF
}

# Prüfe MCP-Umgebung
check_mcp_environment() {
    log_info "Prüfe MCP-Umgebung..."
    
    # Docker verfügbar?
    if ! check_docker; then
        log_err "Docker erforderlich für MCP Services"
        return 1
    fi
    
    # MCP-Verzeichnis vorhanden?
    if [[ ! -d "$REPO_ROOT/mcp" ]]; then
        log_err "MCP-Verzeichnis nicht gefunden: $REPO_ROOT/mcp"
        return 1
    fi
    
    # Compose-Datei vorhanden?
    if [[ ! -f "$MCP_COMPOSE_FILE" ]]; then
        log_err "MCP docker-compose.yml nicht gefunden: $MCP_COMPOSE_FILE"
        return 1
    fi
    
    log_ok "MCP-Umgebung bereit"
    return 0
}

# Setup MCP-Konfiguration
setup_mcp_config() {
    log_info "Konfiguriere MCP-Umgebung..."
    
    # Erstelle Config-Verzeichnis
    mkdir -p "$MCP_CONFIG_DIR"
    
    # .env für MCP erstellen falls nicht vorhanden
    if [[ ! -f "$MCP_ENV_FILE" ]]; then
        if [[ -f "$MCP_ENV_FILE.example" ]]; then
            cp "$MCP_ENV_FILE.example" "$MCP_ENV_FILE"
            log_ok "MCP .env aus Beispiel erstellt"
        else
            # Minimale .env erstellen
            cat > "$MCP_ENV_FILE" << 'EOF'
# MCP Services Konfiguration
POSTGRES_DB=mcp_db
POSTGRES_USER=mcp_user
POSTGRES_PASSWORD=mcp_password
REDIS_PASSWORD=mcp_redis_password

# Service-Konfiguration
MCP_LOG_LEVEL=INFO
MCP_DEBUG=false

# Netzwerk-Konfiguration
MCP_NETWORK=mcp_network
EOF
            log_ok "Minimale MCP .env erstellt"
        fi
        
        log_warn "Bitte überprüfe und passe die MCP-Konfiguration an: $MCP_ENV_FILE"
    fi
    
    # Konfigurationsdateien für Services
    local config_files=(
        "dispatcher.yml:Task Dispatcher Konfiguration"
        "registry.yml:Agent Registry Konfiguration"
        "session.yml:Session Manager Konfiguration"
        "vector.yml:Vector Store Konfiguration"
        "gateway.yml:LLM Gateway Konfiguration"
    )
    
    for config_info in "${config_files[@]}"; do
        local config_file="${config_info%%:*}"
        local config_desc="${config_info#*:}"
        
        if [[ ! -f "$MCP_CONFIG_DIR/$config_file" ]]; then
            # Beispiel-Konfiguration erstellen
            cat > "$MCP_CONFIG_DIR/$config_file" << EOF
# $config_desc
# Automatisch generiert von setup

api:
  host: "0.0.0.0"
  port: 8000
  debug: false

database:
  url: "postgresql://mcp_user:mcp_password@postgres:5432/mcp_db"

redis:
  url: "redis://:mcp_redis_password@redis:6379/0"

logging:
  level: "INFO"
  format: "json"
EOF
            log_debug "$config_desc erstellt: $config_file"
        fi
    done
    
    log_ok "MCP-Konfiguration vollständig"
}

# Starte MCP Services
start_mcp_services() {
    local build_flag=""
    local service_name=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build) build_flag="--build"; shift ;;
            --service) service_name="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    check_mcp_environment || return 1
    
    log_info "Starte MCP Services..."
    
    cd "$REPO_ROOT/mcp" || return 1
    
    local cmd="docker_compose up -d"
    if [[ -n "$build_flag" ]]; then
        cmd="docker_compose up --build -d"
    fi
    
    if [[ -n "$service_name" ]]; then
        cmd="$cmd $service_name"
        log_info "Starte Service: $service_name"
    fi
    
    if eval "$cmd"; then
        log_ok "MCP Services gestartet"
        
        # Kurze Wartezeit
        sleep 5
        
        # Status anzeigen
        show_mcp_status
        
        return 0
    else
        log_err "Fehler beim Starten der MCP Services"
        return 1
    fi
}

# Stoppe MCP Services
stop_mcp_services() {
    local service_name=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service) service_name="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    log_info "Stoppe MCP Services..."
    
    cd "$REPO_ROOT/mcp" || return 1
    
    local cmd="docker_compose down"
    if [[ -n "$service_name" ]]; then
        cmd="docker_compose stop $service_name"
        log_info "Stoppe Service: $service_name"
    fi
    
    if eval "$cmd"; then
        log_ok "MCP Services gestoppt"
    else
        log_warn "Fehler beim Stoppen der MCP Services"
        return 1
    fi
}

# Zeige MCP Service-Status
show_mcp_status() {
    log_info "MCP Service-Status:"
    
    cd "$REPO_ROOT/mcp" || return 1
    
    # Docker Compose Status
    if docker_compose ps 2>/dev/null; then
        echo
    else
        log_warn "Keine MCP Services laufen"
    fi
    
    # Detaillierter Status pro Service
    local running_count=0
    local total_count=${#MCP_SERVICES[@]}
    
    for service_key in "${!MCP_SERVICES[@]}"; do
        local service_info="${MCP_SERVICES[$service_key]}"
        local service_desc="${service_info%%:*}"
        local service_port="${service_info#*:}"
        service_port="${service_port%%:*}"
        
        if docker_compose ps "$service_key" | grep -q "Up"; then
            log_ok "$service_desc läuft (Port: $service_port)"
            running_count=$((running_count + 1))
        else
            log_warn "$service_desc läuft nicht"
        fi
    done
    
    echo
    log_info "Gesamt: $running_count/$total_count Services laufen"
}

# Health-Check für MCP Services
check_mcp_health() {
    log_info "Führe MCP Health-Check durch..."
    
    local healthy_count=0
    local total_count=${#MCP_SERVICES[@]}
    
    for service_key in "${!MCP_SERVICES[@]}"; do
        local service_info="${MCP_SERVICES[$service_key]}"
        local service_desc="${service_info%%:*}"
        local service_port="${service_info#*:}"
        service_port="${service_port%%:*}"
        local health_path="${service_info##*:}"
        
        local health_url="http://localhost:${service_port}${health_path}"
        
        log_debug "Prüfe $service_desc: $health_url"
        
        if curl -f -s --max-time 5 "$health_url" >/dev/null 2>&1; then
            log_ok "$service_desc ist gesund"
            healthy_count=$((healthy_count + 1))
        else
            log_warn "$service_desc antwortet nicht"
        fi
    done
    
    echo
    if [[ $healthy_count -eq $total_count ]]; then
        log_ok "Alle MCP Services sind gesund ($healthy_count/$total_count)"
        return 0
    else
        log_warn "$healthy_count/$total_count MCP Services sind gesund"
        return 1
    fi
}

# Zeige MCP Service-Logs
show_mcp_logs() {
    local service_name=""
    local follow=false
    local lines=50
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --service) service_name="$2"; shift 2 ;;
            --follow) follow=true; shift ;;
            --lines) lines="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    
    cd "$REPO_ROOT/mcp" || return 1
    
    local cmd="docker_compose logs --tail=$lines"
    
    if [[ "$follow" == "true" ]]; then
        cmd="$cmd -f"
    fi
    
    if [[ -n "$service_name" ]]; then
        cmd="$cmd $service_name"
        log_info "Zeige Logs für Service: $service_name"
    else
        log_info "Zeige Logs für alle MCP Services"
    fi
    
    eval "$cmd"
}

# Bereinige MCP-Daten
clean_mcp_data() {
    log_info "Bereinige MCP-Daten..."
    
    # Stoppe Services erst
    stop_mcp_services
    
    cd "$REPO_ROOT/mcp" || return 1
    
    # Entferne Container und Volumes
    if docker_compose down --volumes --rmi local 2>/dev/null; then
        log_ok "MCP Container und Volumes entfernt"
    fi
    
    # Entferne spezifische MCP-Volumes
    local volumes=(
        "mcp_postgres_data"
        "mcp_vector_data"
        "mcp_redis_data"
    )
    
    for vol in "${volumes[@]}"; do
        if docker volume ls -q | grep -q "^${vol}$"; then
            docker volume rm "$vol" 2>/dev/null || true
            log_debug "Volume entfernt: $vol"
        fi
    done
    
    log_ok "MCP-Daten bereinigt"
}

# Hauptfunktion
main() {
    local command="${1:-start}"
    shift || true
    
    case "$command" in
        start)
            start_mcp_services "$@"
            ;;
        stop)
            stop_mcp_services "$@"
            ;;
        restart)
            stop_mcp_services "$@"
            sleep 2
            start_mcp_services "$@"
            ;;
        status)
            show_mcp_status
            ;;
        logs)
            show_mcp_logs "$@"
            ;;
        health)
            check_mcp_health
            ;;
        setup)
            setup_mcp_config
            ;;
        clean)
            clean_mcp_data
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_err "Unbekannter Befehl: $command"
            usage >&2
            exit 1
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
