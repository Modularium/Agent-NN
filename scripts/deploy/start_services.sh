#!/bin/bash
# -*- coding: utf-8 -*-

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
HELPERS_DIR="$REPO_ROOT/scripts/helpers"
LIB_DIR="$REPO_ROOT/scripts/lib"

# Helper laden
source "$HELPERS_DIR/common.sh"
source "$LIB_DIR/docker_utils.sh"
source "$LIB_DIR/env_check.sh"

usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Startet alle Agent-NN Docker-Services mit Prüfungen

OPTIONS:
    -h, --help              Diese Hilfe anzeigen
    -f, --file FILE         Docker Compose Datei (default: docker-compose.yml)
    --dry-run               Befehle nur anzeigen, nicht ausführen
    --check-only            Nur Prüfungen durchführen

BEISPIELE:
    $(basename "$0")                    # Standard-Start
    $(basename "$0") -f docker-compose.production.yml  # Production
    $(basename "$0") --dry-run          # Testlauf

EOF
}

# Hauptfunktion
main() {
    local compose_file="docker-compose.yml"
    local dry_run=false
    local check_only=false
    
    # Parameter parsen
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -f|--file)
                compose_file="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --check-only)
                check_only=true
                shift
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
    done
    
    # Setup
    ensure_utf8
    cd "$REPO_ROOT" || {
        log_err "Kann nicht ins Repository-Verzeichnis wechseln"
        exit 1
    }
    
    log_info "Starte Agent-NN Services..."
    log_debug "Compose-Datei: $compose_file"
    log_debug "Dry-Run: $dry_run"
    
    # Grundlegende Prüfungen
    env_check || exit 1

    if ! has_docker; then
        echo "Docker nicht gefunden" >&2
        exit 1
    fi

    if [[ ! -f "$compose_file" ]]; then
        echo "Compose-Datei $compose_file nicht gefunden" >&2
        exit 1
    fi
    
    
    if [[ "$check_only" == "true" ]]; then
        log_ok "Alle Prüfungen erfolgreich"
        exit 0
    fi
    
    # Services starten
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY-RUN: Würde ausführen:"
        echo "start_compose '$compose_file'"
    else
        if start_compose "$compose_file"; then
            log_ok "Services erfolgreich gestartet"
            
            # Kurze Wartezeit für Service-Start
            sleep 5
            
            # Grundlegende Health-Checks
            log_info "Führe Health-Checks durch..."
            
            local health_urls=(
                "http://localhost:8000/health:API Gateway"
                "http://localhost:3000:Frontend"
            )
            
            for url_desc in "${health_urls[@]}"; do
                local url="${url_desc%:*}"
                local desc="${url_desc#*:}"
                
                if curl -f -s "$url" &>/dev/null; then
                    log_ok "$desc erreichbar ($url)"
                else
                    log_warn "$desc nicht erreichbar ($url)"
                fi
            done
            
            log_ok "Setup abgeschlossen!"
            echo
            echo "🌐 Frontend: http://localhost:3000"
            echo "🔧 API: http://localhost:8000"
            echo "📊 Monitoring: http://localhost:9090"
            echo
        else
            log_err "Fehler beim Starten der Services"
            exit 1
        fi
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
