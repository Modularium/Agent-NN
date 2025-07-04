#!/bin/bash
# -*- coding: utf-8 -*-
# Agent-NN Setup Script - Vollständige Installation und Konfiguration

set -euo pipefail

# Skript-Verzeichnis und Helpers laden
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(dirname "$SCRIPT_DIR")"
readonly HELPERS_DIR="$SCRIPT_DIR/helpers"

# Alle Helper-Module laden
for helper in "$HELPERS_DIR"/{common,env,docker,frontend}.sh; do
    if [[ -f "$helper" ]]; then
        source "$helper"
    else
        echo "FEHLER: Helper-Datei nicht gefunden: $helper" >&2
        exit 1
    fi
done

# Globale Variablen
readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_FILE="$REPO_ROOT/logs/setup.log"
BUILD_FRONTEND=true
START_DOCKER=true
VERBOSE=false

usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Agent-NN Setup Script - Vollständige Installation und Konfiguration

OPTIONS:
    -h, --help              Diese Hilfe anzeigen
    -v, --verbose           Ausführliche Ausgabe aktivieren
    --no-frontend           Frontend-Build überspringen
    --no-docker             Docker-Start überspringen
    --check-only            Nur Umgebungsprüfung durchführen
    --clean                 Entwicklungsumgebung zurücksetzen

BEISPIELE:
    $SCRIPT_NAME                    # Vollständiges Setup
    $SCRIPT_NAME --check-only       # Nur Umgebungsprüfung
    $SCRIPT_NAME --no-docker        # Setup ohne Docker-Start
    $SCRIPT_NAME --verbose          # Mit ausführlicher Ausgabe

VORAUSSETZUNGEN:
    - Python 3.9+
    - Node.js 18+
    - Docker & Docker Compose
    - Poetry
    - Git

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                export DEBUG=1
                ;;
            --no-frontend)
                BUILD_FRONTEND=false
                ;;
            --no-docker)
                START_DOCKER=false
                ;;
            --check-only)
                BUILD_FRONTEND=false
                START_DOCKER=false
                ;;
            --clean)
                clean_environment
                exit 0
                ;;
            *)
                log_err "Unbekannte Option: $1"
                usage >&2
                exit 1
                ;;
        esac
        shift
    done
}

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 1> >(tee -a "$LOG_FILE")
    exec 2> >(tee -a "$LOG_FILE" >&2)
}

print_banner() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   █████╗  ██████╗ ███████╗███╗   ██╗████████╗      ███╗   ██╗███╗   ██╗     ║
║  ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝      ████╗  ██║████╗  ██║     ║
║  ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   █████╗██╔██╗ ██║██╔██╗ ██║     ║
║  ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════╝██║╚██╗██║██║╚██╗██║     ║
║  ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║         ██║ ╚████║██║ ╚████║     ║
║  ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝         ╚═╝  ╚═══╝╚═╝  ╚═══╝     ║
║                                                                              ║
║                    Multi-Agent System Setup Script                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log_info "Agent-NN Setup gestartet ($(date))"
    log_info "Repository: $REPO_ROOT"
    log_info "Log-Datei: $LOG_FILE"
    echo
}

install_python_dependencies() {
    log_info "Installiere Python-Abhängigkeiten mit Poetry..."
    
    cd "$REPO_ROOT" || {
        log_err "Kann nicht ins Repository-Verzeichnis wechseln"
        return 1
    }
    
    # Poetry-Konfiguration optimieren
    poetry config virtualenvs.in-project true 2>/dev/null || true
    
    # Dependencies installieren
    if poetry install; then
        log_ok "Python-Abhängigkeiten installiert"
    else
        log_err "Fehler bei der Installation der Python-Abhängigkeiten"
        log_err "Versuche: poetry install --no-dev"
        if poetry install --no-dev; then
            log_warn "Python-Abhängigkeiten ohne Dev-Dependencies installiert"
        else
            return 1
        fi
    fi
    
    # CLI-Test
    if poetry run agentnn --help &>/dev/null; then
        log_ok "CLI verfügbar: poetry run agentnn"
    else
        log_warn "CLI-Test fehlgeschlagen (möglicherweise normale Dev-Installation)"
    fi
    
    return 0
}

verify_installation() {
    log_info "Verifiziere Installation..."
    
    local verification_steps=(
        "check_env_file"
        "check_docker"
    )
    
    if [[ "$BUILD_FRONTEND" == "true" ]]; then
        verification_steps+=("check_frontend_build")
    fi
    
    local failed_verifications=()
    
    for step in "${verification_steps[@]}"; do
        case "$step" in
            check_frontend_build)
                if [[ ! -f "$REPO_ROOT/frontend/dist/index.html" ]]; then
                    failed_verifications+=("Frontend-Build")
                fi
                ;;
            check_env_file)
                if [[ ! -f "$REPO_ROOT/.env" ]]; then
                    failed_verifications+=("Umgebungskonfiguration")
                fi
                ;;
            check_docker)
                if [[ "$START_DOCKER" == "true" ]] && ! docker ps &>/dev/null; then
                    failed_verifications+=("Docker-Services")
                fi
                ;;
        esac
    done
    
    if [[ ${#failed_verifications[@]} -gt 0 ]]; then
        log_warn "Verifizierung teilweise fehlgeschlagen: ${failed_verifications[*]}"
        return 1
    fi
    
    log_ok "Installation erfolgreich verifiziert"
    return 0
}

print_next_steps() {
    echo
    log_ok "Setup erfolgreich abgeschlossen!"
    echo
    echo "📋 NÄCHSTE SCHRITTE:"
    echo
    echo "1. Konfiguration anpassen:"
    echo "   nano .env"
    echo
    echo "2. Services starten (falls nicht automatisch gestartet):"
    echo "   docker compose up -d"
    echo
    echo "3. Frontend aufrufen:"
    echo "   http://localhost:3000"
    echo
    echo "4. API testen:"
    echo "   curl http://localhost:8000/health"
    echo
    echo "5. CLI verwenden:"
    echo "   poetry run agentnn --help"
    echo
    echo "📖 WEITERE RESSOURCEN:"
    echo "   - Dokumentation: docs/"
    echo "   - Konfiguration: docs/config_reference.md"
    echo "   - Deployment: docs/deployment.md"
    echo "   - Troubleshooting: docs/troubleshooting.md"
    echo
    echo "🚀 Agent-NN ist bereit!"
    echo
}

clean_environment() {
    log_info "Bereinige Entwicklungsumgebung..."
    
    # Docker-Services stoppen
    if docker_compose_down; then
        log_ok "Docker-Services gestoppt"
    fi
    
    # Docker-Volumes entfernen
    local volumes=("postgres_data" "vector_data")
    for vol in "${volumes[@]}"; do
        local full_name="${PWD##*/}_${vol}"
        if docker volume ls -q | grep -q "^${full_name}$"; then
            docker volume rm "$full_name" 2>/dev/null || true
            log_debug "Volume entfernt: $full_name"
        fi
    done
    
    # Lokale Daten bereinigen
    local dirs_to_clean=(
        "data/sessions"
        "data/vectorstore"
        "logs"
        "frontend/dist"
        ".pytest_cache"
        "__pycache__"
    )
    
    for dir in "${dirs_to_clean[@]}"; do
        if [[ -d "$REPO_ROOT/$dir" ]]; then
            rm -rf "$REPO_ROOT/$dir"
            log_debug "Verzeichnis bereinigt: $dir"
        fi
    done
    
    # Frontend bereinigen
    if [[ -d "$REPO_ROOT/frontend/agent-ui" ]]; then
        clean_frontend
    fi
    
    log_ok "Entwicklungsumgebung bereinigt"
}

# Haupt-Setup-Funktion
main() {
    # Setup initialisieren
    setup_error_handling
    ensure_utf8
    setup_logging
    
    # Argumente parsen
    parse_arguments "$@"
    
    # Banner anzeigen
    print_banner
    
    # In Repository-Verzeichnis wechseln
    cd "$REPO_ROOT" || {
        log_err "Kann nicht ins Repository-Verzeichnis wechseln: $REPO_ROOT"
        exit 1
    }
    
    # Umgebungsprüfung
    log_info "=== UMGEBUNGSPRÜFUNG ==="
    if ! check_environment; then
        log_err "Umgebungsprüfung fehlgeschlagen. Setup abgebrochen."
        exit 1
    fi
    
    # Docker-Prüfung
    log_info "=== DOCKER-PRÜFUNG ==="
    if ! check_docker; then
        log_err "Docker-Prüfung fehlgeschlagen. Setup abgebrochen."
        exit 1
    fi
    
    # Python-Dependencies
    log_info "=== PYTHON-ABHÄNGIGKEITEN ==="
    if ! install_python_dependencies; then
        log_err "Installation der Python-Abhängigkeiten fehlgeschlagen. Setup abgebrochen."
        exit 1
    fi
    
    # Frontend-Build
    if [[ "$BUILD_FRONTEND" == "true" ]]; then
        log_info "=== FRONTEND-BUILD ==="
        if ! build_frontend; then
            log_err "Frontend-Build fehlgeschlagen. Setup abgebrochen."
            exit 1
        fi
        cd "$REPO_ROOT"
    fi
    
    # Docker-Services starten
    if [[ "$START_DOCKER" == "true" ]]; then
        log_info "=== DOCKER-SERVICES ==="
        compose_file=$(find_compose_file "docker-compose.yml") || {
            log_err "Docker Compose Datei nicht gefunden. Setup abgebrochen."
            exit 1
        }
        if ! docker_compose_up "$compose_file" "--build"; then
            log_err "Start der Docker-Services fehlgeschlagen. Setup abgebrochen."
            exit 1
        fi
    fi
    
    # Installation verifizieren
    log_info "=== VERIFIZIERUNG ==="
    verify_installation || log_warn "Verifizierung mit Problemen abgeschlossen"
    
    # Abschluss
    log_info "=== SETUP ABGESCHLOSSEN ==="
    print_next_steps
}

# Script ausführen falls direkt aufgerufen
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
