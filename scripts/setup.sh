#!/bin/bash
# -*- coding: utf-8 -*-
# Agent-NN Setup Script - Vollständige Installation und Konfiguration

set -euo pipefail

# Skript-Verzeichnis und Helpers laden
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/helpers/common.sh"
source "$SCRIPT_DIR/helpers/env.sh"
source "$SCRIPT_DIR/helpers/docker.sh"
source "$SCRIPT_DIR/helpers/frontend.sh"

source "$SCRIPT_DIR/lib/env_check.sh"
source "$SCRIPT_DIR/lib/docker_utils.sh"
source "$SCRIPT_DIR/lib/frontend_build.sh"
source "$SCRIPT_DIR/lib/install_utils.sh"

# Globale Variablen
SCRIPT_NAME="$(basename "$0")"
LOG_FILE="$REPO_ROOT/logs/setup.log"
BUILD_FRONTEND=true
START_DOCKER=true
VERBOSE=false
INSTALL_HEAVY=false
WITH_DOCKER=false
AUTO_MODE=false
RUN_MODE="full"

usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Agent-NN Setup Script - Vollständige Installation und Konfiguration

OPTIONS:
    -h, --help              Diese Hilfe anzeigen
    -v, --verbose           Ausführliche Ausgabe aktivieren
    --no-frontend           Frontend-Build überspringen
    --skip-docker           Docker-Start überspringen
    --check-only            Nur Umgebungsprüfung durchführen
    --install-heavy         Zusätzliche Heavy-Dependencies installieren
    --with-docker          Abbruch wenn docker-compose.yml fehlt
    --full                  Komplettes Setup ohne Nachfragen
    --minimal               Nur Python-Abhängigkeiten installieren
    --no-docker             Setup ohne Docker-Schritte
    --clean                 Entwicklungsumgebung zurücksetzen

BEISPIELE:
    $SCRIPT_NAME                    # Vollständiges Setup
    $SCRIPT_NAME --check-only       # Nur Umgebungsprüfung
    $SCRIPT_NAME --skip-docker      # Setup ohne Docker-Start
    $SCRIPT_NAME --verbose          # Mit ausführlicher Ausgabe
    $SCRIPT_NAME --install-heavy    # Heavy-Dependencies installieren

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
            --skip-docker)
                START_DOCKER=false
                ;;
            --check-only)
                BUILD_FRONTEND=false
                START_DOCKER=false
                ;;
            --install-heavy)
                INSTALL_HEAVY=true
                ;;
            --with-docker)
                WITH_DOCKER=true
                ;;
            --full)
                AUTO_MODE=true
                RUN_MODE="full"
                ;;
            --minimal)
                START_DOCKER=false
                BUILD_FRONTEND=false
                AUTO_MODE=true
                RUN_MODE="python"
                ;;
            --no-docker)
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

interactive_menu() {
    PS3="Auswahl: "
    options=(
        "Komplettes Setup (Empfohlen)"
        "Nur Python-Abhängigkeiten installieren"
        "Nur Frontend bauen"
        "Docker-Container starten"
        "Projekt testen"
        "Abbrechen"
    )
    select opt in "${options[@]}"; do
        case $REPLY in
            1) RUN_MODE="full"; break ;;
            2) RUN_MODE="python"; break ;;
            3) RUN_MODE="frontend"; break ;;
            4) RUN_MODE="docker"; break ;;
            5) RUN_MODE="test"; break ;;
            6) exit 0 ;;
            *) echo "Ungültige Auswahl";;
        esac
    done
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
    if [[ "$INSTALL_HEAVY" == "true" ]]; then
        pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
            || echo "⚠️ torch konnte nicht installiert werden – Tests evtl. deaktiviert"
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

run_project_tests() {
    log_info "Starte Tests..."
    if ruff check . && mypy mcp && pytest -m "not heavy" -q; then
        log_ok "Tests erfolgreich"
    else
        log_err "Tests fehlgeschlagen"
        return 1
    fi
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
    local original_args=("$@")
    local arg_count=$#
    # Setup initialisieren
    setup_error_handling
    ensure_utf8
    setup_logging
    
    # Argumente parsen
    parse_arguments "${original_args[@]}"
    if [[ $arg_count -eq 0 ]]; then
        interactive_menu
    fi
    export AUTO_MODE

    if [[ "$WITH_DOCKER" == "true" ]]; then
        if [[ ! -f docker-compose.yml ]]; then
            log_err "docker-compose.yml fehlt"
            exit 1
        fi
        if ! has_docker_compose; then
            log_err "Docker Compose nicht ausführbar"
            exit 1
        fi
    fi
    
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

    # Fehlende Komponenten installieren
    with_spinner "Prüfe Docker" ensure_docker || true
    with_spinner "Prüfe Node.js" ensure_node || true
    with_spinner "Prüfe Python" ensure_python || true
    with_spinner "Prüfe Poetry" ensure_poetry || true
    with_spinner "Prüfe Tools" ensure_python_tools || true
    
    # Docker-Prüfung
    log_info "=== DOCKER-PRÜFUNG ==="
    if ! has_docker; then
        if [[ "$WITH_DOCKER" == "true" ]]; then
            log_err "Docker erforderlich aber nicht gefunden."
            exit 1
        else
            log_warn "Docker nicht verfügbar – Docker-Start wird übersprungen"
            START_DOCKER=false
        fi
    elif ! has_docker_compose; then
        if [[ "$WITH_DOCKER" == "true" ]]; then
            log_err "Docker Compose nicht gefunden."
            exit 1
        else
            log_warn "Docker Compose fehlt – Docker-Start wird übersprungen"
            START_DOCKER=false
        fi
    fi
    
    case "$RUN_MODE" in
        python)
            install_python_dependencies || exit 1
            ;;
        frontend)
            build_frontend || exit 1
            ;;
        docker)
            start_docker_services "docker-compose.yml" || exit 1
            ;;
        test)
            run_project_tests || exit 1
            ;;
        full)
            log_info "=== PYTHON-ABHÄNGIGKEITEN ==="
            install_python_dependencies || exit 1

            if [[ "$BUILD_FRONTEND" == "true" ]]; then
                log_info "=== FRONTEND-BUILD ==="
                build_frontend || exit 1
                cd "$REPO_ROOT"
            fi

            if [[ "$START_DOCKER" == "true" ]]; then
                log_info "=== DOCKER-SERVICES ==="
                compose_file="docker-compose.yml"
                if [[ ! -f "$compose_file" ]]; then
                    compose_file=$(ls docker-compose.*.yml 2>/dev/null | head -n1 || true)
                fi
                if [[ -f "$compose_file" ]]; then
                    start_docker_services "$compose_file" || exit 1
                elif [[ "$WITH_DOCKER" == "true" ]]; then
                    log_err "Docker Compose Datei nicht gefunden. Setup abgebrochen."
                    exit 1
                fi
            fi

            log_info "=== VERIFIZIERUNG ==="
            verify_installation || log_warn "Verifizierung mit Problemen abgeschlossen"

            run_project_tests || true
            print_next_steps
            ;;
    esac
}

# Script ausführen falls direkt aufgerufen
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
