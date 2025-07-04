#!/usr/bin/env sh
# One-click setup: install dependencies, build frontend and start services
set -eu

log_info() { echo "\033[1;34m[...]\033[0m $1"; }
log_ok() { echo "\033[1;32m[✓]\033[0m $1"; }
log_err() { echo "\033[1;31m[✗]\033[0m $1" >&2; }

usage() {
    echo "Usage: $(basename "$0") [--no-services]" >&2
}

NO_SERVICES=false
for arg in "$@"; do
    case $arg in
        --help)
            usage
            exit 0
            ;;
        --no-services)
            NO_SERVICES=true
            ;;
        *)
            echo "Unknown option: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

log_info "Installiere Python-Abh\xC3\xA4ngigkeiten"
if poetry install; then
    log_ok "Poetry-Install abgeschlossen"
else
    log_err "Poetry-Install fehlgeschlagen"
    exit 1
fi

if [ ! -f frontend/dist/index.html ]; then
    log_info "Baue Frontend"
    if scripts/deploy/build_frontend.sh; then
        log_ok "Frontend gebaut"
    else
        log_err "Frontend-Build fehlgeschlagen"
        exit 1
    fi
else
    log_info "Frontend bereits vorhanden"
fi

if ! $NO_SERVICES; then
    log_info "Starte Docker-Services"
    if scripts/deploy/start_services.sh --build; then
        log_ok "Docker-Services gestartet"
    else
        log_err "Docker konnte nicht gestartet werden."
        exit 1
    fi
fi

if ! poetry run agentnn --help >/dev/null 2>&1; then
    log_err "Agent-NN konnte nicht gestartet werden. Bitte pr\xC3\xBCfe die Python-Abh\xC3\xA4ngigkeiten."
    exit 1
else
    log_ok "Agent-NN CLI erfolgreich getestet"
fi

echo "\033[1;32m[✓]\033[0m Setup abgeschlossen"
