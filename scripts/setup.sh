#!/usr/bin/env sh
# Orchestrated setup for Agent-NN
set -eu

log_info() { echo "\033[1;34m[...]\033[0m $1"; }
log_ok() { echo "\033[1;32m[✓]\033[0m $1"; }
log_err() { echo "\033[1;31m[✗]\033[0m $1" >&2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

log_info "Pr\xC3\xBCfe Umgebung"
"$SCRIPT_DIR/check_env.sh"

log_info "Installiere Python-Abh\xC3\xA4ngigkeiten"
poetry install
log_ok "Poetry-Install abgeschlossen"

log_info "Baue Frontend"
"$SCRIPT_DIR/build_frontend.sh"
log_ok "Frontend gebaut"

log_info "Starte Docker"
"$SCRIPT_DIR/start_docker.sh"
log_ok "Docker gestartet"

if poetry run agentnn --help >/dev/null 2>&1; then
  log_ok "CLI verf\xC3\xBCgbar unter: poetry run agentnn"
else
  log_err "CLI-Test fehlgeschlagen"
  exit 1
fi

echo "\033[1;32m[✓]\033[0m Setup abgeschlossen"
