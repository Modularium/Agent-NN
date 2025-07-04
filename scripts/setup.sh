#!/bin/bash
# -*- coding: utf-8 -*-
# Orchestrated setup for Agent-NN

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPERS_DIR="$SCRIPT_DIR/helpers"
source "$HELPERS_DIR/env.sh"
source "$HELPERS_DIR/frontend.sh"
source "$HELPERS_DIR/docker.sh"

usage() {
  echo "Usage: $(basename "$0")" >&2
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$SCRIPT_DIR/.."

log_info "Prüfe Umgebung"
check_env

log_info "Installiere Python-Abhängigkeiten"
poetry install && log_ok "Poetry-Install abgeschlossen"

build_frontend

start_docker

if poetry run agentnn --help >/dev/null 2>&1; then
  log_ok "CLI verfügbar unter: poetry run agentnn"
else
  log_err "CLI-Test fehlgeschlagen"
  exit 1
fi

log_ok "Setup abgeschlossen"
