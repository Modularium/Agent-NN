#!/bin/bash
# -*- coding: utf-8 -*-

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

check_env() {
  if [ ! -f .env ]; then
    log_warn ".env-Datei nicht gefunden. Erstelle aus .env.example..."
    cp .env.example .env || { log_err ".env konnte nicht erstellt werden."; exit 1; }
  fi

  command -v poetry >/dev/null 2>&1 || { log_err "Poetry nicht installiert"; exit 1; }
  command -v node >/dev/null 2>&1 || { log_err "Node.js nicht installiert"; exit 1; }
  command -v npm >/dev/null 2>&1 || { log_err "npm nicht installiert"; exit 1; }
  command -v docker >/dev/null 2>&1 || { log_err "Docker nicht installiert"; exit 1; }
  docker info >/dev/null 2>&1 || { log_err "Docker-Daemon nicht erreichbar"; exit 1; }

  log_ok "Umgebungsprüfungen abgeschlossen"
}
