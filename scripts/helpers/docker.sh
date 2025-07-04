#!/bin/bash
# -*- coding: utf-8 -*-

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

start_docker() {
  log_info "Starte Docker-Services"
  if command -v docker compose &>/dev/null; then
    DOCKER_COMPOSE_COMMAND="docker compose"
  elif command -v docker-compose &>/dev/null; then
    DOCKER_COMPOSE_COMMAND="docker-compose"
  else
    log_err "Docker Compose nicht installiert!"
    exit 1
  fi

  $DOCKER_COMPOSE_COMMAND up --build -d
  log_ok "Docker gestartet"
}
