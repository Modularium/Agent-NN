#!/usr/bin/env sh
# Start Docker services with basic checks
set -e

log_info() { echo "\033[1;34m[...]\033[0m $1"; }
log_ok() { echo "\033[1;32m[✓]\033[0m $1"; }
log_warn() { echo "\033[1;33m[⚠]\033[0m $1"; }
log_err() { echo "\033[1;31m[✗]\033[0m $1" >&2; }

# verify docker
if ! command -v docker >/dev/null 2>&1; then
  log_err "Docker nicht installiert"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  log_err "Docker-Daemon nicht erreichbar"
  exit 1
fi

# ensure .env exists
if [ ! -f ".env" ]; then
  log_warn ".env-Datei nicht gefunden. Erstelle aus .env.example..."
  cp .env.example .env
fi

log_info "Starte Docker-Services"
if command -v docker compose >/dev/null 2>&1; then
  docker compose up --build -d
elif command -v docker-compose >/dev/null 2>&1; then
  docker-compose up --build -d
else
  log_err "Docker Compose nicht gefunden. Bitte docker-compose installieren."
  exit 1
fi

log_ok "Docker gestartet"
