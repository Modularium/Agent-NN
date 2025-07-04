#!/usr/bin/env sh
# Basic environment checks for Agent-NN setup
set -e

log_info() { echo "\033[1;34m[...]\033[0m $1"; }
log_ok() { echo "\033[1;32m[✓]\033[0m $1"; }
log_warn() { echo "\033[1;33m[⚠]\033[0m $1"; }
log_err() { echo "\033[1;31m[✗]\033[0m $1" >&2; }

# ensure .env file exists
if [ ! -f ".env" ]; then
  log_warn ".env-Datei nicht gefunden. Erstelle aus .env.example..."
  cp .env.example .env
fi

# check poetry
if ! command -v poetry >/dev/null 2>&1; then
  log_err "Poetry nicht installiert"
  exit 1
fi

# check node/npm
if ! command -v node >/dev/null 2>&1; then
  log_err "Node.js nicht installiert"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  log_err "npm nicht installiert"
  exit 1
fi

# check docker
if ! command -v docker >/dev/null 2>&1; then
  log_err "Docker nicht installiert"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  log_err "Docker-Daemon nicht erreichbar"
  exit 1
fi

log_ok "Umgebungsprüfungen abgeschlossen"
