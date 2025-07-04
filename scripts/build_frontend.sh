#!/usr/bin/env sh
# Build React frontend with safety checks
set -e

log_info() { echo "\033[1;34m[...]\033[0m $1"; }
log_ok() { echo "\033[1;32m[✓]\033[0m $1"; }
log_err() { echo "\033[1;31m[✗]\033[0m $1" >&2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/../frontend/agent-ui"
TARGET_DIST="$SCRIPT_DIR/../frontend/dist"

cd "$FRONTEND_DIR"

if [ ! -f package.json ]; then
  log_err "package.json nicht gefunden"
  exit 1
fi

log_info "Installiere Node-Abh\xC3\xA4ngigkeiten"
npm ci

log_info "Baue Frontend"
npm run build

OUT_DIR="dist"
mkdir -p "$TARGET_DIST"
if [ ! "$(realpath ../dist 2>/dev/null || echo '')" = "$(realpath "$TARGET_DIST")" ]; then
  if [ ! "$(realpath "$OUT_DIR")" = "$(realpath "$TARGET_DIST")" ]; then
    cp -r "$OUT_DIR"/* "$TARGET_DIST"/
  else
    echo "[skip] source and destination are identical"
  fi
else
  echo "[skip] source and destination are identical"
fi

log_ok "Frontend gebaut"
