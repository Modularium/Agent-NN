#!/bin/bash
# -*- coding: utf-8 -*-

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FRONTEND_DIR="$SCRIPT_DIR/../../frontend/agent-ui"
TARGET_DIST="$SCRIPT_DIR/../../frontend/dist"

build_frontend() {
  log_info "Baue Frontend"
  cd "$FRONTEND_DIR"
  if [ ! -f package.json ]; then
    log_err "package.json nicht gefunden"
    exit 1
  fi
  npm ci
  npm run build
  mkdir -p "$TARGET_DIST"
  OUT_DIR="dist"
  if [ "$(realpath "$OUT_DIR")" != "$(realpath "$TARGET_DIST")" ]; then
    cp -r "$OUT_DIR"/* "$TARGET_DIST"/
  fi
  log_ok "Frontend gebaut"
}
